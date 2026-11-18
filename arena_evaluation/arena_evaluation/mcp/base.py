from __future__ import annotations

import functools
import inspect
import json
import pathlib
import re
import typing
from collections.abc import Callable
from typing import Annotated

import pydantic
from arena_mcp.core import Domain, ToolError
from mcp.server.mcpserver import Audio, Image, MCPServer
from mcp.types import CallToolResult

from arena_evaluation.storage.atomic import atomic_write_text as atomic_write_text
from arena_evaluation.storage.atomic import file_lock as file_lock


def _return_hint(fn: Callable[..., object]) -> object:
    """Return the resolved return type annotation of ``fn``, or None."""
    try:
        return typing.get_type_hints(fn).get("return")
    except (NameError, TypeError, AttributeError):
        return fn.__annotations__.get("return")


def _is_structured_return(fn: Callable[..., object]) -> bool:
    """Return True when ``fn`` returns a dict or Pydantic BaseModel subclass."""
    ret = _return_hint(fn)
    if ret is dict or typing.get_origin(ret) is dict:
        return True
    return isinstance(ret, type) and issubclass(ret, pydantic.BaseModel)


def _returns_call_tool_result(fn: Callable[..., object]) -> bool:
    """Return True when ``fn`` returns a raw MCP CallToolResult."""
    ret = _return_hint(fn)
    return isinstance(ret, type) and issubclass(ret, CallToolResult)


def _pack_result(result: object) -> object:
    """Serialize list/tuple results without media blocks into one JSON text block."""
    if isinstance(result, list | tuple) and not any(
        isinstance(item, Image | Audio) for item in result
    ):
        return json.dumps(result, indent=2, default=str)
    return result


def _packed_tool(fn: Callable[..., object]) -> Callable[..., object]:
    """Wrap ``fn`` so list results serialize as JSON while preserving metadata."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: object, **kwargs: object) -> object:
            return _pack_result(await fn(*args, **kwargs))

        return run_async

    @functools.wraps(fn)
    def run(*args: object, **kwargs: object) -> object:
        return _pack_result(fn(*args, **kwargs))

    return run


class EvaluationDomain(Domain):
    """Domain subclass that publishes Pydantic BaseModel output schemas on MCPServer."""

    def install(self, server: MCPServer) -> None:
        """Register evaluation tools with structured output schemas for dicts and BaseModels."""
        for spec in self.tools:
            structured = _is_structured_return(spec.fn)
            raw_result = _returns_call_tool_result(spec.fn)
            fn = spec.fn if (structured or raw_result) else _packed_tool(spec.fn)
            server.add_tool(
                fn,
                name=spec.name,
                title=spec.title,
                annotations=spec.annotations,
                structured_output=True if structured else False,
            )
        for resource in self.resources:
            server.resource(
                resource.uri,
                mime_type=resource.mime_type,
                description=resource.description,
            )(resource.fn)
        for prompt in self.prompts:
            server.prompt(name=prompt.name, description=prompt.description)(prompt.fn)


domain = EvaluationDomain(
    "evaluation",
    """Benchmarks and their analysis: suites, contests and report manifests, `arena evaluation benchmark` runs and `arena planners test` preflights, processing, reports, plots, recording control and metrics analysis.
    A run lives in $ARENA_DATA_DIR/benchmarks/<run_id>, run tools take that run_id (evaluation_runs lists them, evaluation_status reads one, shared runs included).
    Benchmarks, preflights, processing, reports, extraction, plots and acoustic renders run as jobs: poll jobs_status, read jobs_log, stop a benchmark with evaluation_stop.
    Catalog tools list suites, contests, report manifests and metrics, validate tools run the runner's own parsers. evaluation_aggregate groups by one or more dimensions, evaluation_summarize carries the run's summary columns, and evaluation_compare_planners(n=...) returns the top contestants (the merged query/read/top surface). Task modes, worlds, scenarios, robots and planners come from the episode, worlds, robots and planners domains.""",
    requires=("polars", "yaml", "arena_simulation_setup", "arena_runtime"),
)

RunId = Annotated[
    str,
    pydantic.Field(
        description="benchmark run id, a directory under $ARENA_DATA_DIR/benchmarks, evaluation_runs lists them"
    ),
]
OptionalRunId = Annotated[
    str | None,
    pydantic.Field(description="benchmark run id, the most recent run when omitted"),
]
YamlText = Annotated[str, pydantic.Field(description="complete YAML document")]

_COMPONENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def component(name: str, what: str = "name") -> str:
    """Return ``name`` when it is one safe path component, else raise ``ToolError``."""
    if not _COMPONENT.fullmatch(name):
        raise ToolError(
            f"{what} {name!r} must be one path component of letters, digits, '.', '_' or '-', not starting with a dot",
            hint="pass one path component, e.g. dwb or bench-20260101-120000",
        )
    return name


def data_root() -> pathlib.Path:
    """Return the benchmarks root the runner writes to by default."""
    from arena_evaluation.storage.data_root import benchmarks_root

    return benchmarks_root()


def run_path(run_id: str | None) -> pathlib.Path:
    """Return the directory of ``run_id``, or of the most recent run when ``None``."""
    root = data_root()
    if run_id is None:
        from arena_evaluation.storage.data_root import latest_benchmark

        latest = latest_benchmark(root)
        if latest is None:
            raise ToolError(
                f"no benchmark runs in {root}", hint="evaluation_benchmark starts one"
            )
        return latest
    path = root / component(run_id, "run_id")
    if not path.is_dir():
        raise ToolError(
            f"no run {run_id!r} in {root}", hint="evaluation_runs lists the runs"
        )
    return path


def parse_yaml(text: str, what: str) -> object:
    """Parse ``text`` as YAML, raising ``ToolError`` on empty or invalid input."""
    import yaml

    if not text.strip():
        raise ToolError(
            f"empty {what}",
            hint=f"pass the {what} YAML document, not an empty string",
        )
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolError(
            f"{what} is not valid YAML: {exc}",
            hint=f"fix the YAML syntax of the {what}, or start from evaluation_manifests",
        ) from exc


def config_stem(name: str) -> str:
    """Return the validated config stem without a trailing ``.yaml`` suffix."""
    return component(name.removesuffix(".yaml"))


def is_inline(value: str) -> bool:
    """Return True when ``value`` looks like an inline YAML flow mapping or sequence."""
    return value.lstrip().startswith(("{", "["))


def assert_safe_analysis_write(
    run_dir: pathlib.Path, target: pathlib.Path
) -> pathlib.Path:
    """Refuse writes aimed at ``episodes/`` or published root tables; allow ``analysis/`` and ``reports/``."""
    resolved_run = run_dir.resolve()
    resolved_target = target.resolve()
    try:
        rel = resolved_target.relative_to(resolved_run)
    except ValueError as exc:
        raise ToolError(
            f"refusing write outside run directory: {target}",
            hint="agent writes belong under <run>/analysis/",
        ) from exc
    if not rel.parts or rel.parts[0] == "episodes":
        raise ToolError(
            f"refusing write to protected published data under episodes/: {rel}",
            hint="agent writes belong under <run>/analysis/",
        )
    if rel.parts[0] not in ("analysis", "reports"):
        raise ToolError(
            f"refusing write to protected published file: {rel}",
            hint="agent writes belong under <run>/analysis/ or <run>/reports/",
        )
    return resolved_target


def append_audit_event(
    run_dir: pathlib.Path,
    *,
    role: str,
    tool: str,
    args: dict[str, object],
    produced_ids: list[str] | None = None,
) -> None:
    """Append one JSONL audit entry to ``<run>/analysis/audit.jsonl`` under a file lock."""
    import datetime
    import hashlib
    import json

    audit_path = run_dir / "analysis" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    args_hash = hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    entry = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "role": role,
        "tool": tool,
        "args_hash": args_hash,
        "produced_ids": produced_ids or [],
    }
    with file_lock(audit_path.with_suffix(".jsonl.lock")):
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


__all__ = [
    "OptionalRunId",
    "RunId",
    "YamlText",
    "append_audit_event",
    "assert_safe_analysis_write",
    "atomic_write_text",
    "component",
    "config_stem",
    "data_root",
    "domain",
    "file_lock",
    "is_inline",
    "parse_yaml",
    "run_path",
]
