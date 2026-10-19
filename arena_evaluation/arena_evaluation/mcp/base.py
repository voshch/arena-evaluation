from __future__ import annotations

import pathlib
import re
from typing import Annotated

import pydantic

from arena_mcp.core import Domain, ToolError

domain = Domain(
    "evaluation",
    """Benchmarks and their analysis: suites, contests and report manifests, `arena evaluation benchmark` runs and `arena planners test` preflights, processing, reports, plots, recording control and metrics analysis.
    A run lives in $ARENA_DATA_DIR/benchmarks/<run_id>, run tools take that run_id (evaluation_runs lists them, evaluation_status reads one, shared runs included).
    Benchmarks, preflights, processing, reports, extraction, plots and acoustic renders run as jobs: poll jobs_status, read jobs_log, stop a benchmark with evaluation_stop.
    Catalog tools read the real registries and resolvers (task modes, metrics, drivers, planners, worlds, suites, contests, manifests), validate tools run the runner's own parsers.""",
    requires=("polars", "yaml", "arena_simulation_setup", "task_generator"),
)

RunId = Annotated[str, pydantic.Field(description="benchmark run id, a directory under $ARENA_DATA_DIR/benchmarks, evaluation_runs lists them")]
OptionalRunId = Annotated[str | None, pydantic.Field(description="benchmark run id, the most recent run when omitted")]
YamlText = Annotated[str, pydantic.Field(description="complete YAML document")]

_COMPONENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")


def component(name: str, what: str = "name") -> str:
    """name when it is one safe path component, else a ToolError."""
    if not _COMPONENT.fullmatch(name):
        raise ToolError(f"{what} {name!r} must be one path component of letters, digits, '.', '_' or '-', not starting with a dot")
    return name


def data_root() -> pathlib.Path:
    """The benchmarks root the runner writes to by default."""
    from arena_evaluation.storage.data_root import benchmarks_root

    return benchmarks_root()


def run_path(run_id: str | None) -> pathlib.Path:
    """The directory of run_id, or of the most recent run when None."""
    root = data_root()
    if run_id is None:
        from arena_evaluation.storage.data_root import latest_benchmark

        latest = latest_benchmark(root)
        if latest is None:
            raise ToolError(f"no benchmark runs in {root}", hint="evaluation_benchmark starts one")
        return latest
    path = root / component(run_id, "run_id")
    if not path.is_dir():
        raise ToolError(f"no run {run_id!r} in {root}", hint="evaluation_runs lists the runs")
    return path


def parse_yaml(text: str, what: str) -> object:
    """The YAML document in text, a ToolError when it does not parse."""
    import yaml

    if not text.strip():
        raise ToolError(f"empty {what}")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolError(f"{what} is not valid YAML: {exc}") from exc


def config_stem(name: str) -> str:
    return component(name.removesuffix(".yaml"))


def is_inline(value: str) -> bool:
    return value.lstrip().startswith(("{", "["))
