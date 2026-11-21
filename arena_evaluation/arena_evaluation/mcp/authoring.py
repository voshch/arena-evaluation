from __future__ import annotations

import asyncio
import importlib
import pathlib
from typing import Annotated, Any

import pydantic
from arena_mcp.core import ToolError

from arena_evaluation.storage.config_layout import get_config_kind

from .base import YamlText, atomic_write_text, config_stem, domain, parse_yaml, run_path
from .catalog import config_identifier, config_names


def _described(kind: Any, description: str) -> Any:  # noqa: ANN401 - runtime Annotated type-alias factory for MCP tool schemas
    """Annotate a tool parameter with its published description."""
    return Annotated[kind, pydantic.Field(description=description)]


Overwrite = _described(bool, "replace an existing config of that name")


def _invalid(problems: list[str], *, warnings: bool = True) -> dict[str, Any]:
    """A failed validation verdict; suite and contest verdicts also carry a warnings list."""
    verdict: dict[str, Any] = {"valid": False, "problems": problems}
    if warnings:
        verdict["warnings"] = []
    return verdict


def _parser(module: str, attribute: str, what: str) -> Any:  # noqa: ANN401 - lazy import returns Suite or Contest parser class
    """Import a parser lazily; a missing dependency becomes a hinted ToolError instead of ImportError."""
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise ToolError(
            f"the {what} parser is not available in this build",
            hint="install the arena_evaluation benchmark layer, or validate the declaration on the side stream",
        ) from exc


def world_names() -> list[str]:
    """Names of every local world, for the dashboard's spawn options."""
    from arena_simulation_setup.tree.World import WorldIdentifier

    return sorted({item.shortname for item in WorldIdentifier.listall() if not item.shortname.startswith(".")})


def _check(kind: str, data: object) -> dict[str, Any]:
    """Validate a suite or contest with the runner's own parser and report its shape."""
    accepted: type | tuple[type, ...] = dict if kind == "suite" else (list, dict)
    expected = "a mapping with stages" if kind == "suite" else "a list or a mapping"
    errors: tuple[type[Exception], ...] = (LookupError, ValueError, TypeError) if kind == "suite" else (KeyError, ValueError, TypeError)
    if not isinstance(data, accepted):
        return _invalid([f"a {kind} is {expected}, got {type(data).__name__}"])
    try:
        parsed = _parser("arena_evaluation.benchmark.config", kind.capitalize(), kind).parse("_validate", data)
    except errors as exc:
        return _invalid([f"{type(exc).__name__}: {exc}"])
    if kind == "suite":
        return {
            "valid": True,
            "n_stages": len(parsed.stages),
            "stages": [stage.name for stage in parsed.stages],
            "references": parsed.references,
            "launch": parsed.launch_args,
            "problems": [],
            "warnings": [],
            "side_stream": ("task-mode, world, robot, model and driver availability is checked by evaluation_preflight, episode_task_modes and the robots/worlds/planners tools"),
        }
    return {
        "valid": True,
        "n_contestants": len(parsed.contestants),
        "contestants": [{"name": entry.name, "args": entry.args, "parts": entry.parts} for entry in parsed.contestants],
        "problems": [],
        "warnings": [],
        "side_stream": ("the nav2 planner preflight and driver registration are checked by evaluation_preflight on a run, not by this schema validation"),
    }


def check_suite(data: object) -> dict[str, Any]:
    """Suite.parse verdict; mode, world, robot and model availability runs on the side stream."""
    return _check("suite", data)


def check_contest(data: object) -> dict[str, Any]:
    """Contest.parse verdict; the planner preflight and driver registration run on the side stream."""
    return _check("contest", data)


def check_manifest(data: object) -> dict[str, Any]:
    from arena_evaluation.presentation.viz_manifest import VizManifest

    if not isinstance(data, dict):
        return _invalid([f"a manifest is a mapping, got {type(data).__name__}"], warnings=False)
    try:
        manifest = VizManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        return _invalid([str(exc)], warnings=False)
    for plot in manifest.plots:
        if not plot.uses and plot.data is None:
            return _invalid(
                [f"plot {plot.id!r} has no uses or data specification"],
                warnings=False,
            )
    return {
        "valid": True,
        "n_plots": len(manifest.plots),
        "n_groups": len(manifest.groups),
        "problems": [],
    }


def write_config(kind: str, name: str, text: str, *, overwrite: bool = False) -> pathlib.Path:
    """Write a suite, contest or manifest to the write target its resolver chain names, as <name>/<kind>.yaml."""
    stem = config_stem(name)
    if stem in config_names(kind) and not overwrite:
        raise ToolError(
            f"a {kind} named {stem!r} already exists",
            hint="pick another name or pass overwrite=true",
        )
    target = asyncio.run(config_identifier(kind)(name=stem).resolve_write_path())
    file_kind = "report" if kind == "manifest" else kind
    path = target / get_config_kind(file_kind).filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


_CHECKS = {"suite": check_suite, "contest": check_contest}
_CREATE_HINTS = {
    "suite": "evaluation_config_schema kind=suite shows the schema",
    "contest": "planners_list lists the planners",
}


def _create(kind: str, name: str, yaml_content: str, *, overwrite: bool) -> dict[str, Any]:
    """Validate and write one suite or contest, returning its path and verdict."""
    verdict = _CHECKS[kind](parse_yaml(yaml_content, kind))
    if not verdict["valid"]:
        raise ToolError(
            f"{kind} does not validate: " + "; ".join(verdict["problems"]),
            hint=_CREATE_HINTS[kind],
        )
    return {
        "path": str(write_config(kind, name, yaml_content, overwrite=overwrite)),
        **verdict,
    }


@domain.tool(read_only=True)
def evaluation_validate_suite(yaml_content: YamlText) -> dict[str, Any]:
    """Check a suite with the runner's Suite parser; task-mode, world, robot and model availability is a side-stream check."""
    return check_suite(parse_yaml(yaml_content, "suite"))


@domain.tool(read_only=True)
def evaluation_validate_contest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a contest with the runner's Contest parser; the nav2 preflight and driver registration are side-stream checks."""
    return check_contest(parse_yaml(yaml_content, "contest"))


@domain.tool(read_only=True)
def evaluation_validate_manifest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a report manifest against the VizManifest schema."""
    return check_manifest(parse_yaml(yaml_content, "manifest"))


@domain.tool(read_only=False, destructive=True, idempotent=False)
def evaluation_create_suite(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a suite (evaluation_config_schema and episode_task_modes give its keys) and write it where the runner resolves it by name."""
    return _create("suite", name, yaml_content, overwrite=overwrite)


@domain.tool(read_only=False, destructive=True, idempotent=False)
def evaluation_create_contest(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a contest (planners_list gives the planner names) and write it where the runner resolves it by name."""
    return _create("contest", name, yaml_content, overwrite=overwrite)


@domain.tool(read_only=False, destructive=True, idempotent=False)
def evaluation_create_manifest(
    name: str,
    *,
    yaml_content: _described(str | None, "complete manifest YAML, or omit and give a template") = None,
    template: _described(str | None, "a resolvable manifest to start from (evaluation_manifests)") = None,
    run_id: _described(
        str | None,
        "write into this run's directory instead of the shared manifest root",
    ) = None,
    overwrite: Overwrite = False,
) -> dict[str, Any]:
    """Write a report manifest, from YAML or a copy of a template, into a run directory or the shared manifest root (table notes: options.notes, options.rows, options.note, options.notes_key)."""
    import yaml

    from arena_evaluation.presentation.manifest_registry import resolve_manifest

    if template is not None and yaml_content is None:
        try:
            manifest = resolve_manifest(config_stem(template))
        except FileNotFoundError as exc:
            raise ToolError(str(exc), hint="evaluation_manifests lists the templates") from exc
        manifest.name = name
        yaml_content = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    elif yaml_content is None or template is not None:
        raise ToolError(
            "pass yaml_content or template",
            hint="pass yaml_content, or template from evaluation_manifests",
        )
    verdict = check_manifest(parse_yaml(yaml_content, "manifest"))
    if not verdict["valid"]:
        raise ToolError(
            "manifest does not validate: " + "; ".join(verdict["problems"]),
            hint="fix the listed problems, or start from a template in evaluation_manifests",
        )
    if run_id is None:
        return {
            "path": str(write_config("manifest", name, yaml_content, overwrite=overwrite)),
            **verdict,
        }
    from arena_evaluation.storage.config_layout import bundle_path, run_analysis_root

    path = bundle_path(run_analysis_root(run_path(run_id)), "report", config_stem(name))
    if path.exists() and not overwrite:
        raise ToolError(f"{path} exists", hint="pick another name or pass overwrite=true")
    atomic_write_text(path, yaml_content)
    return {"path": str(path), **verdict}
