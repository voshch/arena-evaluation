from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated, Any

import pydantic
from arena_mcp.core import ToolError

from arena_evaluation.storage.config_layout import get_config_kind

from .base import YamlText, atomic_write_text, config_stem, domain, parse_yaml, run_path
from .catalog import config_identifier, config_names

Overwrite = Annotated[
    bool, pydantic.Field(description="replace an existing config of that name")
]


def _parser(module: str, attribute: str, what: str) -> Any:
    """Import a parser lazily; a missing dependency becomes a hinted ToolError instead of ImportError."""
    import importlib

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

    return sorted(
        {
            item.shortname
            for item in WorldIdentifier.listall()
            if not item.shortname.startswith(".")
        }
    )


def check_suite(data: object) -> dict[str, Any]:
    """Suite.parse verdict; mode, world, robot and model availability runs on the side stream."""
    Suite = _parser("arena_evaluation.benchmark.config", "Suite", "suite")
    if not isinstance(data, dict):
        return {
            "valid": False,
            "problems": [
                f"a suite is a mapping with stages, got {type(data).__name__}"
            ],
            "warnings": [],
        }
    try:
        suite = Suite.parse("_validate", data)
    except (LookupError, ValueError, TypeError) as exc:
        return {
            "valid": False,
            "problems": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
    return {
        "valid": True,
        "n_stages": len(suite.stages),
        "stages": [stage.name for stage in suite.stages],
        "references": suite.references,
        "launch": suite.launch_args,
        "problems": [],
        "warnings": [],
        "side_stream": (
            "task-mode, world, robot, model and driver availability is checked by "
            "evaluation_preflight, episode_task_modes and the robots/worlds/planners tools"
        ),
    }


def check_contest(data: object) -> dict[str, Any]:
    """Contest.parse verdict; the planner preflight and driver registration run on the side stream."""
    Contest = _parser("arena_evaluation.benchmark.config", "Contest", "contest")
    if not isinstance(data, list | dict):
        return {
            "valid": False,
            "problems": [
                f"a contest is a list or a mapping, got {type(data).__name__}"
            ],
            "warnings": [],
        }
    try:
        contest = Contest.parse("_validate", data)
    except (KeyError, ValueError, TypeError) as exc:
        return {
            "valid": False,
            "problems": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }
    return {
        "valid": True,
        "n_contestants": len(contest.contestants),
        "contestants": [
            {"name": entry.name, "args": entry.args, "parts": entry.parts}
            for entry in contest.contestants
        ],
        "problems": [],
        "warnings": [],
        "side_stream": (
            "the nav2 planner preflight and driver registration are checked by "
            "evaluation_preflight on a run, not by this schema validation"
        ),
    }


def check_manifest(data: object) -> dict[str, Any]:
    from arena_evaluation.presentation.viz_manifest import VizManifest

    if not isinstance(data, dict):
        return {
            "valid": False,
            "problems": [f"a manifest is a mapping, got {type(data).__name__}"],
        }
    try:
        manifest = VizManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        return {"valid": False, "problems": [str(exc)]}
    for plot in manifest.plots:
        if not plot.uses and plot.data is None:
            return {
                "valid": False,
                "problems": [f"plot {plot.id!r} has no uses or data specification"],
            }
    return {
        "valid": True,
        "n_plots": len(manifest.plots),
        "n_groups": len(manifest.groups),
        "problems": [],
    }


def write_config(
    kind: str, name: str, text: str, *, overwrite: bool = False
) -> pathlib.Path:
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
def evaluation_create_suite(
    name: str, yaml_content: YamlText, overwrite: Overwrite = False
) -> dict[str, Any]:
    """Validate a suite (evaluation_config_schema and episode_task_modes give its keys) and write it where the runner resolves it by name."""
    verdict = check_suite(parse_yaml(yaml_content, "suite"))
    if not verdict["valid"]:
        raise ToolError(
            "suite does not validate: " + "; ".join(verdict["problems"]),
            hint="evaluation_config_schema kind=suite shows the schema",
        )
    return {
        "path": str(write_config("suite", name, yaml_content, overwrite=overwrite)),
        **verdict,
    }


@domain.tool(read_only=False, destructive=True, idempotent=False)
def evaluation_create_contest(
    name: str, yaml_content: YamlText, overwrite: Overwrite = False
) -> dict[str, Any]:
    """Validate a contest (planners_list gives the planner names) and write it where the runner resolves it by name."""
    verdict = check_contest(parse_yaml(yaml_content, "contest"))
    if not verdict["valid"]:
        raise ToolError(
            "contest does not validate: " + "; ".join(verdict["problems"]),
            hint="planners_list lists the planners",
        )
    return {
        "path": str(write_config("contest", name, yaml_content, overwrite=overwrite)),
        **verdict,
    }


@domain.tool(read_only=False, destructive=True, idempotent=False)
def evaluation_create_manifest(
    name: str,
    *,
    yaml_content: Annotated[
        str | None,
        pydantic.Field(
            description="complete manifest YAML, or omit and give a template"
        ),
    ] = None,
    template: Annotated[
        str | None,
        pydantic.Field(
            description="a resolvable manifest to start from (evaluation_manifests)"
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        pydantic.Field(
            description="write into this run's directory instead of the shared manifest root"
        ),
    ] = None,
    overwrite: Overwrite = False,
) -> dict[str, Any]:
    """Write a report manifest, from YAML or a copy of a template, into a run directory or the shared manifest root (table notes: options.notes, options.rows, options.note, options.notes_key)."""
    import yaml

    from arena_evaluation.presentation.manifest_registry import resolve_manifest

    if template is not None and yaml_content is None:
        try:
            manifest = resolve_manifest(config_stem(template))
        except FileNotFoundError as exc:
            raise ToolError(
                str(exc), hint="evaluation_manifests lists the templates"
            ) from exc
        manifest.name = name
        yaml_content = yaml.safe_dump(
            manifest.model_dump(exclude_none=True), sort_keys=False
        )
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
            "path": str(
                write_config("manifest", name, yaml_content, overwrite=overwrite)
            ),
            **verdict,
        }
    from arena_evaluation.storage.config_layout import bundle_path, run_analysis_root

    path = bundle_path(run_analysis_root(run_path(run_id)), "report", config_stem(name))
    if path.exists() and not overwrite:
        raise ToolError(
            f"{path} exists", hint="pick another name or pass overwrite=true"
        )
    atomic_write_text(path, yaml_content)
    return {"path": str(path), **verdict}
