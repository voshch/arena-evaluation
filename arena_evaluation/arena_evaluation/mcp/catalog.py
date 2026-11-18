from __future__ import annotations

import asyncio
import pathlib
from typing import TYPE_CHECKING, Annotated, Any, Literal

import attrs
import pydantic
from arena_mcp.core import ToolError
from arena_mcp.core.msgs import jsonable

from arena_evaluation.storage.config_layout import (
    CONFIGS_DIR_NAME,
    bundle_path,
    get_config_kind,
    run_analysis_root,
    share_dir,
    source_tree_dir,
)

from .base import config_stem, domain, run_path

if TYPE_CHECKING:
    from arena_simulation_setup.tree import AssetIdentifier

ConfigKind = Literal["suite", "contest", "manifest"]
Network = Annotated[
    bool,
    pydantic.Field(description="also list what the network buckets hold, slower"),
]


def config_identifier(kind: str) -> type[AssetIdentifier]:
    """Return the asset identifier class for ``suite``, ``contest``, or ``manifest``."""
    from arena_evaluation.benchmark.tree import (
        ContestIdentifier,
        ManifestIdentifier,
        SuiteIdentifier,
    )

    return {
        "suite": SuiteIdentifier,
        "contest": ContestIdentifier,
        "manifest": ManifestIdentifier,
    }[kind]


def config_names(kind: str, *, network: bool = False) -> list[str]:
    """Return sorted config names for ``kind``."""
    return sorted(
        {item.shortname for item in config_identifier(kind).listall(network=network)}
    )


def config_file(kind: str, name: str) -> pathlib.Path:
    """Resolve a suite, contest, or manifest name to its YAML file path."""
    identifier = config_identifier(kind)(name=config_stem(name))
    try:
        path = asyncio.run(identifier.resolve_path())
    except FileNotFoundError as exc:
        raise ToolError(
            f"no {kind} named {name!r}",
            hint=f"evaluation_{kind}s lists them, with network=true for the buckets",
        ) from exc
    filename = get_config_kind("report" if kind == "manifest" else kind).filename
    return path / filename if path.is_dir() else path


@domain.tool(read_only=True)
def evaluation_suites(network: Network = False) -> dict[str, Any]:
    """Suite names the runner resolves: directory bundles and with network the benchmark buckets."""
    return {"suites": config_names("suite", network=network)}


@domain.tool(read_only=True)
def evaluation_contests(network: Network = False) -> dict[str, Any]:
    """Contest names the runner resolves: directory bundles and with network the benchmark buckets."""
    return {"contests": config_names("contest", network=network)}


def run_manifests(path: pathlib.Path) -> list[str]:
    """Return valid custom report manifest names under ``<run>/analysis/reports/<name>/report.yaml``."""
    import yaml

    from arena_evaluation.presentation.viz_manifest import VizManifest

    reports_dir = run_analysis_root(path) / get_config_kind("report").folder
    if not reports_dir.is_dir():
        return []
    found: list[str] = []
    report_filename = get_config_kind("report").filename
    for entry in sorted(reports_dir.iterdir()):
        if not entry.is_dir():
            continue
        candidate = entry / report_filename
        if not candidate.is_file():
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "plots" in data:
                VizManifest.model_validate(data)
                found.append(entry.name)
        except (OSError, yaml.YAMLError, pydantic.ValidationError):
            continue
    return found


@domain.tool(read_only=True)
def evaluation_manifests(
    network: Network = False,
    run_id: Annotated[
        str | None,
        pydantic.Field(
            description="also list the custom manifests written into this run"
        ),
    ] = None,
) -> dict[str, Any]:
    """Report manifest names: the resolvable ones and, for a run, the custom ones in its analysis directory."""
    out: dict[str, Any] = {"manifests": config_names("manifest", network=network)}
    if run_id is not None:
        out["run_manifests"] = run_manifests(run_path(run_id))
    return out


@domain.tool(read_only=True)
def evaluation_config_template(
    kind: ConfigKind,
    name: str,
    run_id: Annotated[
        str | None,
        pydantic.Field(
            description="for a manifest, look in this run's analysis directory first"
        ),
    ] = None,
) -> dict[str, Any]:
    """The full YAML of a suite, contest or report manifest by name, and where it resolved from."""
    stem = config_stem(name)
    if kind == "manifest" and run_id is not None:
        local = bundle_path(run_analysis_root(run_path(run_id)), "report", stem)
        if local.is_file():
            return {
                "kind": kind,
                "name": stem,
                "path": str(local),
                "content": local.read_text(encoding="utf-8"),
            }
    path = config_file(kind, name)
    return {
        "kind": kind,
        "name": stem,
        "path": str(path),
        "content": path.read_text(encoding="utf-8"),
    }


def _readme_section(word: str) -> str:
    """Return matching sections from ``configs/README.md``."""
    for base in (share_dir(), source_tree_dir()):
        if base is None:
            continue
        readme = base / CONFIGS_DIR_NAME / "README.md"
        if readme.is_file():
            sections = readme.read_text(encoding="utf-8").split("\n## ")
            return "\n\n".join(
                f"## {section}"
                for section in sections[1:]
                if word.lower() in section.splitlines()[0].lower()
            )
    return ""


@domain.tool(read_only=True)
def evaluation_config_schema(
    kind: Literal["suite", "contest"],
) -> dict[str, Any]:
    """The suite or contest schema from the runner's config models, with the configs README section that documents it."""
    from arena_runtime.env.constants import Constants

    from arena_evaluation.benchmark import config

    if kind == "suite":
        stage = config.Suite.Stage
        out: dict[str, Any] = {
            "suite_keys": {
                "stages": "list of stages (required)",
                "launch": "launch args for the whole run",
                "task_layer": "task runtime name, default $ARENA_TASK_LAYER or task_generator",
                "references": f"bool, default {config.Suite._field_defaults['references']}",
                "metrics": f"metric overrides, keys {sorted(config._SUITE_METRIC_KEYS)}",
            },
            "stage_fields": {
                field: {
                    "type": str(stage.__annotations__.get(field)),
                    **(
                        {"default": jsonable(stage._field_defaults[field])}
                        if field in stage._field_defaults
                        else {}
                    ),
                }
                for field in stage._fields
            },
            "stage_defaults": {
                "timeout": f"{config._DEFAULT_TIMEOUT_S}s sim time",
                "seed": "hash of the stage fields except config, so stages with different names differ, set it explicitly to share starts and goals; episode i runs with seed + i",
            },
            "launch_rejects": {
                "keys": sorted(config._STAGE_OWNED_KEYS),
                "prefixes": list(config._STAGE_OWNED_PREFIXES),
            },
            "human_blocks": {
                "namespaces": sorted(Constants.HUMAN_PARAM_NAMESPACES.values()),
                "reserved": Constants.HUMAN_PARAM_RESERVED,
            },
            "task_modes": "episode_task_modes source=registry gives every config.<mode> parameter",
        }
    else:
        out = {
            "contestant_fields": [
                field.name for field in attrs.fields(config.Contestant)
            ],
            "forms": [
                "list of {name, <args>}",
                "mapping {name, description, contestants: [...]}",
                "sweep mapping: list values are axes, scalars are shared, cap dicts may hold axes",
            ],
            "caps_and_drivers": "planners_list names the planners, robots_catalog the robots and their caps",
        }
    out["readme"] = _readme_section(kind)
    return out


@domain.resource(
    "arena://evaluation/suites",
    mime_type="application/json",
    description="Suite names the runner resolves locally",
)
def suites_resource() -> list[str]:
    return config_names("suite")


@domain.resource(
    "arena://evaluation/contests",
    mime_type="application/json",
    description="Contest names the runner resolves locally",
)
def contests_resource() -> list[str]:
    return config_names("contest")


@domain.resource(
    "arena://evaluation/manifests",
    mime_type="application/json",
    description="Report manifest names resolvable locally",
)
def manifests_resource() -> list[str]:
    return config_names("manifest")
