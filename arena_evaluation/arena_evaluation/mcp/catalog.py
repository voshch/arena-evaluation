from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import pathlib
import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import attrs
import pydantic

from arena_mcp.core import ToolError
from arena_mcp.core.msgs import jsonable

from .base import OptionalRunId, component, config_stem, domain, run_path

if TYPE_CHECKING:
    import enum

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_simulation_setup.tree import AssetIdentifier, DomainAssetIdentifier
    from arena_simulation_setup.tree.World import WorldView
    from rcl_interfaces.msg import ParameterDescriptor
    from task_generator.tasks.registry import TaskModeRegistry

    from arena_evaluation.benchmark.config import Suite

Axis = Literal["tm_robots", "tm_obstacles", "tm_modules"]
ConfigKind = Literal["suite", "contest", "manifest"]
Network = Annotated[bool, pydantic.Field(description="also list what the network buckets hold, slower")]

PARAM_TYPES = {1: "bool", 2: "int", 3: "double", 4: "string", 5: "byte_array", 6: "bool_array", 7: "int_array", 8: "double_array", 9: "string_array"}


def mode_registries() -> dict[str, TaskModeRegistry]:
    """The three task-mode registries, keyed by their suite stage field."""
    for package in ("robots", "obstacles", "modules"):
        importlib.import_module(f"task_generator.tasks.{package}")
    from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES

    return {"tm_robots": ROBOTS_MODES, "tm_obstacles": OBSTACLES_MODES, "tm_modules": MODULE_MODES}


class _Declarations:
    """Collects what a task-mode schema declares, standing in for the node it normally declares on."""

    def __init__(self) -> None:
        self.rosparam = self
        self.declared: list[tuple[str, object, ParameterDescriptor]] = []

    def declare_forward(self, name: str, value: object, *, descriptor: ParameterDescriptor) -> None:
        self.declared.append((name, value, descriptor))


def _constraint_tokens(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in filter(None, text.split(";")):
        key, sep, value = token.partition(":")
        out[key] = (value.split(",") if key == "enum" else value) if sep else True
    return out


def mode_params(registry: TaskModeRegistry, key: enum.Enum) -> list[dict[str, Any]]:
    """The parameters a registered task mode declares, with defaults, descriptions and constraints."""
    meta = registry.meta(key)
    if meta.schema is None:
        return []
    collector = _Declarations()
    meta.schema(cast("ROSParamServer", collector), meta.namespace)
    prefix = f"{meta.namespace}."
    params = []
    for name, value, descriptor in collector.declared:
        entry: dict[str, Any] = {"name": name.removeprefix(prefix), "type": PARAM_TYPES.get(descriptor.type, descriptor.type), "default": jsonable(value), "description": descriptor.description}
        entry.update(_constraint_tokens(descriptor.additional_constraints))
        for bound in (*descriptor.floating_point_range, *descriptor.integer_range):
            entry["min"], entry["max"] = jsonable(bound.from_value), jsonable(bound.to_value)
        params.append(entry)
    return params


def _config_block(params: list[dict[str, Any]]) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for param in params:
        parts = param["name"].split(".")
        default = param["default"]
        if parts[-1] == "n" and param.get("range") == "int_pair" and len(parts) > 1 and isinstance(default, list) and len(default) == 2:
            parts, default = parts[:-1], {"min": default[0], "max": default[1]}
        node = block
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        leaf = node.setdefault(parts[-1], {}) if isinstance(default, dict) else None
        if leaf is not None:
            leaf.update(default)
        else:
            node[parts[-1]] = default
    return block


def _mode_purpose(registry: TaskModeRegistry, key: enum.Enum) -> str:
    try:
        return inspect.getdoc(registry.get(key)) or ""
    except ImportError as exc:
        return f"the implementation does not import here: {exc}"


def describe_mode(axis: str, registry: TaskModeRegistry, key: enum.Enum) -> dict[str, Any]:
    params = mode_params(registry, key)
    mode = key.value
    for param in params:
        param["config_key"] = f"config.{mode}.{param['name']}"
    return {
        "mode": mode,
        "axis": axis,
        "namespace": str(registry.meta(key).namespace),
        "purpose": _mode_purpose(registry, key),
        "params": params,
        "stage_example": {axis: mode, "config": {mode: _config_block(params)}} if params else {axis: mode},
    }


def domain_asset_names(identifier: type[DomainAssetIdentifier], *, network: bool = False) -> list[str]:
    """Names of a domain asset kind the way task modes reference them."""
    from arena_simulation_setup import DOMAIN_DEFAULT

    return sorted({item.name if item.domain == DOMAIN_DEFAULT else item.shortname for item in identifier.listall(network=network)})


def world_names(*, network: bool = False) -> list[str]:
    from arena_simulation_setup.tree.World import WorldIdentifier

    return sorted({item.shortname for item in WorldIdentifier.listall(network=network) if not item.shortname.startswith(".")})


def world_resolves(name: str, known: set[str]) -> bool:
    """True when every world of a possibly composite, level-filtered world name is known."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import composite_parts

    return all(WorldIdentifier.split_filter(part)[0] in known for part in composite_parts(name))


def config_identifier(kind: str) -> type[AssetIdentifier]:
    from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier

    return {"suite": SuiteIdentifier, "contest": ContestIdentifier, "manifest": ManifestIdentifier}[kind]


def config_names(kind: str, *, network: bool = False) -> list[str]:
    return sorted({item.shortname for item in config_identifier(kind).listall(network=network)})


def config_file(kind: str, name: str) -> pathlib.Path:
    """The YAML file a suite, contest or manifest name resolves to, fetched when only a bucket has it."""
    identifier = config_identifier(kind)(name=config_stem(name))
    try:
        path = asyncio.run(identifier.resolve_path())
    except FileNotFoundError as exc:
        raise ToolError(f"no {kind} named {name!r}", hint=f"evaluation_{kind}s lists them, with network=true for the buckets") from exc
    return path / f"{kind}.yaml" if path.is_dir() else path


def suites_by_name() -> dict[str, Suite]:
    """Every local suite parsed, skipping the ones that fail to parse."""
    from arena_evaluation.benchmark.tree import SuiteIdentifier

    out: dict[str, Suite] = {}
    for name in config_names("suite"):
        try:
            out[name] = SuiteIdentifier(name=name).resolve_sync()
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            continue
    return out


def planner_catalog() -> dict[str, Any]:
    """Drivers per cap from the bringup and adapter registries, nav2 configs the runner checks, and arena_planners planners."""
    from ament_index_python.packages import get_package_share_directory
    from arena_robots.bringup import BRINGUPS
    from task_generator.tasks.robots.adapters import ADAPTERS

    for cap in ADAPTERS:
        importlib.import_module(f"task_generator.tasks.robots.adapters.{cap}")
    nav2 = pathlib.Path(get_package_share_directory("arena_robots")) / "config" / "nav2"

    def configs(sub: str, marker: str | None) -> list[str]:
        folder = nav2 / sub
        if not folder.is_dir():
            return []
        return sorted(entry.name for entry in folder.iterdir() if entry.is_dir() and (marker is None or (entry / marker).is_file()))

    out: dict[str, Any] = {
        "caps": sorted(set(BRINGUPS) | set(ADAPTERS)),
        "bringups": {cap: sorted(registry.keys()) for cap, registry in BRINGUPS.items()},
        "adapters": {cap: sorted(registry.keys()) for cap, registry in ADAPTERS.items()},
        "nav2": {
            "local_planners": configs("controllers", "controller_config.yaml"),
            "inter_planners": configs("interplanners", "interplanner_config.yaml"),
            "global_planners": configs("planners", None),
            "params_overlays": sorted(path.stem for path in (nav2 / "overlays").glob("*.yaml")) if (nav2 / "overlays").is_dir() else [],
        },
    }
    if importlib.util.find_spec("arena_planners") is not None:
        from arena_planners.resolver import list_available

        out["arena_planners"] = list_available()
    return out


def drivers(catalog: dict[str, Any], cap: str) -> set[str]:
    return set(catalog["bringups"].get(cap, ())) | set(catalog["adapters"].get(cap, ()))


@domain.tool(read_only=True)
def evaluation_task_modes() -> dict[str, Any]:
    """Task modes a suite stage can select, from the task-mode registry, with each mode's parameter namespace."""
    out: dict[str, Any] = {}
    for axis, registry in mode_registries().items():
        out[axis] = [{"mode": key.value, "namespace": str(registry.meta(key).namespace), "has_params": registry.meta(key).schema is not None} for key in sorted(registry.keys(), key=lambda key: key.value)]
    out["note"] = "tm_obstacles prompt is attached by the human simulator at runtime, it is selectable only when a human backend provides it"
    return out


@domain.tool(read_only=True)
def evaluation_describe_task_mode(
    mode: Annotated[str | None, pydantic.Field(description="mode name, e.g. random or scenario, all modes when omitted")] = None,
    axis: Annotated[Axis | None, pydantic.Field(description="restrict to one axis, needed when a name exists on several")] = None,
) -> dict[str, Any]:
    """A task mode's purpose, its parameters with defaults and constraints, and the stage config block they map to (config.<mode>.<param>)."""
    found = []
    for name, registry in mode_registries().items():
        if axis is not None and name != axis:
            continue
        for key in sorted(registry.keys(), key=lambda key: key.value):
            if mode is None or key.value == mode:
                found.append(describe_mode(name, registry, key))
    if not found:
        raise ToolError(f"no task mode {mode!r}" + (f" on {axis}" if axis else ""), hint="evaluation_task_modes lists them")
    return {"modes": found}


@domain.tool(read_only=True)
def evaluation_models(kind: Literal["human", "object"], network: Network = False) -> dict[str, Any]:
    """Pedestrian (human) or obstacle (object) model names usable in task-mode model lists, from the asset resolvers."""
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    identifier = HumanIdentifier if kind == "human" else ObjectIdentifier
    return {"kind": kind, "models": domain_asset_names(identifier, network=network), "fetch": f"arena asset pull {kind} <name> fetches a bucket model into the local cache"}


@domain.tool(read_only=True)
def evaluation_maps(network: Network = False, run_id: Annotated[str | None, pydantic.Field(description="also list the world snapshots a run recorded")] = None) -> dict[str, Any]:
    """World names the runtime resolves (local roots, imported, bundled and with network the buckets), and suite maps that do not resolve."""
    worlds = world_names(network=network)
    known = set(worlds)
    out: dict[str, Any] = {"worlds": worlds, "syntax": "a stage map may filter levels, world[1,3], or stack worlds as levels, a,b[1]"}
    suite_maps: dict[str, set[str]] = {}
    for name, suite in suites_by_name().items():
        for stage in suite.stages:
            if not world_resolves(stage.map, known):
                suite_maps.setdefault(stage.map, set()).add(name)
    out["unresolved_suite_maps"] = {world: sorted(suites) for world, suites in sorted(suite_maps.items())}
    if run_id is not None:
        snapshot = run_path(run_id) / "worlds"
        out["run_worlds"] = sorted(entry.name for entry in snapshot.iterdir() if entry.is_dir()) if snapshot.is_dir() else []
    return out


@domain.tool(read_only=True)
def evaluation_robots() -> dict[str, Any]:
    """Robot models a suite stage can name, from the robot catalog."""
    from arena_robots.Robot import RobotIdentifier

    return {"robots": sorted({item.shortname for item in RobotIdentifier.listall()}), "note": "robot: auto lets the runner pick a robot that fits the contestant"}


@domain.tool(read_only=True)
def evaluation_planners() -> dict[str, Any]:
    """What contest entries can name: caps, drivers per cap, nav2 local, inter and global planners, params overlays and arena_planners bridge planners."""
    out = planner_catalog()
    out["usage"] = {
        "nav2": "mobile: {driver: nav2, local_planner: <nav2.local_planners>, inter_planner: <nav2.inter_planners>}",
        "bridge": "mobile: {driver: drl, planner: <arena_planners name>}, vla for planners that declare goal inputs",
    }
    return out


@domain.tool(read_only=True)
def evaluation_suites(network: Network = False) -> dict[str, Any]:
    """Suite names the runner resolves: flat files, directory bundles and with network the benchmark buckets."""
    return {"suites": config_names("suite", network=network)}


@domain.tool(read_only=True)
def evaluation_contests(network: Network = False) -> dict[str, Any]:
    """Contest names the runner resolves: flat files, directory bundles and with network the benchmark buckets."""
    return {"contests": config_names("contest", network=network)}


def run_manifests(path: pathlib.Path) -> list[str]:
    """Report manifests written into a run directory, the yaml files there that validate as one."""
    import yaml

    from arena_evaluation.presentation.viz_manifest import VizManifest

    found = []
    for candidate in sorted(path.glob("*.yaml")):
        try:
            data = yaml.safe_load(candidate.read_text())
            if isinstance(data, dict) and "plots" in data:
                VizManifest.model_validate(data)
                found.append(candidate.stem)
        except (OSError, yaml.YAMLError, pydantic.ValidationError):
            continue
    return found


@domain.tool(read_only=True)
def evaluation_manifests(network: Network = False, run_id: Annotated[str | None, pydantic.Field(description="also list the custom manifests written into this run")] = None) -> dict[str, Any]:
    """Report manifest names: the resolvable ones and, for a run, the custom ones in its directory."""
    out: dict[str, Any] = {"manifests": config_names("manifest", network=network)}
    if run_id is not None:
        out["run_manifests"] = run_manifests(run_path(run_id))
    return out


@domain.tool(read_only=True)
def evaluation_config_template(kind: ConfigKind, name: str, run_id: Annotated[str | None, pydantic.Field(description="for a manifest, look in this run's directory first")] = None) -> dict[str, Any]:
    """The full YAML of a suite, contest or report manifest by name, and where it resolved from."""
    if kind == "manifest" and run_id is not None:
        local = run_path(run_id) / f"{config_stem(name)}.yaml"
        if local.is_file():
            return {"kind": kind, "name": local.stem, "path": str(local), "content": local.read_text()}
    path = config_file(kind, name)
    return {"kind": kind, "name": config_stem(name), "path": str(path), "content": path.read_text()}


def _readme_section(word: str) -> str:
    from arena_evaluation.presentation.manifest_registry import share_dir, source_tree_dir

    for base in (share_dir(), source_tree_dir()):
        if base is None:
            continue
        readme = base / "configs" / "benchmark" / "README.md"
        if readme.is_file():
            sections = readme.read_text().split("\n## ")
            return "\n\n".join(f"## {section}" for section in sections[1:] if section.lower().startswith(word))
    return ""


@domain.tool(read_only=True)
def evaluation_config_schema(kind: Literal["suite", "contest"]) -> dict[str, Any]:
    """The suite or contest schema from the runner's config models, with the configs README section that documents it."""
    from task_generator.constants import Constants

    from arena_evaluation.benchmark import config

    if kind == "suite":
        stage = config.Suite.Stage
        out: dict[str, Any] = {
            "suite_keys": {"stages": "list of stages (required)", "launch": "launch args for the whole run", "references": f"bool, default {config.Suite._field_defaults['references']}", "metrics": f"metric overrides, keys {sorted(config._SUITE_METRIC_KEYS)}"},
            "stage_fields": {field: {"type": str(stage.__annotations__.get(field)), **({"default": jsonable(stage._field_defaults[field])} if field in stage._field_defaults else {})} for field in stage._fields},
            "stage_defaults": {"timeout": f"{config._DEFAULT_TIMEOUT_S}s sim time", "seed": "hash of the stage fields except config, so stages with different names differ, set it explicitly to share starts and goals; episode i runs with seed + i"},
            "launch_rejects": {"keys": sorted(config._STAGE_OWNED_KEYS), "prefixes": list(config._STAGE_OWNED_PREFIXES)},
            "human_blocks": {"namespaces": sorted(Constants.HUMAN_PARAM_NAMESPACES.values()), "reserved": Constants.HUMAN_PARAM_RESERVED},
            "task_modes": "evaluation_describe_task_mode gives every config.<mode> key",
        }
    else:
        out = {
            "contestant_fields": [field.name for field in attrs.fields(config.Contestant)],
            "forms": ["list of {name, <args>}", "mapping {name, description, contestants: [...]}", "sweep mapping: list values are axes, scalars are shared, cap dicts may hold axes"],
            "caps_and_drivers": "evaluation_planners",
        }
    out["readme"] = _readme_section(kind)
    return out


def _bounds(points: list[tuple[float, float]]) -> dict[str, list[float]]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return {"x": [round(min(xs), 2), round(max(xs), 2)], "y": [round(min(ys), 2), round(max(ys), 2)]}


def world_view(map_name: str, run_id: str | None = None) -> tuple[WorldView, set[str] | None, str]:
    """The world a stage map names, from the run's snapshot when given, else through the world resolvers by exact name."""
    from arena_simulation_setup.tree.World import WorldIdentifier, WorldView

    bare, levels = WorldIdentifier.split_filter(map_name)
    if run_id is not None and "," not in bare:
        snapshot = run_path(run_id) / "worlds" / component(bare, "map_name")
        if snapshot.is_dir():
            return WorldView(snapshot), levels, "run snapshot"
    try:
        view = WorldIdentifier(bare).resolve_sync()
    except FileNotFoundError as exc:
        raise ToolError(f"world {map_name!r} does not resolve", hint="evaluation_maps lists the world names, with network=true for the buckets") from exc
    return view, levels, "world resolvers"


def zone_summary(view: WorldView, levels: set[str] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Zones with their bounds per level, and per level preview map metadata."""
    import yaml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        world = view.load(validate=False, level_filter=levels)
    zones = []
    previews: dict[str, Any] = {}
    for level_id, level in world.levels.items():
        for zone in level.zones:
            corners = [(corner.x, corner.y) for corner in zone.polygon]
            if corners:
                zones.append({"name": zone.name, "level": level_id, "bounds": _bounds(corners)})
        preview = view.map_yaml(level_id)
        if preview.is_file():
            meta = yaml.safe_load(preview.read_text()) or {}
            previews[level_id] = {"yaml": str(preview), "resolution": meta.get("resolution"), "origin": meta.get("origin"), "image": meta.get("image")}
    return zones, previews


@domain.tool(read_only=True)
def evaluation_inspect_map(map_name: str, run_id: OptionalRunId = None) -> dict[str, Any]:
    """A world's levels, zone bounds, preview map metadata and scenarios with robot starts, so stages and scenarios use reachable coordinates."""
    import yaml
    from arena_simulation_setup.tree.World.Scenario import ScenarioView, parse_scenario

    view, levels, source = world_view(map_name, run_id)
    zones, previews = zone_summary(view, levels)
    out: dict[str, Any] = {"map": map_name, "world_dir": str(view.path), "resolved_by": source, "levels": sorted(previews) or sorted({zone["level"] for zone in zones}), "zones": zones, "previews": previews}
    if zones:
        corners = [(x, y) for zone in zones for x in zone["bounds"]["x"] for y in zone["bounds"]["y"]]
        out["map_bounds"] = _bounds(corners)
    scenarios = []
    for identifier in sorted(view.scenario.listall(), key=lambda item: item.name):
        entry: dict[str, Any] = {"name": identifier.name}
        scenario_view = ScenarioView(view.path / "scenarios" / identifier.name)
        if pathlib.Path(scenario_view.scenario_path).is_file():
            try:
                raw = scenario_view.read()
                parse_scenario(raw)
            except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
                entry["problem"] = str(exc)
            else:
                robots = raw.get("robots") or []
                entry["robots"] = [{"start": robot.get("start"), "phases": len(robot.get("phases") or [])} for robot in robots]
                entry["humans"] = len(raw.get("humans") or [])
                entry["objects"] = len(raw.get("objects") or [])
                entry["regions"] = sorted(raw.get("regions") or {})
        scenarios.append(entry)
    out["scenarios"] = scenarios
    return out


@domain.resource("arena://evaluation/suites", mime_type="application/json", description="Suite names the runner resolves locally")
def suites_resource() -> list[str]:
    return config_names("suite")


@domain.resource("arena://evaluation/contests", mime_type="application/json", description="Contest names the runner resolves locally")
def contests_resource() -> list[str]:
    return config_names("contest")


@domain.resource("arena://evaluation/manifests", mime_type="application/json", description="Report manifest names resolvable locally")
def manifests_resource() -> list[str]:
    return config_names("manifest")


@domain.resource("arena://evaluation/maps", mime_type="application/json", description="World names the runtime resolves locally")
def maps_resource() -> list[str]:
    return world_names()


@domain.resource("arena://evaluation/robots", mime_type="application/json", description="Robot models a suite stage can name")
def robots_resource() -> list[str]:
    return evaluation_robots()["robots"]
