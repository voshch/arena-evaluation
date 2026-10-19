from __future__ import annotations

import asyncio
import pathlib
from typing import Annotated, Any

import pydantic

from arena_mcp.core import ToolError

from .base import OptionalRunId, YamlText, component, config_stem, domain, parse_yaml, run_path
from .catalog import config_identifier, config_names, domain_asset_names, drivers, mode_params, mode_registries, planner_catalog, world_names, world_resolves, world_view, zone_summary

Overwrite = Annotated[bool, pydantic.Field(description="replace an existing config of that name")]


def _leaves(block: dict[str, Any], prefix: str = "") -> list[str]:
    out = []
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and not ({"min", "max"} <= set(value)):
            out += _leaves(value, f"{name}.")
        elif isinstance(value, dict):
            out.append(f"{name}.n")
            out += _leaves({key: item for key, item in value.items() if key not in ("min", "max")}, f"{name}.")
        else:
            out.append(name)
    return out


def _at(block: dict[str, Any], dotted: str) -> object:
    node: object = block
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_suite(data: object) -> dict[str, Any]:
    """Suite.parse verdict plus warnings for task modes, config keys, worlds, robots and models the registries and resolvers do not know."""
    from arena_robots.Robot import RobotIdentifier
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
    from task_generator.constants import Constants
    from task_generator.manager.robot_manager.robots_manager import split_robot_arg

    from arena_evaluation.benchmark.config import Suite

    if not isinstance(data, dict):
        return {"valid": False, "problems": [f"a suite is a mapping with stages, got {type(data).__name__}"]}
    try:
        suite = Suite.parse("_validate", data)
    except (KeyError, ValueError, TypeError) as exc:
        return {"valid": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    registries = mode_registries()
    worlds = set(world_names())
    robots = {item.shortname for item in RobotIdentifier.listall()}
    catalogs = {"pedestrians": set(domain_asset_names(HumanIdentifier)), "objects": set(domain_asset_names(ObjectIdentifier))}
    human_blocks = {*Constants.HUMAN_PARAM_NAMESPACES.values(), Constants.HUMAN_PARAM_RESERVED}
    problems: list[str] = []
    warnings: list[str] = []
    for raw, stage in zip(data["stages"], suite.stages, strict=True):
        where = f"stage {stage.name!r}"
        selected = {"tm_robots": stage.tm_robots, "tm_obstacles": stage.tm_obstacles}
        for axis, key in selected.items():
            if key not in registries[axis]:
                problems.append(f"{where}: {axis} {key.value!r} is not registered, see evaluation_task_modes")
        if Constants.HUMAN_PARAM_RESERVED in stage.config:
            problems.append(f"{where}: config block {Constants.HUMAN_PARAM_RESERVED!r} is reserved and fails the stage")
        modes = {key.value: (axis, key) for axis, key in selected.items()}
        for block_name, block in stage.config.items():
            if block_name in human_blocks:
                continue
            if block_name not in modes:
                warnings.append(f"{where}: config block {block_name!r} matches neither tm_robots nor tm_obstacles and is not sent")
                continue
            axis, key = modes[block_name]
            if key not in registries[axis] or not isinstance(block, dict):
                continue
            params = mode_params(registries[axis], key)
            declared = {param["name"] for param in params}
            for leaf in _leaves(block):
                if leaf not in declared:
                    warnings.append(f"{where}: config.{block_name}.{leaf} is not a parameter of {block_name}, known: {sorted(declared)}")
            for param in params:
                catalog = catalogs.get(str(param.get("catalog")))
                names = _at(block, param["name"]) if catalog is not None else None
                if catalog is None or not isinstance(names, list):
                    continue
                for name in names:
                    if name not in catalog:
                        warnings.append(f"{where}: config.{block_name}.{param['name']} model {name!r} is not in the {param['catalog']} catalog, fetch it with arena asset pull or fix the name")
        if not world_resolves(stage.map, worlds):
            warnings.append(f"{where}: map {stage.map!r} does not resolve locally, it may be bucket-only (evaluation_maps network=true) or misspelled")
        unknown = [entry for entry in split_robot_arg(stage.robot) if entry.partition("[")[0].strip() not in robots]
        if raw.get("robot") != "auto" and unknown:
            warnings.append(f"{where}: robot {', '.join(unknown)} not in the robot catalog (evaluation_robots)")
    return {"valid": not problems, "n_stages": len(suite.stages), "stages": [stage.name for stage in suite.stages], "references": suite.references, "launch": suite.launch_args, "problems": problems, "warnings": warnings}


def _contest_drivers(args: dict[str, Any], cap: str) -> str | None:
    value = args.get(cap)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("driver"), str):
        return value["driver"]
    flat = args.get(f"{cap}.driver")
    return flat if isinstance(flat, str) else None


def check_contest(data: object) -> dict[str, Any]:
    """Contest.parse verdict plus the runner's nav2 preflight and drivers checked against the bringup and adapter registries."""
    from arena_evaluation.benchmark.config import Contest
    from arena_evaluation.benchmark.runner import _preflight_contest

    if not isinstance(data, list | dict):
        return {"valid": False, "problems": [f"a contest is a list or a mapping, got {type(data).__name__}"]}
    try:
        contest = Contest.parse("_validate", data)
    except (KeyError, ValueError, TypeError) as exc:
        return {"valid": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    catalog = planner_catalog()
    problems = list(_preflight_contest(contest))
    for contestant in contest.contestants:
        for cap in catalog["caps"]:
            driver = _contest_drivers(contestant.args, cap)
            if driver is not None and driver not in drivers(catalog, cap):
                problems.append(f"{contestant.name}: {cap} driver {driver!r} is not registered, known: {sorted(drivers(catalog, cap))}")
    return {"valid": not problems, "n_contestants": len(contest.contestants), "contestants": [{"name": entry.name, "args": entry.args, "parts": entry.parts} for entry in contest.contestants], "problems": problems}


def check_manifest(data: object) -> dict[str, Any]:
    from arena_evaluation.presentation.viz_manifest import VizManifest

    if not isinstance(data, dict):
        return {"valid": False, "problems": [f"a manifest is a mapping, got {type(data).__name__}"]}
    try:
        manifest = VizManifest.model_validate(data)
    except pydantic.ValidationError as exc:
        return {"valid": False, "problems": [str(exc)]}
    return {"valid": True, "n_plots": len(manifest.plots), "n_groups": len(manifest.groups), "problems": []}


def write_config(kind: str, name: str, text: str, overwrite: bool) -> pathlib.Path:
    """Write a suite, contest or manifest to the write target its resolver chain names, as a flat <name>.yaml."""
    stem = config_stem(name)
    if stem in config_names(kind) and not overwrite:
        raise ToolError(f"a {kind} named {stem!r} already exists", hint="pick another name or pass overwrite=true")
    target = asyncio.run(config_identifier(kind)(name=stem).resolve_write_path())
    path = target.parent / f"{target.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@domain.tool(read_only=True)
def evaluation_validate_suite(yaml_content: YamlText) -> dict[str, Any]:
    """Check a suite with the runner's Suite parser, then warn about task modes, config keys, maps, robots and models the registries do not know."""
    return check_suite(parse_yaml(yaml_content, "suite"))


@domain.tool(read_only=True)
def evaluation_validate_contest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a contest with the runner's Contest parser and its nav2 planner preflight, and its drivers against the registries."""
    return check_contest(parse_yaml(yaml_content, "contest"))


@domain.tool(read_only=True)
def evaluation_validate_manifest(yaml_content: YamlText) -> dict[str, Any]:
    """Check a report manifest against the VizManifest schema."""
    return check_manifest(parse_yaml(yaml_content, "manifest"))


@domain.tool()
def evaluation_create_suite(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a suite (evaluation_config_schema and evaluation_describe_task_mode give its keys) and write it where the runner resolves it by name."""
    verdict = check_suite(parse_yaml(yaml_content, "suite"))
    if not verdict["valid"]:
        raise ToolError("suite does not validate: " + "; ".join(verdict["problems"]), hint="evaluation_config_schema kind=suite shows the schema")
    return {"path": str(write_config("suite", name, yaml_content, overwrite)), **verdict}


@domain.tool()
def evaluation_create_contest(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a contest (evaluation_planners gives drivers and planner names) and write it where the runner resolves it by name."""
    verdict = check_contest(parse_yaml(yaml_content, "contest"))
    if not verdict["valid"]:
        raise ToolError("contest does not validate: " + "; ".join(verdict["problems"]), hint="evaluation_planners lists drivers and planners")
    return {"path": str(write_config("contest", name, yaml_content, overwrite)), **verdict}


@domain.tool()
def evaluation_create_manifest(
    name: str,
    yaml_content: Annotated[str | None, pydantic.Field(description="complete manifest YAML, or omit and give a template")] = None,
    template: Annotated[str | None, pydantic.Field(description="a resolvable manifest to start from (evaluation_manifests)")] = None,
    run_id: Annotated[str | None, pydantic.Field(description="write into this run's directory instead of the shared manifest root")] = None,
    overwrite: Overwrite = False,
) -> dict[str, Any]:
    """Write a report manifest, from YAML or a copy of a template, into a run directory or the shared manifest root (table notes: options.notes, options.rows, options.note, options.notes_key)."""
    import yaml

    from arena_evaluation.presentation.manifest_registry import resolve_manifest

    if (yaml_content is None) == (template is None):
        raise ToolError("pass yaml_content or template")
    if template is not None:
        try:
            manifest = resolve_manifest(config_stem(template))
        except FileNotFoundError as exc:
            raise ToolError(str(exc), hint="evaluation_manifests lists the templates") from exc
        manifest.name = name
        yaml_content = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    verdict = check_manifest(parse_yaml(yaml_content, "manifest"))
    if not verdict["valid"]:
        raise ToolError("manifest does not validate: " + "; ".join(verdict["problems"]))
    if run_id is None:
        return {"path": str(write_config("manifest", name, yaml_content, overwrite)), **verdict}
    path = run_path(run_id) / f"{config_stem(name)}.yaml"
    if path.exists() and not overwrite:
        raise ToolError(f"{path} exists", hint="pick another name or pass overwrite=true")
    path.write_text(yaml_content)
    return {"path": str(path), **verdict}


def _bounds_warnings(raw: dict[str, Any], bounds: dict[str, list[float]]) -> list[str]:
    (low_x, high_x), (low_y, high_y) = bounds["x"], bounds["y"]
    warnings = []

    def check(point: object, label: str) -> None:
        if isinstance(point, list | tuple) and len(point) >= 2 and all(isinstance(value, int | float) for value in point[:2]):
            x, y = float(point[0]), float(point[1])
            if not (low_x - 1.0 <= x <= high_x + 1.0 and low_y - 1.0 <= y <= high_y + 1.0):
                warnings.append(f"{label} ({x}, {y}) lies outside the zones X:[{low_x}, {high_x}] Y:[{low_y}, {high_y}]")

    for index, robot in enumerate(raw.get("robots") or []):
        check(robot.get("start"), f"robots[{index}].start")
        for phase_index, phase in enumerate(robot.get("phases") or []):
            if isinstance(phase, dict):
                check(phase.get("goto"), f"robots[{index}].phases[{phase_index}].goto")
    for index, human in enumerate(raw.get("humans") or []):
        check(human.get("pose"), f"humans[{index}].pose")
        for point_index, point in enumerate(human.get("waypoints") or []):
            check(point, f"humans[{index}].waypoints[{point_index}]")
    return warnings


def check_scenario(raw: object, map_name: str | None, run_id: str | None) -> dict[str, Any]:
    """parse_scenario verdict, counts, and with a map the coordinates that fall outside its zones."""
    from arena_simulation_setup.tree.World.Scenario import parse_scenario

    try:
        scenario = parse_scenario(raw)
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "problems": [str(exc)]}
    out: dict[str, Any] = {
        "valid": True,
        "problems": [],
        "robots": len(scenario.robots),
        "humans": len(scenario.humans),
        "objects": len(scenario.objects),
        "sounds": len(scenario.sounds),
        "lights": len(scenario.lights),
        "regions": sorted(scenario.regions),
        "timeline": len(scenario.timeline),
        "conditions": len(scenario.conditions),
        "warnings": [],
    }
    if map_name is not None:
        view, levels, _ = world_view(map_name, run_id)
        zones, _ = zone_summary(view, levels)
        if zones:
            corners = [(x, y) for zone in zones for x in zone["bounds"]["x"] for y in zone["bounds"]["y"]]
            bounds = {"x": [min(x for x, _ in corners), max(x for x, _ in corners)], "y": [min(y for _, y in corners), max(y for _, y in corners)]}
            out["warnings"] = _bounds_warnings(raw, bounds)
    return out


@domain.tool(read_only=True)
def evaluation_validate_scenario(yaml_content: YamlText, map_name: Annotated[str | None, pydantic.Field(description="world to check coordinates against")] = None, run_id: OptionalRunId = None) -> dict[str, Any]:
    """Check a format 2 scenario with the world loader's parse_scenario, and with map_name the coordinates outside its zones."""
    return check_scenario(parse_yaml(yaml_content, "scenario"), map_name, run_id)


@domain.tool()
def evaluation_create_scenario(map_name: str, scenario_name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a format 2 scenario and write it as scenarios/<scenario_name>/scenario.yaml of the world map_name resolves to."""
    verdict = check_scenario(parse_yaml(yaml_content, "scenario"), map_name, None)
    if not verdict["valid"]:
        raise ToolError("scenario does not validate: " + "; ".join(verdict["problems"]))
    view, _, source = world_view(map_name)
    target = asyncio.run(view.scenario(component(scenario_name, "scenario_name")).resolve_write_path()) / "scenario.yaml"
    if target.exists() and not overwrite:
        raise ToolError(f"{target} exists", hint="pick another scenario_name or pass overwrite=true")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml_content)
    return {"path": str(target), "world_resolved_by": source, **verdict}
