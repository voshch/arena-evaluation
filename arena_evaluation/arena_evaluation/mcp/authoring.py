from __future__ import annotations

import asyncio
import importlib
import pathlib
from typing import TYPE_CHECKING, Annotated, Any, cast

import pydantic

from arena_mcp.core import ToolError

from .base import YamlText, config_stem, domain, parse_yaml, run_path
from .catalog import config_identifier, config_names

if TYPE_CHECKING:
    import enum

    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_simulation_setup.tree import DomainAssetIdentifier
    from rcl_interfaces.msg import ParameterDescriptor
    from task_generator.tasks.registry import TaskModeRegistry

Overwrite = Annotated[bool, pydantic.Field(description="replace an existing config of that name")]


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
        self.declared: list[tuple[str, ParameterDescriptor]] = []

    def declare_forward(self, name: str, value: object, *, descriptor: ParameterDescriptor) -> None:
        self.declared.append((name, descriptor))


def _constraint_tokens(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for token in filter(None, text.split(";")):
        key, sep, value = token.partition(":")
        out[key] = (value.split(",") if key == "enum" else value) if sep else True
    return out


def mode_params(registry: TaskModeRegistry, key: enum.Enum) -> list[dict[str, Any]]:
    """The parameter names a registered task mode declares, below its namespace, with their constraint tokens."""
    meta = registry.meta(key)
    if meta.schema is None:
        return []
    collector = _Declarations()
    meta.schema(cast("ROSParamServer", collector), meta.namespace)
    prefix = f"{meta.namespace}."
    return [{"name": name.removeprefix(prefix), **_constraint_tokens(descriptor.additional_constraints)} for name, descriptor in collector.declared]


def domain_asset_names(identifier: type[DomainAssetIdentifier]) -> list[str]:
    """Names of a domain asset kind the way task modes reference them."""
    from arena_simulation_setup import DOMAIN_DEFAULT

    return sorted({item.name if item.domain == DOMAIN_DEFAULT else item.shortname for item in identifier.listall()})


def world_names() -> list[str]:
    from arena_simulation_setup.tree.World import WorldIdentifier

    return sorted({item.shortname for item in WorldIdentifier.listall() if not item.shortname.startswith(".")})


def world_resolves(name: str, known: set[str]) -> bool:
    """True when every world of a possibly composite, level-filtered world name is known."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import composite_parts

    return all(WorldIdentifier.split_filter(part)[0] in known for part in composite_parts(name))


def registered_drivers() -> dict[str, set[str]]:
    """Drivers per cap from the bringup and adapter registries."""
    from arena_robots.bringup import BRINGUPS
    from task_generator.tasks.robots.adapters import ADAPTERS

    for cap in ADAPTERS:
        importlib.import_module(f"task_generator.tasks.robots.adapters.{cap}")
    return {cap: set(BRINGUPS.get(cap, {}).keys()) | set(ADAPTERS.get(cap, {}).keys()) for cap in sorted(set(BRINGUPS) | set(ADAPTERS))}


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
                problems.append(f"{where}: {axis} {key.value!r} is not registered, see episode_task_modes")
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
            warnings.append(f"{where}: map {stage.map!r} does not resolve locally, it may be bucket-only (assets_ls kind=world network=true) or misspelled")
        unknown = [entry for entry in split_robot_arg(stage.robot) if entry.partition("[")[0].strip() not in robots]
        if raw.get("robot") != "auto" and unknown:
            warnings.append(f"{where}: robot {', '.join(unknown)} not in the robot catalog (robots_catalog)")
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
    drivers = registered_drivers()
    problems = list(_preflight_contest(contest))
    for contestant in contest.contestants:
        for cap, known in drivers.items():
            driver = _contest_drivers(contestant.args, cap)
            if driver is not None and driver not in known:
                problems.append(f"{contestant.name}: {cap} driver {driver!r} is not registered, known: {sorted(known)}")
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
    """Validate a suite (evaluation_config_schema and episode_task_modes give its keys) and write it where the runner resolves it by name."""
    verdict = check_suite(parse_yaml(yaml_content, "suite"))
    if not verdict["valid"]:
        raise ToolError("suite does not validate: " + "; ".join(verdict["problems"]), hint="evaluation_config_schema kind=suite shows the schema")
    return {"path": str(write_config("suite", name, yaml_content, overwrite)), **verdict}


@domain.tool()
def evaluation_create_contest(name: str, yaml_content: YamlText, overwrite: Overwrite = False) -> dict[str, Any]:
    """Validate a contest (planners_list gives the planner names) and write it where the runner resolves it by name."""
    verdict = check_contest(parse_yaml(yaml_content, "contest"))
    if not verdict["valid"]:
        raise ToolError("contest does not validate: " + "; ".join(verdict["problems"]), hint="planners_list lists the planners")
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

    if template is not None and yaml_content is None:
        try:
            manifest = resolve_manifest(config_stem(template))
        except FileNotFoundError as exc:
            raise ToolError(str(exc), hint="evaluation_manifests lists the templates") from exc
        manifest.name = name
        yaml_content = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    elif yaml_content is None or template is not None:
        raise ToolError("pass yaml_content or template")
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
