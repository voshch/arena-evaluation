"""Scenario and world files: format 1 shapes in memory, written as format 2 with the `edge_case` block in a sidecar."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World.model import Level, World

SIDECAR = "edge_case.yaml"
BLOCK_KEY = "edge_case"
# the format 1 upgrader would rewrite these
_FORMAT2_ONLY = ("objects", "sounds", "lights", "regions", "timeline", "conditions")


def read_scenario(path: Path) -> dict[str, Any]:
    doc = dict(yaml.safe_load(path.read_text()) or {})
    if doc.pop("format", None) is not None:
        doc["dynamic"] = doc.pop("humans", None) or []
    sidecar = path.with_name(SIDECAR)
    if sidecar.is_file():
        doc[BLOCK_KEY] = yaml.safe_load(sidecar.read_text()) or {}
    return doc


def _with_radius(agents: list[Any], out: Path) -> list[Any]:
    """Each agent block with its type's mean radius: the HumanSim adapter overrides an undeclared one with its own default.
    Scenario-local types resolve against `out`, so they are written first."""
    from task_generator.tasks.obstacles.edge_case.agent_types import AgentTypeError, load

    pinned = []
    for agent in agents:
        block = agent.get("agent") if isinstance(agent, dict) else None
        if isinstance(block, dict) and "radius" not in block:
            ref = str(block.get("agent_type") or "adult")
            try:
                radius = load(str(out / ref[2:]) if ref.startswith("./") else ref).agent_radius.mean
            except AgentTypeError:
                radius = None  # a type the runtime cannot load either; it falls back to `adult` there
            if radius:
                agent = {**agent, "agent": {**block, "radius": radius}}
        pinned.append(agent)
    return pinned


def write_scenario(out: Path, doc: Mapping[str, Any], header: str = "") -> Path:
    from arena_simulation_setup.tree.World.Scenario import upgrade_scenario

    work = dict(doc)
    if isinstance(work.get("dynamic"), list):
        work["dynamic"] = _with_radius(work["dynamic"], out)
    block = work.pop(BLOCK_KEY, None)
    kept = {key: work.pop(key) for key in _FORMAT2_ONLY if key in work}
    scenario, warnings = upgrade_scenario(work)
    if warnings:
        raise ValueError(f"{out}: format 2 would lose {warnings}")
    scenario.update(kept)
    out.mkdir(parents=True, exist_ok=True)
    dump = {"sort_keys": False, "default_flow_style": None, "allow_unicode": True}
    (out / "scenario.yaml").write_text(header + yaml.safe_dump(scenario, **dump))
    sidecar = out / SIDECAR
    if block:
        sidecar.write_text(header + yaml.safe_dump(dict(block), **dump))
    else:
        sidecar.unlink(missing_ok=True)
    return out / "scenario.yaml"


def _world(root: Path) -> World:
    from arena_simulation_setup.tree.World.view import WorldView

    return WorldView(root.resolve()).load()


def _working_level(level: Level) -> dict[str, Any]:
    import shapely

    zones = [{"name": zone.name, "corners": [{"x": p.x, "y": p.y} for p in zone.polygon], "entities": {"static": []}} for zone in level.zones]
    shapes = [shapely.Polygon([(c["x"], c["y"]) for c in z["corners"]]) if len(z["corners"]) >= 3 else None for z in zones]
    for obj in level.objects:
        x, y = obj.pose.position.x, obj.pose.position.y
        home = next((z for z, shape in zip(zones, shapes, strict=True) if shape is not None and shape.covers(shapely.Point(x, y))), None)
        if home is not None:
            home["entities"]["static"].append({"name": obj.name, "model": obj.model.name, "pose": [x, y]})
    return {"id": level.id, "zones": zones}


def levels(root: Path) -> list[dict[str, Any]]:
    return [_working_level(level) for level in _world(root).levels.values() if level.zones]


def first_level(root: Path) -> dict[str, Any] | None:
    return next(iter(levels(root)), None)


def is_world(root: Path) -> bool:
    return (root / "world.yaml").is_file() or any(root.glob("*/world.yaml"))
