"""Scenario and world files as the generator reads and writes them.

The generator works on format 1 shapes in memory: a scenario's `dynamic:` list, a level's `zones:` with `corners:`
and the furniture under each zone's `entities: static:`. Format 2 files (`arena-scenario/2`, `arena-world/2`) are read
into that shape here. Every scenario is written as format 2 through arena_simulation_setup's own upgrader, with the
`edge_case` block in a sidecar, since format 2 rejects keys it does not know.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

#: The edge_case block of a scenario, beside its `scenario.yaml`.
SIDECAR = "edge_case.yaml"
BLOCK_KEY = "edge_case"
#: Format 2 scenario keys the format 1 upgrader would rewrite; a format 2 document carries them through as they are.
_FORMAT2_ONLY = ("objects", "sounds", "lights", "regions", "timeline", "conditions")


def read_scenario(path: Path) -> dict[str, Any]:
    """The scenario at `path` in the working shape, its edge_case block (sidecar, or a format 1 key) under `edge_case`."""
    doc = dict(yaml.safe_load(path.read_text()) or {})
    if doc.pop("format", None) is not None:
        doc["dynamic"] = doc.pop("humans", None) or []
    sidecar = path.with_name(SIDECAR)
    if sidecar.is_file():
        doc[BLOCK_KEY] = yaml.safe_load(sidecar.read_text()) or {}
    return doc


def write_scenario(out: Path, doc: Mapping[str, Any], header: str = "") -> Path:
    """`out/scenario.yaml` in format 2 and, when `doc` carries a block, `out/edge_case.yaml`, both under `header`."""
    from arena_simulation_setup.tree.World.Scenario import upgrade_scenario  # noqa: PLC0415 - the runtime package, only needed to write

    work = dict(doc)
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


def _world_v2(root: Path) -> dict[str, Any] | None:
    path = root / "world.yaml"
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    return doc if isinstance(doc, dict) and doc.get("format") else None


def _working_level(level: Mapping[str, Any]) -> dict[str, Any]:
    """One format 2 level as zones with `corners`, each holding the furniture inside it under `entities: static:`."""
    import shapely  # noqa: PLC0415

    zones = []
    for raw in level.get("zones") or []:
        corners = [{"x": float(x), "y": float(y)} for x, y, *_ in raw.get("polygon") or []]
        zones.append({"name": str(raw["name"]), "corners": corners, "entities": {"static": []}})
    shapes = [shapely.Polygon([(c["x"], c["y"]) for c in z["corners"]]) if len(z["corners"]) >= 3 else None for z in zones]
    for obj in level.get("objects") or []:
        pose = _pose(obj.get("pose"))
        if pose is None or not zones:
            continue
        x, y, yaw = pose
        entity = {"name": obj.get("name", "?"), "model": str(obj.get("model", "")), "pose": {"position": {"x": x, "y": y}, "orientation": {"w": math.cos(yaw / 2.0), "x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0)}}}
        point = shapely.Point(x, y)
        home = next((z for z, shape in zip(zones, shapes, strict=True) if shape is not None and shape.covers(point)), zones[0])
        home["entities"]["static"].append(entity)
    return {"id": str(level.get("id", "")), "zones": zones}


def _pose(raw: object) -> tuple[float, float, float] | None:
    """x, y, yaw of any pose form format 2 accepts but a zone name, which an object never uses."""
    from arena_simulation_setup.utils.geometry import Pose  # noqa: PLC0415

    if raw is None or isinstance(raw, str):
        return None
    if isinstance(raw, Mapping) and isinstance(raw.get("position"), Mapping):
        pos, ori = raw["position"], raw.get("orientation") or {}
        w, z = float(ori.get("w", 1.0)), float(ori.get("z", 0.0))
        x_, y_ = float(ori.get("x", 0.0)), float(ori.get("y", 0.0))
        return float(pos["x"]), float(pos["y"]), math.atan2(2.0 * (w * z + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z * z))
    try:
        pose = Pose.parse(raw)
    except Exception:  # noqa: BLE001 - an object the generator cannot place is one it does not stamp
        return None
    return float(pose.position.x), float(pose.position.y), float(pose.orientation.to_yaw())


def levels(root: Path) -> list[dict[str, Any]]:
    """Every level of the world at `root` in the working shape (`zones` with `corners`), in level order."""
    world = _world_v2(root)
    if world is None:
        out = []
        for path in sorted(root.glob("*/world.yaml")):
            doc = yaml.safe_load(path.read_text())
            out.append({"id": path.parent.name, "zones": doc} if isinstance(doc, list) else {"id": path.parent.name, **dict(doc or {})})
        return out
    return [_working_level(lv) for lv in world.get("levels") or [] if isinstance(lv, Mapping) and lv.get("zones")]


def first_level(root: Path) -> dict[str, Any] | None:
    """The world's first level in the working shape. None when the world has no level file."""
    return next(iter(levels(root)), None)


def level_map(root: Path) -> Path | None:
    """The first level's `map.yaml`: `map/<level id>.yaml` in format 2, `<level>/map.yaml` in format 1."""
    if _world_v2(root) is None:
        return next(iter(sorted(root.glob("*/map.yaml"))), None)
    level = first_level(root)
    path = root / "map" / f"{level['id'] or '0'}.yaml" if level is not None else None
    return path if path is not None and path.is_file() else None


def is_world(root: Path) -> bool:
    return _world_v2(root) is not None or any(root.glob("*/world.yaml"))
