"""World furniture footprints from arena_simulation_setup, stamped onto the runtime map's walls.
An unannotated object is the box `WorldManager` stamps for it."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import shapely
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.model import World
    from arena_simulation_setup.tree.World.view import WorldView


def _footprints(view: WorldView, world: World, objects: list[Obstacle]) -> list[shapely.Polygon | None]:
    """Each object's footprint with the world's own and linked assets resolvable, as the runtime does when it loads a world."""
    from arena_simulation_setup.tree import DynamicPaths, Identifier
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    async def collect() -> list[shapely.Polygon | None]:
        # concurrent resolves contend and take minutes for a furnished world
        return [await obj.footprint() for obj in objects]

    def point_objects_at(path: Path | None) -> None:
        DynamicPaths.WORLD.path = path
        for resolver in ObjectIdentifier._resolvers:
            resolver.invalidate()

    previous, linked = DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths
    point_objects_at(view.path)
    DynamicPaths.LINKED.paths = list(world.sources)
    try:
        return Identifier._run_sync(collect())
    finally:
        point_objects_at(previous)
        DynamicPaths.LINKED.paths = linked


def static_polygon(obj: dict[str, Any]) -> list[tuple[float, float]] | None:
    return obj.get("polygon")


def world_statics(world_root: str | Path) -> list[dict[str, Any]]:
    """Every object of the world's first level, with its footprint `polygon` and a covering `radius`."""
    import shapely
    from arena_simulation_setup.tree.World.view import WorldView
    from task_generator.manager.world_manager.world_manager import _UNKNOWN_FOOTPRINT_RADIUS

    view = WorldView(Path(world_root).resolve())
    world = view.load()
    level = next(iter(world.levels.values()), None)
    if level is None:
        return []
    out: list[dict[str, Any]] = []
    for obj, footprint in zip(level.objects, _footprints(view, world, level.objects), strict=True):
        x, y = obj.pose.position.x, obj.pose.position.y
        if footprint is None:
            r = _UNKNOWN_FOOTPRINT_RADIUS
            footprint = shapely.box(x - r, y - r, x + r, y + r)
        polygon = list(footprint.exterior.coords)[:-1]
        out.append({"name": obj.name, "model": obj.model.name, "pose": [x, y], "polygon": polygon, "radius": max(math.hypot(px - x, py - y) for px, py in polygon)})
    return out
