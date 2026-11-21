"""The benchmark's world geometry, one row per vertex."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Final

import polars as pl

from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.path.static_environment import obstacle_footprint
from arena_evaluation.storage.registry.dimensions import DimWorldGeometry
from arena_evaluation.storage.star_schema import conform, empty_dimension

_WALL: Final = "wall"
_DOOR: Final = "door"
_ELEVATOR: Final = "elevator"
_ZONE: Final = "zone"
_OBJECT: Final = "object"
_WALL_NAME_PREFIX: Final = "wall_"
_WALL_INDEX_WIDTH: Final = 3
# What the world loader and obstacle_footprint raise for a world they cannot read.
_UNRESOLVED_WORLD_ERRORS: Final = (FileNotFoundError, ValueError)

_log = logging.getLogger(__name__)

_Vertex = tuple[float, float]
# A part is either its vertex sequence or a shapely polygon whose exterior ring supplies them.
_Part = Sequence[_Vertex] | object
_Element = tuple[str, str, Iterable[object]]
_Row = dict[str, str | int | float]


def benchmark_world_geometry(worlds: Iterable[str], run_dir: pathlib.Path | None) -> pl.DataFrame:
    """The rows of every world that resolves; a world that does not is skipped with a warning."""
    frames = [empty_dimension(DimWorldGeometry)]
    for world in worlds:
        try:
            frames.append(world_geometry_rows(world, run_dir))
        except _UNRESOLVED_WORLD_ERRORS as error:
            _log.warning("world %r has no published geometry: %s", world, error)
    return pl.concat(frames)


def world_geometry_rows(world_name: str, run_dir: pathlib.Path | None) -> pl.DataFrame:
    """One row per vertex of a world's compacted level; an unresolved world raises."""
    from arena_simulation_setup.tree.World.model import (
        render_door_polygons,
        render_elevator_polygons,
    )

    with MapRegistry.compacted_world_scope(world_name, run_dir=run_dir) as (level, _):
        if level is None:
            raise ValueError(f"world {world_name!r} compacts to no level")
        walls = [
            (
                _WALL,
                f"{_WALL_NAME_PREFIX}{index:0{_WALL_INDEX_WIDTH}d}",
                (((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)),),
            )
            for index, wall in enumerate(level.all_walls)
        ]
        doors = [(_DOOR, door.name, render_door_polygons(door)) for door in level.doors]
        elevators = [(_ELEVATOR, elevator.name, render_elevator_polygons(elevator)) for elevator in level.elevators]
        zones = [
            (
                _ZONE,
                zone.name,
                (tuple((corner.x, corner.y) for corner in zone.polygon),),
            )
            for zone in level.zones
        ]
        objects = [(_OBJECT, obstacle.name, (footprint,)) for obstacle in level.objects if (footprint := obstacle_footprint(obstacle)) is not None]
        rows = _element_rows(world_name, chain(walls, doors, elevators, zones, objects))
    return conform(pl.DataFrame(rows), DimWorldGeometry)


def _element_rows(world_name: str, elements: Iterable[_Element]) -> list[_Row]:
    """One registry row per vertex, in the authored element and part order.

    A part that is a vertex sequence is used as is; any other part is a shapely polygon whose
    exterior ring, without its closing vertex, supplies the vertices.
    """
    return [
        {
            "world": world_name,
            "element_kind": kind,
            "element_name": name,
            "part_idx": part_idx,
            "vertex_idx": vertex_idx,
            "x": float(vertex[0]),
            "y": float(vertex[1]),
        }
        for kind, name, parts in elements
        for part_idx, part in enumerate(parts)
        for vertex_idx, vertex in enumerate(
            part if isinstance(part, tuple) else tuple(part.exterior.coords)[:-1]  # type: ignore[union-attr]
        )
    ]
