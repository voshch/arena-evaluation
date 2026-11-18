"""The benchmark's world geometry, one row per vertex."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final

import polars as pl

from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.path.static_environment import obstacle_footprint
from arena_evaluation.storage.registry.dimensions import DimWorldGeometry
from arena_evaluation.storage.star_schema import conform, empty_dimension

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.tree.World import Level

_WALL: Final = "wall"
_DOOR: Final = "door"
_ELEVATOR: Final = "elevator"
_ZONE: Final = "zone"
_OBJECT: Final = "object"
_WALL_NAME_PREFIX: Final = "wall_"
_WALL_INDEX_WIDTH: Final = 3
_SINGLE_PART: Final = 0
# What the world loader and obstacle_footprint raise for a world they cannot read.
_UNRESOLVED_WORLD_ERRORS: Final = (FileNotFoundError, ValueError)

_log = logging.getLogger(__name__)

_RowValue = str | int | float
_Row = dict[str, _RowValue]


def benchmark_world_geometry(
    worlds: Iterable[str], run_dir: pathlib.Path | None
) -> pl.DataFrame:
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
    # The simulator package is imported only when a world is read, as in MapRegistry.
    from arena_simulation_setup.tree.World.model import (
        render_door_polygons,
        render_elevator_polygons,
    )

    with MapRegistry.compacted_world_scope(world_name, run_dir=run_dir) as (level, _):
        if level is None:
            raise ValueError(f"world {world_name!r} compacts to no level")
        rows = [
            *_wall_rows(world_name, level),
            *_polygon_element_rows(
                world_name=world_name,
                element_kind=_DOOR,
                elements=(
                    (door.name, render_door_polygons(door)) for door in level.doors
                ),
            ),
            *_polygon_element_rows(
                world_name=world_name,
                element_kind=_ELEVATOR,
                elements=(
                    (elevator.name, render_elevator_polygons(elevator))
                    for elevator in level.elevators
                ),
            ),
            *_zone_rows(world_name, level),
            *_object_rows(world_name, level),
        ]
    return conform(pl.DataFrame(rows), DimWorldGeometry)


def _wall_rows(world_name: str, level: Level) -> list[_Row]:
    """The two vertices of every wall, named by its index in the level's wall order."""
    return [
        row
        for index, wall in enumerate(level.all_walls)
        for row in _vertex_rows(
            world_name=world_name,
            element_kind=_WALL,
            element_name=f"{_WALL_NAME_PREFIX}{index:0{_WALL_INDEX_WIDTH}d}",
            part_idx=_SINGLE_PART,
            vertices=((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)),
        )
    ]


def _zone_rows(world_name: str, level: Level) -> list[_Row]:
    """The vertices of every zone polygon, in its authored order."""
    return [
        row
        for zone in level.zones
        for row in _vertex_rows(
            world_name=world_name,
            element_kind=_ZONE,
            element_name=zone.name,
            part_idx=_SINGLE_PART,
            vertices=tuple((corner.x, corner.y) for corner in zone.polygon),
        )
    ]


def _object_rows(world_name: str, level: Level) -> list[_Row]:
    """The vertices of each object footprint that blocks the robot's passage."""
    rows: list[_Row] = []
    for obstacle in level.objects:
        footprint = obstacle_footprint(obstacle)
        if footprint is None:
            continue
        rows.extend(
            _polygon_element_rows(
                world_name=world_name,
                element_kind=_OBJECT,
                elements=((obstacle.name, (footprint,)),),
            )
        )
    return rows


def _polygon_element_rows(
    *,
    world_name: str,
    element_kind: str,
    elements: Iterable[tuple[str, Sequence[shapely.Polygon]]],
) -> list[_Row]:
    """The vertices of every exterior ring of each named element's polygons."""
    return [
        row
        for element_name, polygons in elements
        for part_idx, polygon in enumerate(polygons)
        for row in _vertex_rows(
            world_name=world_name,
            element_kind=element_kind,
            element_name=element_name,
            part_idx=part_idx,
            vertices=_exterior_ring(polygon),
        )
    ]


def _exterior_ring(polygon: shapely.Polygon) -> tuple[tuple[float, float], ...]:
    """The exterior ring without its closing vertex, projected onto x and y."""
    open_ring = tuple(polygon.exterior.coords)[:-1]
    return tuple((float(vertex[0]), float(vertex[1])) for vertex in open_ring)


def _vertex_rows(
    *,
    world_name: str,
    element_kind: str,
    element_name: str,
    part_idx: int,
    vertices: Sequence[tuple[float, float]],
) -> list[_Row]:
    """One registry row per vertex, in the order given."""
    return [
        {
            "world": world_name,
            "element_kind": element_kind,
            "element_name": element_name,
            "part_idx": part_idx,
            "vertex_idx": vertex_idx,
            "x": float(x),
            "y": float(y),
        }
        for vertex_idx, (x, y) in enumerate(vertices)
    ]
