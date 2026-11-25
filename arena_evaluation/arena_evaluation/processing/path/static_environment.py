"""Static environment of a map: walls and furniture inflated into a lethal mask for plan_route."""

from __future__ import annotations

import functools
import math
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry.base import BaseGeometry

from arena_evaluation.processing.map_registry import MapRegistry
from arena_evaluation.processing.path.theta_star import StaticGrid

if TYPE_CHECKING:
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World import Level

Box = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

# Same height rule as the simulator's map renderer: an obstacle whose lowest point is above this does not block the robot.
PASSAGE_CLEARANCE_M: Final = 2.0
DEFAULT_RESOLUTION_M: Final = 0.05
DEFAULT_PADDING_CELLS: Final = 5
ROUNDING_DECIMALS: Final = 6


def grid_for_map(map_name: str | None, run_dir: pathlib.Path | None, robot_radius_m: float) -> StaticGrid | None:
    """Lethal grid of a named map's world for a robot radius; None for an unnamed map. A world that does not resolve raises."""
    if not map_name or map_name == "unknown":
        return None
    return _world_grid(map_name, run_dir, robot_radius_m)


@functools.cache
def _world_grid(map_name: str, run_dir: pathlib.Path | None, robot_radius_m: float) -> StaticGrid:
    with MapRegistry.compacted_world_scope(map_name, run_dir=run_dir) as (compacted, _):
        if compacted is None:
            raise ValueError(f"world {map_name!r} compacts to no level")
        return build_static_grid(compacted, robot_radius_m=robot_radius_m)


def circumradius_m(footprint: Sequence[tuple[float, float]]) -> float:
    """Radius of the smallest disk about the model origin that holds every footprint corner."""
    return max(math.hypot(x, y) for x, y in footprint)


def footprint_polygon(*, x: float, y: float, yaw_rad: float, box: Box) -> shapely.Polygon:
    """The annotated box in the map frame: turned about the model origin, then moved to the pose."""
    (x_min, x_max), (y_min, y_max), _ = box
    turned = shapely.affinity.rotate(
        shapely.box(x_min, y_min, x_max, y_max),
        yaw_rad,
        origin=(0.0, 0.0),
        use_radians=True,
    )
    return shapely.affinity.translate(turned, x, y)


def obstacle_footprint(obstacle: Obstacle) -> shapely.Polygon | None:
    """The obstacle's footprint in the map frame, None when it clears the robot's passage height."""
    box = _obstacle_box(obstacle)
    _, _, (z_min, _) = box
    if obstacle.pose.position.z + z_min > PASSAGE_CLEARANCE_M:
        return None
    return footprint_polygon(
        x=obstacle.pose.position.x,
        y=obstacle.pose.position.y,
        yaw_rad=obstacle.pose.orientation.to_yaw(),
        box=box,
    )


def _obstacle_box(obstacle: Obstacle) -> Box:
    authored = obstacle.extra.get("bbox")
    if authored is not None:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = authored
        return (
            (float(x_min), float(x_max)),
            (float(y_min), float(y_max)),
            (float(z_min), float(z_max)),
        )
    extent = obstacle.model.resolve_sync().extent
    if extent is None:
        raise ValueError(f"static obstacle {obstacle.name!r} ({obstacle.model}) has no bounding box: annotate its asset or give the world a bbox for it")
    return extent


def static_footprints(level: Level) -> tuple[shapely.Polygon, ...]:
    """Footprints of the level's static obstacles that block the robot, in the level's map frame."""
    footprints = (obstacle_footprint(obstacle) for obstacle in level.objects)
    return tuple(footprint for footprint in footprints if footprint is not None)


def build_static_grid(
    level: Level,
    *,
    robot_radius_m: float,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    padding_cells: int = DEFAULT_PADDING_CELLS,
) -> StaticGrid:
    """Lethal mask of a compacted level, drawn the way the evaluator's map is: rooms, cabins and doors are free and everything else blocks."""
    from arena_simulation_setup.tree.World.model import (
        render_door_polygons,
        render_elevator_polygons,
    )

    door_polygons = [polygon for door in level.doors for polygon in render_door_polygons(door)]
    cabin_polygons = [polygon for elevator in level.elevators for polygon in render_elevator_polygons(elevator)]
    room_polygons = [zone.shape() for zone in level.zones if len(zone.polygon) >= 3]
    walls = [shapely.LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]) for wall in level.all_walls]
    return build_static_grid_from_parts(
        free=shapely.union_all([*room_polygons, *cabin_polygons, *door_polygons]),
        walls=shapely.union_all(walls),
        doors=shapely.union_all(door_polygons),
        footprints=static_footprints(level),
        resolution_m=resolution_m,
        robot_radius_m=robot_radius_m,
        padding_cells=padding_cells,
    )


def build_static_grid_from_parts(
    *,
    free: BaseGeometry,
    walls: BaseGeometry,
    doors: BaseGeometry,
    footprints: Sequence[shapely.Polygon],
    resolution_m: float,
    robot_radius_m: float,
    padding_cells: int,
) -> StaticGrid:
    """Lethal mask of map-frame geometry: space outside the free shapes, walls not cut by a door, and footprints block."""
    if robot_radius_m <= 0.0:
        raise ValueError(f"robot radius must be positive, got {robot_radius_m}")
    blocking = shapely.union_all([walls.difference(doors), *footprints])
    min_x, min_y, max_x, max_y = shapely.union_all([free, walls, *footprints]).bounds
    origin_x = min_x - padding_cells * resolution_m
    origin_y = min_y - padding_cells * resolution_m
    cols = math.ceil(round((max_x - min_x) / resolution_m + 2 * padding_cells, ROUNDING_DECIMALS))
    rows = math.ceil(round((max_y - min_y) / resolution_m + 2 * padding_cells, ROUNDING_DECIMALS))
    grid_box = shapely.box(
        origin_x,
        origin_y,
        origin_x + cols * resolution_m,
        origin_y + rows * resolution_m,
    )
    outside = grid_box.difference(free)
    inflated = shapely.buffer(shapely.union_all([outside, blocking]), robot_radius_m)
    centre_x, centre_y = np.meshgrid(
        origin_x + (np.arange(cols) + 0.5) * resolution_m,
        origin_y + (np.arange(rows) + 0.5) * resolution_m,
    )
    lethal = shapely.intersects(inflated, shapely.points(centre_x.ravel(), centre_y.ravel())).reshape(rows, cols)
    return StaticGrid(origin_xy=(origin_x, origin_y), resolution_m=resolution_m, lethal=lethal)
