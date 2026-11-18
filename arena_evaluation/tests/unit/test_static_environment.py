from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import shapely

from arena_evaluation.processing.path.static_environment import (
    build_static_grid_from_parts,
    circumradius_m,
    footprint_polygon,
    grid_for_map,
    obstacle_footprint,
)
from arena_evaluation.processing.path.theta_star import StaticGrid, plan_route

JACKAL_FOOTPRINT = ((0.229, 0.2078), (0.229, -0.2078), (-0.229, -0.2078), (-0.229, 0.2078))
RESOLUTION = 0.1
RADIUS = 0.3
NO_DOORS = shapely.Polygon()
NO_WALLS = shapely.GeometryCollection()


def _grid(*, free: shapely.Geometry, walls: shapely.Geometry = NO_WALLS, footprints: tuple[shapely.Polygon, ...] = ()) -> StaticGrid:
    return build_static_grid_from_parts(
        free=free,
        walls=walls,
        doors=NO_DOORS,
        footprints=footprints,
        resolution_m=RESOLUTION,
        robot_radius_m=RADIUS,
        padding_cells=5,
    )


def _lethal_at(grid: StaticGrid, x: float, y: float) -> bool:
    cell = grid.cell_of(x, y)
    assert cell is not None
    gx, gy = cell
    return bool(grid.lethal[gy, gx])


def _obstacle(*, x: float, y: float, yaw: float = 0.0, z: float = 0.0, asset_box=None, extra_box=None, name: str = "obstacle") -> SimpleNamespace:
    extra = {} if extra_box is None else {"bbox": extra_box}
    return SimpleNamespace(
        name=name,
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z), orientation=SimpleNamespace(to_yaw=lambda: yaw)),
        extra=extra,
        model=SimpleNamespace(resolve_sync=lambda: SimpleNamespace(extent=asset_box)),
    )


@pytest.mark.parametrize("map_name", [None, "", "unknown"])
def test_unnamed_maps_have_no_grid(map_name: str | None):
    assert grid_for_map(map_name, None, RADIUS) is None


def test_circumradius_of_the_jackal_footprint_is_the_corner_distance():
    assert circumradius_m(JACKAL_FOOTPRINT) == pytest.approx(math.hypot(0.229, 0.2078))


def test_footprint_turns_about_the_model_origin_not_the_box_centre():
    poly = footprint_polygon(x=5.0, y=5.0, yaw_rad=math.pi / 2, box=((0.0, 1.0), (0.0, 1.0), (0.0, 0.75)))
    assert poly.bounds == pytest.approx((4.0, 5.0, 5.0, 6.0), abs=1e-9)


def test_footprint_area_equals_the_box_area_at_any_heading():
    poly = footprint_polygon(x=1.0, y=2.0, yaw_rad=0.7, box=((-0.2, 0.3), (-0.4, 0.1), (0.0, 1.0)))
    assert poly.area == pytest.approx(0.5 * 0.5)


def test_authored_box_overrides_the_asset_extent():
    obstacle = _obstacle(x=1.0, y=2.0, asset_box=((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.75)), extra_box=((-1.0, 1.0), (-1.0, 1.0), (0.0, 0.75)))
    assert obstacle_footprint(obstacle).bounds == pytest.approx((0.0, 1.0, 2.0, 3.0))


def test_obstacle_whose_lowest_point_is_above_the_passage_clearance_does_not_block():
    obstacle = _obstacle(x=1.0, y=1.0, z=1.9, asset_box=((-0.5, 0.5), (-0.5, 0.5), (0.2, 0.75)))
    assert obstacle_footprint(obstacle) is None


def test_obstacle_below_the_passage_clearance_blocks():
    obstacle = _obstacle(x=1.0, y=1.0, z=0.0, asset_box=((-0.5, 0.5), (-0.5, 0.5), (0.0, 2.5)))
    assert obstacle_footprint(obstacle) is not None


def test_obstacle_without_any_box_is_an_error_naming_it():
    with pytest.raises(ValueError, match="shelf_3"):
        obstacle_footprint(_obstacle(x=0.0, y=0.0, name="shelf_3", asset_box=None))


def test_grid_origin_and_shape_pad_the_free_space_by_five_cells():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0))
    assert grid.origin_xy == pytest.approx((-0.5, -0.5))
    assert grid.lethal.shape == (110, 110)


def test_cells_are_floor_indexed_and_points_outside_the_grid_have_no_cell():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0))
    assert grid.cell_of(-0.5, -0.5) == (0, 0)
    assert grid.cell_of(-0.51, 0.0) is None


def test_lethal_cells_are_exactly_the_centres_within_the_radius_of_an_obstacle():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0), footprints=(shapely.box(4.5, 4.5, 5.5, 5.5),))
    assert _lethal_at(grid, 4.25, 5.05)
    assert not _lethal_at(grid, 4.15, 5.05)


def test_gap_below_two_radii_is_blocked_and_gap_of_two_radii_is_passable():
    room = shapely.box(0.0, 0.0, 10.0, 10.0)

    def walls_with_gap(gap: float) -> shapely.Geometry:
        return shapely.MultiLineString([((5.0, 0.0), (5.0, 4.0)), ((5.0, 4.0 + gap), (5.0, 10.0))])

    blocked = plan_route(_grid(free=room, walls=walls_with_gap(0.5)), (4.0, 4.25), (6.0, 4.25))
    passable = plan_route(_grid(free=room, walls=walls_with_gap(0.7)), (4.0, 4.35), (6.0, 4.35))
    assert blocked.null_reason == "no_path"
    assert passable.null_reason is None
    assert passable.length_m == pytest.approx(2.0)


def test_path_around_a_table_avoids_lethal_cells_and_is_longer_than_the_straight_line():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0), footprints=(shapely.box(4.0, 4.0, 6.0, 6.0),))
    path = plan_route(grid, (1.0, 5.0), (9.0, 5.0))
    assert path.null_reason is None
    assert path.length_m > 8.2
    assert not any(_lethal_at(grid, x, y) for x, y in path.points[1:-1])


def test_path_around_an_inner_corner_keeps_the_radius_clear_of_the_outside():
    room = shapely.union(shapely.box(0.0, 0.0, 10.0, 2.0), shapely.box(8.0, 0.0, 10.0, 10.0))
    grid = _grid(free=room)
    path = plan_route(grid, (1.0, 1.0), (9.0, 9.0))
    assert path.null_reason is None
    assert not any(_lethal_at(grid, x, y) for x, y in path.points[1:-1])


@pytest.mark.parametrize(("start", "goal", "reason"), [((3.9, 5.0), (9.0, 5.0), "start_in_lethal_cell"), ((1.0, 5.0), (3.9, 5.0), "goal_in_lethal_cell")])
def test_endpoint_inside_the_inflated_area_gives_an_explicit_null(start, goal, reason):
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0), footprints=(shapely.box(4.0, 4.0, 6.0, 6.0),))
    path = plan_route(grid, start, goal)
    assert path.null_reason == reason
    assert path.points is None
    assert path.length_m is None


def test_endpoint_outside_the_grid_is_reported_and_never_extrapolated():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0))
    assert plan_route(grid, (-100.0, 5.0), (5.0, 5.0)).null_reason == "start_outside_grid"


def test_start_enclosed_by_walls_has_no_path():
    grid = _grid(free=shapely.box(0.0, 0.0, 10.0, 10.0), walls=shapely.box(2.0, 2.0, 8.0, 8.0).boundary)
    assert plan_route(grid, (5.0, 5.0), (9.5, 9.5)).null_reason == "no_path"
