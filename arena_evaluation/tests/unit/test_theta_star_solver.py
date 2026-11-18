import math

import numpy as np
import pytest

from arena_evaluation.processing.path import theta_star
from arena_evaluation.processing.path.theta_star import StaticGrid, _line_of_sight, plan_route


def _open(size: int = 40, resolution: float = 0.1, origin: tuple[float, float] = (0.0, 0.0)) -> StaticGrid:
    return StaticGrid(origin_xy=origin, resolution_m=resolution, lethal=np.zeros((size, size), dtype=bool))


def test_line_of_sight_leaving_the_grid_is_blocked():
    grid = np.zeros((5, 5), dtype=bool)

    assert _line_of_sight(grid, 0, 0, 7, 0) is False


@pytest.mark.parametrize("blocked", [(0, 1), (1, 0)])
def test_line_of_sight_diagonal_step_refuses_to_cut_an_occupied_corner(blocked: tuple[int, int]):
    grid = np.zeros((3, 3), dtype=bool)
    grid[blocked] = True

    assert _line_of_sight(grid, 0, 0, 1, 1) is False


@pytest.mark.parametrize(("start", "end"), [((0, 0), (2, 2)), ((2, 2), (0, 0)), ((0, 0), (1, 3)), ((3, 1), (0, 0))])
def test_line_of_sight_through_free_cells_in_every_direction(start: tuple[int, int], end: tuple[int, int]):
    grid = np.zeros((4, 4), dtype=bool)

    assert _line_of_sight(grid, *start, *end) is True


def test_line_of_sight_stops_at_an_obstacle_on_the_line():
    grid = np.zeros((5, 5), dtype=bool)
    grid[0, 2] = True

    assert _line_of_sight(grid, 0, 0, 4, 0) is False


def test_cell_of_floors_each_point_into_its_cell_and_rejects_points_outside():
    grid = _open(size=10, resolution=0.5, origin=(-1.0, 2.0))

    assert grid.cell_of(-1.0, 2.0) == (0, 0)
    assert grid.cell_of(0.2, 3.3) == (2, 2)
    assert grid.cell_of(4.0, 2.0) is None
    assert grid.cell_of(float("inf"), 2.0) is None


def test_cell_centre_lies_half_a_cell_into_its_cell():
    grid = _open(size=10, resolution=0.5, origin=(-1.0, 2.0))

    assert grid.cell_centre(2, 2) == pytest.approx((0.25, 3.25))


def test_open_grid_route_is_the_exact_straight_segment_with_its_hypot_length():
    grid = _open()
    start = (0.27502482987015087, 0.46768277259018687)
    goal = (3.3, 3.1)

    res = plan_route(grid, start, goal)

    assert res.null_reason is None
    assert res.points.tolist() == [list(start), list(goal)]
    assert res.length_m == math.hypot(goal[0] - start[0], goal[1] - start[1])


def test_route_around_a_wall_is_longer_than_the_line_and_keeps_exact_endpoints():
    grid = _open()
    grid.lethal[0:30, 20] = True

    res = plan_route(grid, (1.03, 1.0), (3.07, 1.0))

    assert res.null_reason is None
    assert res.points[0].tolist() == [1.03, 1.0]
    assert res.points[-1].tolist() == [3.07, 1.0]
    assert len(res.points) >= 3
    assert res.points[:, 1].max() >= 3.0
    assert res.length_m > 2 * math.hypot(1.0, 2.0) - 0.2


def test_goal_walled_in_has_no_route_and_no_straight_line():
    grid = _open(size=30)
    grid.lethal[16:25, 16] = True
    grid.lethal[16:25, 24] = True
    grid.lethal[16, 16:25] = True
    grid.lethal[24, 16:25] = True

    res = plan_route(grid, (0.5, 0.5), (2.0, 2.0))

    assert res.points is None
    assert res.length_m is None
    assert res.null_reason == "no_path"


@pytest.mark.parametrize(
    ("start", "goal", "reason"),
    [
        ((-0.5, 1.0), (3.0, 3.0), "start_outside_grid"),
        ((1.0, 1.0), (9.0, 1.0), "goal_outside_grid"),
        ((float("nan"), 1.0), (3.0, 3.0), "endpoint_not_finite"),
    ],
)
def test_endpoint_that_cannot_be_planned_from_gives_its_reason(start: tuple[float, float], goal: tuple[float, float], reason: str):
    res = plan_route(_open(), start, goal)

    assert res.null_reason == reason
    assert res.points is None


def test_endpoint_inside_a_lethal_cell_is_refused_not_routed_around():
    grid = _open()
    grid.lethal[10:14, 10:14] = True

    assert plan_route(grid, (1.05, 1.05), (3.0, 3.0)).null_reason == "start_in_lethal_cell"
    assert plan_route(grid, (3.0, 3.0), (1.05, 1.05)).null_reason == "goal_in_lethal_cell"


def test_missing_search_library_raises_instead_of_routing_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(theta_star, "_SOLVER_LIBRARY", tmp_path / "solver.so")
    monkeypatch.setattr(theta_star, "_SOLVER_SOURCE", tmp_path / "solver.cpp")
    grid = _open()
    grid.lethal[0:30, 20] = True
    theta_star._solver_library.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="search library"):
            plan_route(grid, (1.0, 1.0), (3.0, 1.0))
    finally:
        theta_star._solver_library.cache_clear()


def test_search_that_exhausts_its_iteration_budget_reports_the_cap_not_no_path():
    lethal = np.zeros((40, 40), dtype=bool)
    lethal[:-1, 20] = True
    assert theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=1) == "iteration_cap"
    assert theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=100000) != "iteration_cap"


def test_search_reports_no_path_when_a_full_wall_separates_the_endpoints():
    lethal = np.zeros((40, 40), dtype=bool)
    lethal[:, 20] = True
    assert theta_star._search(lethal, 0.1, (5, 5), (35, 5), max_iterations=100000) == "no_path"
