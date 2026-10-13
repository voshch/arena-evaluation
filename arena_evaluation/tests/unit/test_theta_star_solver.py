import math

import numpy as np
import pytest

from arena_evaluation.processing.path import theta_star
from arena_evaluation.processing.path.theta_star import (
    GeometricThetaStar,
    _line_of_sight,
    compute_theta_star_for_episode,
    load_map_solver,
)


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


def test_integer_grid_treats_occupied_and_unknown_cells_as_obstacles():
    grid = np.array([[0, 100, -1], [50, 51, 0]], dtype=np.int16)

    solver = GeometricThetaStar(grid, resolution=0.1, robot_radius=0.0)

    assert solver.dilated_grid.tolist() == [[False, True, True], [False, True, False]]
    assert (solver.width, solver.height) == (3, 2)


def test_float_grid_thresholds_at_one_half():
    grid = np.array([[0.2, 0.5, 0.9]])

    solver = GeometricThetaStar(grid, resolution=0.1, robot_radius=0.0)

    assert solver.dilated_grid.tolist() == [[False, False, True]]


def test_non_positive_resolution_falls_back_to_five_centimeters():
    solver = GeometricThetaStar(np.zeros((2, 2), dtype=bool), resolution=0.0, robot_radius=0.0)

    assert solver.resolution == pytest.approx(0.05)


def test_robot_radius_inflates_obstacles_by_a_disk():
    grid = np.zeros((7, 7), dtype=bool)
    grid[3, 3] = True

    solver = GeometricThetaStar(grid, resolution=0.05, robot_radius=0.1)

    assert int(solver.dilated_grid.sum()) == 13
    assert solver.dilated_grid[3, 1] and solver.dilated_grid[1, 3]
    assert not solver.dilated_grid[1, 1]


def test_obstacle_grid_clears_the_window_around_each_cell_clipped_to_the_map():
    solver = GeometricThetaStar(np.ones((6, 6), dtype=bool), resolution=0.1, robot_radius=0.0)

    cleared = solver.obstacle_grid((0, 0), (5, 5))

    assert int((~cleared).sum()) == 18
    assert not cleared[2, 2] and not cleared[3, 3]
    assert cleared[0, 5] and cleared[5, 0]
    assert solver.dilated_grid.all()


def test_world_and_grid_coordinates_round_trip_through_origin_and_resolution():
    solver = GeometricThetaStar(np.zeros((10, 10), dtype=bool), resolution=0.5, origin=(-1.0, 2.0), robot_radius=0.0)

    assert solver.world_to_grid(0.0, 3.0) == (2, 2)
    assert solver.world_to_grid(0.2, 3.3) == (2, 3)
    assert solver.grid_to_world(2, 2) == (0.0, 3.0)


def test_out_of_map_endpoints_return_the_straight_segment_and_cache_it():
    solver = GeometricThetaStar(np.zeros((10, 10), dtype=bool), resolution=0.1, robot_radius=0.0)

    pts, length = solver.solve((-3.0, 0.0), (0.0, 4.0), map_id="theta_test_out_of_map")
    again = solver.solve((-3.0, 0.0), (0.0, 4.0), map_id="theta_test_out_of_map")

    assert pts.tolist() == [[-3.0, 0.0], [0.0, 4.0]]
    assert length == pytest.approx(5.0)
    assert again[0] is pts


def test_unreachable_goal_falls_back_to_the_euclidean_segment():
    grid = np.zeros((30, 30), dtype=bool)
    grid[16:25, 16] = True
    grid[16:25, 24] = True
    grid[16, 16:25] = True
    grid[24, 16:25] = True
    solver = GeometricThetaStar(grid, resolution=0.1, robot_radius=0.0)

    pts, length = solver.solve((0.5, 0.5), (2.0, 2.0), map_id="theta_test_walled_goal")

    assert pts.tolist() == [[0.5, 0.5], [2.0, 2.0]]
    assert length == pytest.approx(math.hypot(1.5, 1.5))


def test_detour_around_a_wall_is_longer_than_the_line_and_keeps_exact_endpoints():
    grid = np.zeros((40, 40), dtype=bool)
    grid[0:30, 20] = True
    solver = GeometricThetaStar(grid, resolution=0.1, robot_radius=0.0)

    pts, length = solver.solve((1.03, 1.0), (3.07, 1.0), map_id="theta_test_wall_detour")

    assert pts[0].tolist() == [1.03, 1.0]
    assert pts[-1].tolist() == [3.07, 1.0]
    assert len(pts) >= 3
    assert pts[:, 1].max() >= 3.0
    assert length > 2 * math.hypot(1.0, 2.0) - 0.2


@pytest.mark.parametrize("name", ["", "unknown"])
def test_unnamed_maps_have_no_solver(name: str):
    assert load_map_solver(name) is None


def test_episode_path_without_a_map_is_the_euclidean_segment():
    res = compute_theta_star_for_episode("unknown", (1.0, 1.0), (4.0, 5.0))

    assert res.success is True
    assert res.geodesic_length == pytest.approx(5.0)
    assert res.path_x.tolist() == [1.0, 4.0]
    assert res.path_y.tolist() == [1.0, 5.0]


def test_episode_path_reuses_the_cached_map_solver():
    grid = np.zeros((40, 40), dtype=bool)
    grid[0:30, 20] = True
    theta_star._solver_instances["theta_test_cached_map_0.0"] = GeometricThetaStar(grid, resolution=0.1, robot_radius=0.0)

    res = compute_theta_star_for_episode("theta_test_cached_map", (1.0, 1.0), (3.0, 1.0), robot_radius=0.0)

    assert load_map_solver("theta_test_cached_map", robot_radius=0.0) is theta_star._solver_instances["theta_test_cached_map_0.0"]
    assert res.success is True
    assert len(res.path_points) >= 3
    assert res.geodesic_length > 2.0
