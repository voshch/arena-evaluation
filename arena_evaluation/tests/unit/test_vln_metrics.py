import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.performance.path_metrics import PathMetricsCalculator
from arena_evaluation.processing.metrics.performance.vln_metrics import (
    VlnMetricsCalculator,
    geodesic_distances,
    geodesic_field,
    ndtw,
    route,
    stuck_window,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.path.theta_star import GeometricThetaStar
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

RES = 0.1
SIZE = 200
RADIUS = 0.3
START = (2.0, 10.0)
GOAL = (15.0, 10.0)
TOLERANCE = 3.0
OCTILE_MAX_RATIO = math.sqrt(4.0 - 2.0 * math.sqrt(2.0))


def _solver(wall: bool) -> GeometricThetaStar:
    grid = np.zeros((SIZE, SIZE), dtype=bool)
    if wall:
        grid[0:180, 80:82] = True
    return GeometricThetaStar(grid, resolution=RES, origin=(0.0, 0.0), robot_radius=RADIUS)


def _field(solver: GeometricThetaStar, goal: tuple[float, float]) -> np.ndarray:
    cell = solver.world_to_grid(*goal)
    return geodesic_field(solver.obstacle_grid(cell), cell, solver.resolution)


def _geodesic(solver: GeometricThetaStar, field: np.ndarray, x: float, y: float) -> float:
    return float(geodesic_distances(field, np.array([x]), np.array([y]), solver.resolution, solver.origin)[0])


def _densify(waypoints: list[tuple[float, float]], step: float) -> np.ndarray:
    pts = [np.array(waypoints[0], dtype=float)]
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        pts.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.array(pts)


def _phases(gotos: list[tuple[float, float]], tolerance: float | None) -> dict:
    phase = {} if tolerance is None else {"tolerance_radius": tolerance}
    return {
        "phases": [{"goto": [x, y, 0.0], **phase} for x, y in gotos],
        "conditions": [],
        "map_poses": [[x, y, 0.0] for x, y in gotos],
    }


def _episode(
    waypoints: list[tuple[float, float]],
    map_name: str,
    dwell_s: float = 0.0,
    goal: tuple[float, float] = GOAL,
    tolerance: float | None = TOLERANCE,
    legs: tuple[tuple[float, float], ...] = (),
) -> AlignedEpisodeBundle:
    pts = _densify(waypoints, 0.1)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(steps)))
    yaw = np.arctan2(np.diff(pts[:, 1]), np.diff(pts[:, 0]))
    yaw = np.concatenate((yaw[:1], yaw))
    if dwell_s > 0:
        extra = np.arange(0.1, dwell_s + 1e-9, 0.1)
        pts = np.vstack((pts, np.repeat(pts[-1:], len(extra), axis=0)))
        t = np.concatenate((t, t[-1] + extra))
        yaw = np.concatenate((yaw, np.full(len(extra), yaw[-1])))
    data = pl.DataFrame(
        {
            "time_ns": (t * 1e9).astype(np.int64),
            "pos_x": pts[:, 0],
            "pos_y": pts[:, 1],
            "yaw": yaw,
            "pos_x_gt": pts[:, 0],
            "pos_y_gt": pts[:, 1],
            "yaw_gt": yaw,
        }
    )
    return AlignedEpisodeBundle(
        episode_id=0,
        data=data,
        start_pos=[pts[0, 0], pts[0, 1], yaw[0]],
        goal_pos=list(goal),
        phases=_phases([*legs, goal], tolerance),
        map=map_name,
    )


def _run(episode: AlignedEpisodeBundle, solver: GeometricThetaStar, success: bool) -> dict:
    params = RobotParams(robot_radius=RADIUS)
    prior = PathMetricsCalculator(params).calculate(episode, {})
    prior.update(success=success)
    return VlnMetricsCalculator(params).calculate_on_grid(episode, prior, solver)


def test_geodesic_around_wall_exceeds_euclidean() -> None:
    solver = _solver(wall=True)
    field = _field(solver, (10.0, 5.0))
    geodesic = _geodesic(solver, field, 6.0, 5.0)
    assert math.isfinite(geodesic)
    assert geodesic > 4.0 + 10.0


def test_geodesic_matches_euclidean_in_free_space() -> None:
    solver = _solver(wall=False)
    field = _field(solver, (10.0, 10.0))
    assert _geodesic(solver, field, 10.0, 16.0) == pytest.approx(6.0, abs=RES)
    assert _geodesic(solver, field, 4.0, 10.0) == pytest.approx(6.0, abs=RES)
    assert _geodesic(solver, field, 14.0, 14.0) == pytest.approx(4.0 * math.sqrt(2.0), abs=RES)
    euclid = math.hypot(6.0, 2.0)
    off_axis = _geodesic(solver, field, 16.0, 12.0)
    assert euclid - RES <= off_axis <= OCTILE_MAX_RATIO * euclid + RES


def test_geodesic_unreachable_and_off_grid_are_inf() -> None:
    blocked = np.zeros((50, 50), dtype=bool)
    blocked[:, 25] = True
    field = geodesic_field(blocked, (10, 10), 0.1)
    assert np.isinf(field[10, 40])
    assert np.isinf(geodesic_distances(field, np.array([-1.0, 99.0]), np.array([1.0, 1.0]), 0.1, (0.0, 0.0))).all()


def test_stop_one_meter_from_goal_succeeds() -> None:
    out = _run(_episode([START, (14.0, 10.0)], "vln_free_stop_1m"), _solver(wall=False), success=True)
    assert out["ne"] == pytest.approx(1.0, abs=RES)
    assert out["osr"] is True
    assert out["success_geodesic"] is True
    assert out["spl_geodesic"] == pytest.approx(1.0)
    assert 0.0 < out["sdtw"] == out["ndtw"] <= 1.0
    assert out["stuck"] is False


def test_stop_four_meters_from_goal_fails() -> None:
    out = _run(_episode([START, (11.0, 10.0)], "vln_free_stop_4m"), _solver(wall=False), success=True)
    assert out["ne"] == pytest.approx(4.0, abs=RES)
    assert out["osr"] is False
    assert out["success_geodesic"] is False
    assert out["spl_geodesic"] == 0.0
    assert out["ndtw"] > 0.0
    assert out["sdtw"] == 0.0


def test_threshold_is_the_final_goto_tolerance() -> None:
    out = _run(_episode([START, (11.0, 10.0)], "vln_free_stop_4m_wide", tolerance=5.0), _solver(wall=False), success=True)
    assert out["success_geodesic"] is True
    assert out["osr"] is True


def test_goto_without_tolerance_keeps_only_ne() -> None:
    out = _run(_episode([START, (11.0, 10.0)], "vln_free_no_tolerance", tolerance=None), _solver(wall=False), success=True)
    assert out["ne"] == pytest.approx(4.0, abs=RES)
    assert all(out[k] is None for k in ("osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"))


def test_passing_near_goal_then_leaving_is_oracle_success_only() -> None:
    out = _run(_episode([START, (13.0, 10.0), (10.0, 10.0)], "vln_free_overshoot"), _solver(wall=False), success=True)
    assert out["ne"] == pytest.approx(5.0, abs=RES)
    assert out["osr"] is True
    assert out["success_geodesic"] is False
    assert out["sdtw"] == 0.0


def test_goal_behind_wall_within_euclidean_threshold_fails() -> None:
    goal = (9.5, 10.0)
    out = _run(_episode([START, (7.0, 10.0)], "vln_wall_behind", goal=goal), _solver(wall=True), success=True)
    assert math.hypot(goal[0] - 7.0, 0.0) < TOLERANCE
    assert out["ne"] > TOLERANCE
    assert out["osr"] is False
    assert out["success_geodesic"] is False
    assert out["sdtw"] == 0.0


def test_runtime_failure_zeroes_sdtw_even_near_goal() -> None:
    out = _run(_episode([START, (14.0, 10.0)], "vln_free_runtime_fail"), _solver(wall=False), success=False)
    assert out["ne"] == pytest.approx(1.0, abs=RES)
    assert out["success_geodesic"] is False
    assert out["ndtw"] > 0.0
    assert out["sdtw"] == 0.0
    assert out["spl_geodesic"] == 0.0


def test_ndtw_identical_paths_is_one() -> None:
    path = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    assert ndtw(path, path, 3.0, 0.25) == pytest.approx(1.0)


def test_ndtw_decreases_with_lateral_offset() -> None:
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    scores = [ndtw(ref, ref + [0.0, offset], 3.0, 0.25) for offset in (0.0, 0.5, 1.0, 2.0)]
    assert scores[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(scores[:-1], scores[1:], strict=True))


def test_ndtw_independent_of_query_sampling_density() -> None:
    ref = np.array([[0.0, 0.0], [10.0, 0.0]])
    sparse = np.array([[0.0, 0.5], [5.0, 1.5], [10.0, 0.5]])
    dense = _densify([tuple(p) for p in sparse], 0.01)
    assert len(dense) > 50 * len(sparse)
    assert ndtw(ref, dense, 3.0, 0.25) == pytest.approx(ndtw(ref, sparse, 3.0, 0.25), abs=1e-9)


def test_stuck_window_detects_stationary_stretch() -> None:
    t = np.arange(0, 30.0, 0.1)
    x = np.where(t < 5.0, t, 5.0)
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)
    time_ns = (t * 1e9).astype(np.int64)
    assert stuck_window(x, y, yaw, time_ns, 10.0, 0.2, math.radians(15.0))
    assert not stuck_window(t, y, yaw, time_ns, 10.0, 0.2, math.radians(15.0))
    turning = np.where(t < 5.0, 0.0, (t - 5.0) * 0.5)
    assert not stuck_window(x, y, turning, time_ns, 10.0, 0.2, math.radians(15.0))


def test_stuck_flags_failed_dwell_and_ignores_success() -> None:
    solver = _solver(wall=False)
    long_dwell = _episode([START, (11.0, 10.0)], "vln_free_dwell_long", dwell_s=15.0)
    assert _run(long_dwell, solver, success=False)["stuck"] is True
    short_dwell = _episode([START, (11.0, 10.0)], "vln_free_dwell_short", dwell_s=5.0)
    assert _run(short_dwell, solver, success=False)["stuck"] is False
    at_goal = _episode([START, (14.0, 10.0)], "vln_free_dwell_goal", dwell_s=15.0)
    assert _run(at_goal, solver, success=True)["stuck"] is False


def test_calculate_without_map_leaves_geodesic_outputs_none() -> None:
    episode = _episode([START, (11.0, 10.0)], "", dwell_s=15.0)
    episode.map = None
    out = VlnMetricsCalculator(RobotParams(robot_radius=RADIUS)).calculate(episode, {"success": False, "path_length": 9.0})
    assert set(out) == set(VlnMetricsCalculator.output_keys())
    assert all(out[k] is None for k in ("ne", "osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"))
    assert out["stuck"] is True


def test_registry_runs_vln_after_its_dependencies() -> None:
    registry = MetricRegistry(RobotParams(robot_radius=RADIUS))
    stage_of = {name: i for i, stage in enumerate(registry.execution_order()) for name in stage}
    for dep in VlnMetricsCalculator.DEPENDS_ON:
        assert stage_of[dep] < stage_of[VlnMetricsCalculator.NAME]


def test_route_through_leg_goals_scores_against_the_chained_reference() -> None:
    leg = (8.0, 16.0)
    path = [START, leg, GOAL]
    chained = _run(_episode(path, "vln_free_two_legs", legs=(leg,)), _solver(wall=False), success=True)
    direct = _run(_episode(path, "vln_free_two_legs_direct"), _solver(wall=False), success=True)
    assert chained["ndtw"] == pytest.approx(1.0, abs=0.02)
    assert chained["spl_geodesic"] == pytest.approx(1.0, abs=0.02)
    assert direct["ndtw"] < chained["ndtw"] - 0.2
    assert direct["spl_geodesic"] < 0.8


def test_skipping_a_leg_goal_lowers_ndtw_but_not_success() -> None:
    leg = (8.0, 16.0)
    out = _run(_episode([START, GOAL], "vln_free_skip_leg", legs=(leg,)), _solver(wall=False), success=True)
    assert out["success_geodesic"] is True
    assert out["ndtw"] < 0.8


def test_route_is_the_map_pose_of_every_goto_and_the_last_tolerance() -> None:
    phases = {
        "phases": [{"goto": [1.0, 2.0, 0.0], "tolerance_radius": 0.3}, {"gesture": "wave"}, {"goto": "kitchen", "tolerance_radius": 0.8}],
        "conditions": [],
        "map_poses": [[6.0, 7.0, 0.0], None, [9.5, 4.0, 1.57]],
    }
    assert route(phases) == ([(6.0, 7.0), (9.5, 4.0)], 0.8)


def test_route_is_empty_without_a_goto() -> None:
    assert route(None) == ([], None)
    assert route({"phases": [{"gesture": "wave"}], "conditions": [], "map_poses": [None]}) == ([], None)


def test_episode_without_phases_leaves_geodesic_outputs_none() -> None:
    episode = _episode([START, (14.0, 10.0)], "vln_free_no_phases")
    episode.phases = None
    out = _run(episode, _solver(wall=False), success=True)
    assert all(out[k] is None for k in ("ne", "osr", "success_geodesic", "spl_geodesic", "ndtw", "sdtw"))
    assert out["stuck"] is False
