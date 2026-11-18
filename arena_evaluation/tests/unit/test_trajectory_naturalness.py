import math

import numpy as np
import polars as pl
import pytest
import shapely

from arena_evaluation.processing.metrics.common.routes import _planned_waypoints, _reached_waypoints
from arena_evaluation.processing.metrics.naturalness import trajectory as trajectory_module
from arena_evaluation.processing.metrics.naturalness.trajectory import (
    TrajectoryMetricsCalculator,
    _discrete_frechet,
)
from arena_evaluation.processing.path.static_environment import build_static_grid_from_parts
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
OPEN_ROOM = "trajectory_test_open_room"


@pytest.fixture(autouse=True)
def open_room_grid(monkeypatch):
    def grid_for_map(map_name, run_dir, robot_radius_m):
        if map_name != OPEN_ROOM:
            return None
        return build_static_grid_from_parts(
            free=shapely.box(-5.0, -5.0, 15.0, 15.0),
            walls=shapely.GeometryCollection(),
            doors=shapely.Polygon(),
            footprints=(),
            resolution_m=0.1,
            robot_radius_m=robot_radius_m,
            padding_cells=5,
        )

    monkeypatch.setattr(trajectory_module, "grid_for_map", grid_for_map)


def _calc() -> TrajectoryMetricsCalculator:
    return TrajectoryMetricsCalculator(RobotParams(robot_radius=0.25))


def _gt(xs: list[float], ys: list[float], yaws: list[float] | None = None) -> pl.DataFrame:
    n = len(xs)
    times = [i * SEC for i in range(n)]
    return pl.DataFrame(
        {
            "t_ns": times,
            "pos_x_gt": [float(x) for x in xs],
            "pos_y_gt": [float(y) for y in ys],
            "yaw_gt": [0.0] * n if yaws is None else yaws,
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "yaw": [0.0] * n if yaws is None else yaws,
        },
    )


def _episode(gt: pl.DataFrame, goal: list[float], *, map_name: str | None = None, peds: pl.DataFrame | None = None, plans: pl.DataFrame | None = None) -> AlignedEpisodeBundle:
    topics = {"robot_tracks": gt, "tf_gt": gt}
    if peds is not None:
        topics["pedestrian_tracks"] = peds
        topics["peds"] = peds
    if plans is not None:
        topics["robot_plans"] = plans
    return AlignedEpisodeBundle(1, gt, [], goal, map=map_name, topics=topics, peds=peds)


def test_frechet_of_an_empty_curve_is_zero():
    assert _discrete_frechet(np.empty((0, 2)), np.array([[0.0, 0.0]])) is None


def test_frechet_of_parallel_polylines_is_their_offset():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert _discrete_frechet(p, p + [0.0, 1.0]) == pytest.approx(1.0)


def test_frechet_decimates_long_curves_and_keeps_the_offset():
    p = np.column_stack((np.linspace(0.0, 10.0, 600), np.zeros(600)))
    q = np.column_stack((np.linspace(0.0, 10.0, 700), np.ones(700)))

    assert _discrete_frechet(p, q) == pytest.approx(1.0, abs=0.05)


def test_empty_trajectory_yields_none_for_every_output():
    res = _calc().calculate(_episode(_gt([], []), [1.0, 1.0]), {})

    assert res == dict.fromkeys(TrajectoryMetricsCalculator.output_keys())


def test_single_sample_without_map_has_no_reference_length_and_says_why():
    res = _calc().calculate(_episode(_gt([0.0], [0.0]), [3.0, 4.0]), {})

    assert res["fde"] == pytest.approx(5.0)
    assert res["theta_star_length"] is None
    assert res["theta_star_length_null_reason"] == "no_map"
    assert res["path_irregularity"] is None
    assert res["topological_complexity"] is None
    assert res["ade"] is None


def test_path_irregularity_needs_more_than_ten_centimeters_of_travel():
    res = _calc().calculate(_episode(_gt([0.0, 0.05], [0.0, 0.0]), [3.0, 0.0]), {})

    assert res["path_irregularity"] is None
    assert res["fde"] == pytest.approx(2.95)


def test_path_irregularity_is_turning_beyond_the_initial_heading_error_per_meter():
    res = _calc().calculate(_episode(_gt([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [math.pi / 2, 0.0, 0.0]), [2.0, 0.0]), {})

    assert res["path_irregularity"] == pytest.approx(0.0, abs=1e-12)


def test_topological_complexity_counts_net_winding_per_pedestrian():
    gt = _gt([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    peds = pl.DataFrame(
        {
            "t_ns": [0, 0, SEC],
            "ped_id": ["ped_0", "ped_1", "ped_0"],
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 5.0, 1.0],
        }
    )

    res = _calc().calculate(_episode(gt, [1.0, 1.0], peds=peds), {})

    assert res["topological_complexity"] == pytest.approx(0.25)


def test_unparseable_pedestrian_column_leaves_winding_at_zero():
    gt = _gt([0.0, 1.0], [0.0, 0.0])
    peds = pl.DataFrame({"t_ns": [0, SEC], "x": ["invalid", "types"]})

    res = _calc().calculate(_episode(gt, [1.0, 0.0], peds=peds), {})

    assert res["topological_complexity"] is None


def test_straight_drive_on_the_reference_line_scores_hand_computed_errors():
    res = _calc().calculate(_episode(_gt([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]), [4.0, 1.0], map_name=OPEN_ROOM), {})

    assert res["theta_star_length"] == pytest.approx(3.0)
    assert res["fde"] == pytest.approx(0.0)
    assert res["ade"] == pytest.approx(0.0)
    assert res["mhd"] == pytest.approx(0.5)
    assert res["frechet_distance"] == pytest.approx(1.0)
    assert res["adtw"] == pytest.approx(1.0 / 3.0)
    assert res["path_irregularity"] == pytest.approx(0.0)


def test_stationary_robot_on_a_map_has_no_arc_length_ade():
    res = _calc().calculate(_episode(_gt([1.0, 1.0], [1.0, 1.0]), [4.0, 1.0], map_name=OPEN_ROOM), {})

    assert res["ade"] is None
    assert res["fde"] == pytest.approx(3.0)
    assert res["mhd"] == pytest.approx(1.5)


def test_long_trajectories_are_decimated_before_frechet():
    xs = list(np.linspace(1.0, 4.0, 1201))

    res = _calc().calculate(_episode(_gt(xs, [1.0] * len(xs)), [4.0, 1.0], map_name=OPEN_ROOM), {})

    assert res["frechet_distance"] == pytest.approx(1.5, abs=0.01)
    assert res["ade"] == pytest.approx(0.0, abs=1e-9)


def test_diverged_last_sample_scores_mhd_through_the_dense_distance_fallback():
    res = _calc().calculate(_episode(_gt([1.0, 1.05, math.inf], [1.0, 1.0, 1.0]), [4.0, 1.0], map_name=OPEN_ROOM), {})

    assert res["theta_star_length"] == pytest.approx(3.0)
    assert res["mhd"] == math.inf


def test_diverged_first_sample_is_refused_without_a_straight_line_reference():
    res = _calc().calculate(_episode(_gt([math.inf, 1.0, 1.05], [1.0, 1.0, 1.0]), [4.0, 1.0], map_name=OPEN_ROOM), {})

    assert res["theta_star_length"] is None
    assert res["theta_star_length_null_reason"] == "endpoint_not_finite"
    assert res["fde"] == pytest.approx(2.95)
    assert res["mhd"] is None


def test_missing_goal_leaves_reference_metrics_unset():
    res = _calc().calculate(_episode(_gt([0.0, 1.0], [0.0, 0.0]), []), {})

    assert res["fde"] is None
    assert res["theta_star_length"] is None
    assert res["path_irregularity"] is None
    assert res["topological_complexity"] is None


def test_theta_star_length_sums_the_recorded_mission_legs():
    plans = pl.DataFrame(
        {
            "t_ns": [SEC, 2 * SEC],
            "plan_id": ["plan_0", "plan_1"],
            "waypoint_idx": [0, 0],
            "x": [3.0, 0.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    gt = _gt([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], [0.0] * 7)

    res = _calc().calculate(_episode(gt, [0.0, 0.0], map_name=OPEN_ROOM, plans=plans), {})

    assert res["theta_star_length"] == pytest.approx(6.0)


def test_unvisited_commanded_goals_do_not_inflate_the_reference_route():
    plans = pl.DataFrame(
        {
            "t_ns": [SEC, 2 * SEC],
            "plan_id": ["plan_0", "plan_1"],
            "waypoint_idx": [0, 0],
            "x": [50.0, 0.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    gt = _gt([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0], [0.0] * 7)

    res = _calc().calculate(_episode(gt, [0.0, 0.0], map_name=OPEN_ROOM, plans=plans), {})

    assert res["theta_star_length"] is None


def test_unreached_first_leg_falls_back_to_the_commanded_mission_target():
    plans = pl.DataFrame(
        {
            "t_ns": [SEC],
            "plan_id": ["plan_0"],
            "waypoint_idx": [0],
            "x": [4.0],
            "y": [0.0],
            "yaw": [0.0],
        }
    )
    gt = _gt([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])

    res = _calc().calculate(_episode(gt, [4.0, 0.0], map_name=OPEN_ROOM, plans=plans), {})

    assert res["theta_star_length"] == pytest.approx(4.0)


def _plans(xs: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [SEC * (index + 1) for index in range(len(xs))],
            "plan_id": [f"plan_{index}" for index in range(len(xs))],
            "waypoint_idx": [0] * len(xs),
            "x": [float(value) for value in xs],
            "y": [0.0] * len(xs),
            "yaw": [0.0] * len(xs),
        }
    )


def test_planned_waypoints_merge_goals_closer_than_the_quarter_metre_tolerance():
    assert _planned_waypoints({"robot_plans": _plans([3.0, 3.2])}) == ((3.0, 0.0),)


def test_planned_waypoints_keep_goals_more_than_the_quarter_metre_apart():
    assert _planned_waypoints({"robot_plans": _plans([3.0, 4.2])}) == ((3.0, 0.0), (4.2, 0.0))


def test_reached_waypoints_count_a_ninety_centimetre_pass():
    track_x = np.array([0.0, 0.0])
    track_y = np.array([0.0, 0.0])

    assert _reached_waypoints(track_x=track_x, track_y=track_y, waypoints=((0.0, 0.9),)) == ((0.0, 0.9),)


def test_reached_waypoints_reject_a_one_point_one_metre_pass():
    track_x = np.array([0.0, 0.0])
    track_y = np.array([0.0, 0.0])

    assert _reached_waypoints(track_x=track_x, track_y=track_y, waypoints=((0.0, 1.1),)) == ()
