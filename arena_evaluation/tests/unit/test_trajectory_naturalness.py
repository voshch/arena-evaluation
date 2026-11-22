import math
import pathlib

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.processing.metrics.naturalness.trajectory import (
    TrajectoryMetricsCalculator,
    _discrete_frechet,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
OPEN_ROOM = "trajectory_test_open_room"


@pytest.fixture
def open_room_run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    world_dir = tmp_path / "worlds" / OPEN_ROOM
    (world_dir / "assets").mkdir(parents=True)
    world_spec = {
        "format": "arena-world/2",
        "levels": [
            {
                "id": "0",
                "zones": [
                    {
                        "name": "open",
                        "polygon": [
                            [-5.0, -5.0],
                            [15.0, -5.0],
                            [15.0, 15.0],
                            [-5.0, 15.0],
                        ],
                    }
                ],
            }
        ],
    }
    (world_dir / "world.yaml").write_text(yaml.safe_dump(world_spec))
    return tmp_path


def _calc() -> TrajectoryMetricsCalculator:
    return TrajectoryMetricsCalculator(make_robot_params(robot_radius=0.25))


def _gt(xs: list[float], ys: list[float], yaws: list[float] | None = None) -> pl.DataFrame:
    n = len(xs)
    times = [i * SEC for i in range(n)]
    return pl.DataFrame(
        {
            "t_ns": times,
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "yaw": [0.0] * n if yaws is None else yaws,
        },
    )


def _episode(
    gt: pl.DataFrame,
    goal: list[float],
    *,
    map_name: str | None = None,
    peds: pl.DataFrame | None = None,
    plans: pl.DataFrame | None = None,
    run_dir: pathlib.Path | None = None,
) -> AlignedEpisodeBundle:
    topics = {"robot_tracks": gt, "tf_gt": gt}
    if peds is not None:
        topics["pedestrian_arena_peds"] = peds
        topics["peds"] = peds
    if plans is not None:
        topics["robot_plans"] = plans
    return AlignedEpisodeBundle(1, gt, [], goal, map=map_name, topics=topics, peds=peds, run_dir=run_dir)


def test_frechet_of_an_empty_curve_is_zero():
    assert _discrete_frechet(np.empty((0, 2)), np.array([[0.0, 0.0]])) is None


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
    assert res["theta_star_length_null_reason"] == "not_recorded"
    assert res["path_irregularity"] is None
    assert res["topological_complexity"] is None
    assert res["ade"] is None


def test_unparseable_pedestrian_column_leaves_winding_at_zero():
    gt = _gt([0.0, 1.0], [0.0, 0.0])
    peds = pl.DataFrame({"t_ns": [0, SEC], "x": ["invalid", "types"]})

    res = _calc().calculate(_episode(gt, [1.0, 0.0], peds=peds), {})

    assert res["topological_complexity"] is None


def test_long_trajectories_are_decimated_before_frechet(
    open_room_run_dir: pathlib.Path,
):
    xs = list(np.linspace(1.0, 4.0, 1201))

    res = _calc().calculate(
        _episode(
            _gt(xs, [1.0] * len(xs)),
            [4.0, 1.0],
            map_name=OPEN_ROOM,
            run_dir=open_room_run_dir,
        ),
        {},
    )

    assert res["frechet_distance"] == pytest.approx(1.5, abs=0.01)
    assert res["ade"] == pytest.approx(0.0, abs=1e-9)


def test_diverged_last_sample_scores_mhd_through_the_dense_distance_fallback(
    open_room_run_dir: pathlib.Path,
):
    res = _calc().calculate(
        _episode(
            _gt([1.0, 1.05, math.inf], [1.0, 1.0, 1.0]),
            [4.0, 1.0],
            map_name=OPEN_ROOM,
            run_dir=open_room_run_dir,
        ),
        {},
    )

    # The reference ends on the goal's 1 m arrival radius, so the 3 m leg is 2 m long.
    assert res["theta_star_length"] == pytest.approx(2.0)
    assert res["mhd"] == math.inf


def test_diverged_first_sample_is_refused_without_a_straight_line_reference(
    open_room_run_dir: pathlib.Path,
):
    res = _calc().calculate(
        _episode(
            _gt([math.inf, 1.0, 1.05], [1.0, 1.0, 1.0]),
            [4.0, 1.0],
            map_name=OPEN_ROOM,
            run_dir=open_room_run_dir,
        ),
        {},
    )

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


def test_the_reference_route_follows_every_commanded_goal_even_unvisited(
    open_room_run_dir: pathlib.Path,
):
    plans = pl.DataFrame(
        {
            "t_ns": [SEC, 2 * SEC],
            "plan_id": ["plan_0", "plan_1"],
            "waypoint_idx": [0, 0],
            "x": [2.0, 4.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
    )
    # The track turns back at 2 m: the second commanded goal stays unvisited, yet it stays in the mission.
    # Each leg ends on its goal's 1 m arrival radius: 0 -> 2 m is 1 m, then 1 -> 4 m is 2 m.
    gt = _gt([0.0, 1.0, 2.0, 1.0, 0.0], [0.0] * 5)

    res = _calc().calculate(
        _episode(gt, [0.0, 0.0], map_name=OPEN_ROOM, plans=plans, run_dir=open_room_run_dir),
        {},
    )

    assert res["theta_star_length"] == pytest.approx(3.0)


def test_unreached_first_leg_falls_back_to_the_commanded_mission_target(
    open_room_run_dir: pathlib.Path,
):
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

    res = _calc().calculate(
        _episode(gt, [4.0, 0.0], map_name=OPEN_ROOM, plans=plans, run_dir=open_room_run_dir),
        {},
    )

    # The 4 m leg ends on the goal's 1 m arrival radius.
    assert res["theta_star_length"] == pytest.approx(3.0)


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
