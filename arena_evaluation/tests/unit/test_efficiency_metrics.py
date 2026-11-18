import pathlib

import numpy as np
import polars as pl
import pytest
import yaml

from arena_evaluation.processing.metrics.naturalness.trajectory import (
    TrajectoryMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.efficiency_metrics import (
    PATH_EFFICIENCY,
    PathEfficiencyCalculator,
)
from arena_evaluation.processing.metrics.performance.vln_metrics import SPL_GEODESIC
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
OPEN_ROOM = "efficiency_test_open_room"


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
                            [25.0, -5.0],
                            [25.0, 25.0],
                            [-5.0, 25.0],
                        ],
                    }
                ],
            }
        ],
    }
    (world_dir / "world.yaml").write_text(yaml.safe_dump(world_spec))
    return tmp_path


def _params() -> RobotParams:
    return make_robot_params(robot_radius=0.25)


def _gt(xs: list[float], ys: list[float]) -> pl.DataFrame:
    times = [i * SEC for i in range(len(xs))]
    return pl.DataFrame(
        {
            "t_ns": times,
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "yaw": [0.0] * len(xs),
        }
    )


def _episode(
    gt: pl.DataFrame,
    goal: list[float],
    *,
    map_name: str | None = None,
    plans: pl.DataFrame | None = None,
    run_dir: pathlib.Path | None = None,
) -> AlignedEpisodeBundle:
    topics = {"robot_tracks": gt, "tf_gt": gt}
    if plans is not None:
        topics["robot_plans"] = plans
    return AlignedEpisodeBundle(
        1, gt, [], goal, map=map_name, topics=topics, run_dir=run_dir
    )


def _straight_episode(run_dir: pathlib.Path | None = None) -> AlignedEpisodeBundle:
    xs = [float(value) for value in np.linspace(0.0, 10.0, 21)]
    return _episode(
        _gt(xs, [0.0] * len(xs)), [10.0, 0.0], map_name=OPEN_ROOM, run_dir=run_dir
    )


def test_both_route_ratio_descriptions_state_the_shared_commanded_reference_rule():
    for metric in (PATH_EFFICIENCY, SPL_GEODESIC):
        assert "commanded" in metric.description
        assert "NULL" in metric.description
        assert "longer than the sampled path" in metric.description


def test_straight_ten_metre_path_in_an_empty_map_has_efficiency_one(
    open_room_run_dir: pathlib.Path,
):
    episode = _straight_episode(open_room_run_dir)

    trajectory = TrajectoryMetricsCalculator(_params()).calculate(episode, {})
    efficiency = PathEfficiencyCalculator(_params()).calculate(
        episode,
        {"path_length_m": 10.0, "theta_star_length": trajectory["theta_star_length"]},
    )

    assert trajectory["theta_star_length"] == pytest.approx(10.0)
    assert efficiency["path_efficiency"] == pytest.approx(1.0)


def test_closed_loop_without_a_recorded_route_has_no_geodesic_reference(
    open_room_run_dir: pathlib.Path,
):
    gt = _gt([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    episode = _episode(gt, [0.05, 0.0], map_name=OPEN_ROOM, run_dir=open_room_run_dir)

    trajectory = TrajectoryMetricsCalculator(_params()).calculate(episode, {})
    efficiency = PathEfficiencyCalculator(_params()).calculate(
        episode,
        {"path_length_m": 2.0, "theta_star_length": trajectory["theta_star_length"]},
    )

    assert trajectory["theta_star_length"] is None
    assert efficiency["path_efficiency"] is None


def test_relative_throughput_is_planned_time_over_time_to_goal_at_the_free_speed():
    times = np.arange(0.0, 4.0 + 1e-9, 0.5)
    robot = pl.DataFrame(
        {
            "t_ns": (times * SEC).astype(np.int64),
            "x": times,
            "y": np.zeros_like(times),
            "yaw": np.zeros_like(times),
            "vx": np.ones_like(times),
            "vy": np.zeros_like(times),
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0],
        goal_pos=[4.0, 0.0],
        topics={"robot_tracks": robot},
    )

    results = PathEfficiencyCalculator(_params()).calculate(
        episode, {"path_length_m": 4.0, "theta_star_length": 4.0, "time_to_goal": 8.0}
    )

    assert results["relative_throughput"] == pytest.approx(0.5)


