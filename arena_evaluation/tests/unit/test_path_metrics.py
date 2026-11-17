"""Path geometry: scalar aggregates in the metric row, per-sample path in the 1NF series table."""

from __future__ import annotations


import polars as pl
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import Direction
from arena_evaluation.processing.metrics.performance.path_metrics import (
    SERIES_SCHEMA,
    PathMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _episode(
    poses: list[tuple[float, float, float]], *, episode_id: int = 1
) -> AlignedEpisodeBundle:
    """Episode frame with the native /tf ground-truth poses."""
    xs = [pose[0] for pose in poses]
    ys = [pose[1] for pose in poses]
    yaws = [pose[2] for pose in poses]
    data = pl.DataFrame(
        {
            "t_ns": pl.Series(
                "t_ns", [index * SEC for index in range(len(poses))], dtype=pl.Int64
            ),
            "x": pl.Series("x", xs, dtype=pl.Float64),
            "y": pl.Series("y", ys, dtype=pl.Float64),
            "yaw": pl.Series("yaw", yaws, dtype=pl.Float64),
        }
    )
    return AlignedEpisodeBundle(
        episode_id=episode_id, data=data, start_pos=[], goal_pos=[]
    )


def _series(poses: list[tuple[float, float, float]]) -> pl.DataFrame:
    calculator = PathMetricsCalculator(make_robot_params(robot_radius=0.2))
    return calculator.series_table(_episode(poses), run_id="run", robot_id="jackal_0")


def test_path_length_orders_results_lower():
    assert PathMetricsCalculator.DIRECTIONS["path_length_m"] is Direction.LOWER
    assert set(PathMetricsCalculator.PRIMARY_OUTPUTS) <= set(
        PathMetricsCalculator.DIRECTIONS
    )


def test_without_an_aligned_frame_the_aggregates_and_the_series_are_absent():
    calculator = PathMetricsCalculator(make_robot_params(robot_radius=0.2))
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(
        PathMetricsCalculator.output_keys()
    )
    assert_frame_equal(
        calculator.series_table(episode),
        pl.DataFrame(schema=SERIES_SCHEMA),
        check_dtypes=True,
    )


