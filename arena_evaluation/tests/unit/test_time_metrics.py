"""Time aggregates: the episode span, the robot's first goal arrival, and idling from the speed track."""

from __future__ import annotations

import polars as pl

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX
from arena_evaluation.processing.metrics.performance.motion_metrics import (
    SERIES_SCHEMA as MOTION_SERIES_SCHEMA,
)
from arena_evaluation.processing.metrics.performance.path_metrics import (
    SERIES_SCHEMA as PATH_SERIES_SCHEMA,
)
from arena_evaluation.processing.metrics.performance.time_metrics import (
    TimeMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _calculator() -> TimeMetricsCalculator:
    return TimeMetricsCalculator(make_robot_params(robot_radius=0.2))


def _episode(
    t_ns: list[int],
    speed_mps: list[float | None] | None = None,
    goal: list[float] | None = None,
) -> AlignedEpisodeBundle:
    columns: dict[str, pl.Series] = {"t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64)}
    if speed_mps is not None:
        columns["speed_mps"] = pl.Series("speed_mps", speed_mps, dtype=pl.Float64)
    return AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(columns), start_pos=[], goal_pos=goal or [])


def _ground_truth_episode(xs: list[float], times_s: list[float], goal: list[float]) -> AlignedEpisodeBundle:
    t_ns = [int(t * SEC) for t in times_s]
    data = pl.DataFrame(
        {
            "t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64),
            "x": pl.Series("x", xs, dtype=pl.Float64),
            "y": pl.Series("y", [0.0] * len(xs), dtype=pl.Float64),
            "yaw": pl.Series("yaw", [0.0] * len(xs), dtype=pl.Float64),
        }
    )
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0, 0.0], goal_pos=goal)


def test_a_track_that_never_comes_within_one_metre_of_the_goal_has_no_time_to_goal():
    episode = _ground_truth_episode(xs=[0.0, 1.0, 2.0, 3.0], times_s=[1.0, 2.0, 3.0, 4.0], goal=[9.0, 0.0])

    results = _calculator().calculate(episode, {})

    assert results["time_to_goal"] is None
    assert results[f"time_to_goal{NULL_REASON_SUFFIX}"] == "no_samples"


def test_a_single_sample_episode_has_no_duration_and_no_idling():
    results = _calculator().calculate(_episode([SEC]), {})

    assert results == dict.fromkeys(TimeMetricsCalculator.output_keys())


def test_the_retired_clock_is_the_t_ns_column_of_the_series_tables():
    """Every 1NF series table already carries the sample clock as ``t_ns``, so ``time`` needs no table of its own."""
    assert "t_ns" in MOTION_SERIES_SCHEMA
    assert "t_ns" in PATH_SERIES_SCHEMA
