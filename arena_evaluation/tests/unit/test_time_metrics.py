"""Time aggregates: the episode span, the robot's first goal arrival, and idling from the speed track."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX
from arena_evaluation.processing.metrics.performance.motion_metrics import SERIES_SCHEMA as MOTION_SERIES_SCHEMA
from arena_evaluation.processing.metrics.performance.path_metrics import SERIES_SCHEMA as PATH_SERIES_SCHEMA
from arena_evaluation.processing.metrics.performance.time_metrics import TimeMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _calculator() -> TimeMetricsCalculator:
    return TimeMetricsCalculator(RobotParams(robot_radius=0.2))


def _episode(t_ns: list[int], vel_linear: list[float | None] | None = None, goal: list[float] | None = None) -> AlignedEpisodeBundle:
    columns: dict[str, pl.Series] = {"t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64)}
    if vel_linear is not None:
        columns["vel_linear"] = pl.Series("vel_linear", vel_linear, dtype=pl.Float64)
    return AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(columns), start_pos=[], goal_pos=goal or [])


def _ground_truth_episode(xs: list[float], times_s: list[float], goal: list[float]) -> AlignedEpisodeBundle:
    t_ns = [int(t * SEC) for t in times_s]
    data = pl.DataFrame(
        {
            "t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64),
            "pos_x_gt": pl.Series("pos_x_gt", xs, dtype=pl.Float64),
            "pos_y_gt": pl.Series("pos_y_gt", [0.0] * len(xs), dtype=pl.Float64),
            "yaw_gt": pl.Series("yaw_gt", [0.0] * len(xs), dtype=pl.Float64),
        }
    )
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[0.0, 0.0, 0.0], goal_pos=goal)


def test_output_keys_do_not_carry_the_sample_clock():
    expected = ["time_diff", "time_to_goal", "idling_time"]

    assert TimeMetricsCalculator.output_keys() == expected
    assert "time" not in TimeMetricsCalculator.output_keys()
    assert {metric.name for metric in TimeMetricsCalculator.OUTPUTS} == set(expected)


def test_the_span_and_idling_come_from_the_recorded_speed_track():
    episode = _episode([1 * SEC, 2 * SEC, 3 * SEC, 4 * SEC], vel_linear=[0.5, 0.0, 0.0, 0.5])

    results = _calculator().calculate(episode, {})

    assert results["time_diff"] == 3 * SEC
    assert results["idling_time"] == pytest.approx(2.0)


def test_the_robot_reaches_its_goal_at_the_first_sample_within_one_metre():
    episode = _ground_truth_episode(xs=[0.0, 1.0, 2.0, 3.0], times_s=[1.0, 2.0, 3.0, 4.0], goal=[3.0, 0.0])

    results = _calculator().calculate(episode, {})

    # x = 2.0 is the first sample 1 m from the goal at x = 3.0, two seconds after the first sample.
    assert results["time_to_goal"] == pytest.approx(2.0)


def test_a_track_that_never_comes_within_one_metre_of_the_goal_has_no_time_to_goal():
    episode = _ground_truth_episode(xs=[0.0, 1.0, 2.0, 3.0], times_s=[1.0, 2.0, 3.0, 4.0], goal=[9.0, 0.0])

    results = _calculator().calculate(episode, {})

    assert results["time_to_goal"] is None
    assert results[f"time_to_goal{NULL_REASON_SUFFIX}"] == "no_samples"


def test_without_a_mission_goal_the_time_to_goal_is_null_with_its_reason():
    episode = _ground_truth_episode(xs=[0.0, 1.0], times_s=[1.0, 2.0], goal=[])

    results = _calculator().calculate(episode, {})

    assert results["time_to_goal"] is None
    assert results[f"time_to_goal{NULL_REASON_SUFFIX}"] == "reference_missing"


def test_a_single_sample_episode_has_no_duration_and_no_idling():
    results = _calculator().calculate(_episode([SEC]), {})

    assert results == dict.fromkeys(TimeMetricsCalculator.output_keys())


def test_without_a_speed_track_idling_is_null_and_the_span_survives():
    results = _calculator().calculate(_episode([0, SEC, 2 * SEC]), {})

    assert results["time_diff"] == 2 * SEC
    assert results["idling_time"] is None


def test_without_an_aligned_frame_every_time_output_is_absent():
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert _calculator().calculate(episode, {}) == dict.fromkeys(TimeMetricsCalculator.output_keys())


def test_the_retired_clock_is_the_t_ns_column_of_the_series_tables():
    """Every 1NF series table already carries the sample clock as ``t_ns``, so ``time`` needs no table of its own."""
    assert "t_ns" in MOTION_SERIES_SCHEMA
    assert "t_ns" in PATH_SERIES_SCHEMA
