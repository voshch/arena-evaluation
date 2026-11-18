"""Motion aggregates in the metric row per sample kinematics in the 1NF motion series table."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.performance.motion_metrics import (
    SERIES_SCHEMA,
    MotionMetricsCalculator,
    _speed_from_pose,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _episode(
    *,
    speed_mps: list[float | None] | None = None,
    x: list[float] | None = None,
    sample_count: int | None = None,
) -> AlignedEpisodeBundle:
    """Episode frame at 1 s spacing, with a twist stream or a ground-truth pose track."""
    if x is not None:
        count = len(x)
        data = pl.DataFrame(
            {
                "t_ns": pl.Series(
                    "t_ns", [index * SEC for index in range(count)], dtype=pl.Int64
                ),
                "x": pl.Series("x", x, dtype=pl.Float64),
                "y": pl.Series("y", [0.0] * count, dtype=pl.Float64),
                "yaw": pl.Series("yaw", [0.0] * count, dtype=pl.Float64),
            }
        )
    else:
        count = len(speed_mps) if speed_mps is not None else (sample_count or 0)
        data = pl.DataFrame(
            {
                "t_ns": pl.Series(
                    "t_ns", [index * SEC for index in range(count)], dtype=pl.Int64
                ),
                "speed_mps": pl.Series(
                    "speed_mps",
                    speed_mps if speed_mps is not None else [None] * count,
                    dtype=pl.Float64,
                ),
            }
        )
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])


def _calculator() -> MotionMetricsCalculator:
    return MotionMetricsCalculator(make_robot_params(robot_radius=0.2))


def test_output_keys_keep_only_the_aggregates():
    expected = [
        "mean_velocity_mps",
        "velocity_max",
        "acceleration_mean",
        "max_acceleration_mps2",
        "jerk_mean",
        "jerk_max_mps3",
    ]

    assert MotionMetricsCalculator.output_keys() == expected
    assert {metric.name for metric in MotionMetricsCalculator.OUTPUTS} == set(expected)


def _dense_speed_episode(
    speed_mps: float, *, rate_hz: float = 50.0, duration_s: float = 6.0
) -> AlignedEpisodeBundle:
    """Constant-speed episode on a grid fine enough for the declared 1.0 s smoothing window."""
    count = int(round(duration_s * rate_hz))
    t_ns = (np.arange(count, dtype=np.float64) / rate_hz * SEC).astype(np.int64)
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(
            {
                "t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64),
                "speed_mps": pl.Series(
                    "speed_mps", np.full(count, speed_mps), dtype=pl.Float64
                ),
            }
        ),
        start_pos=[],
        goal_pos=[],
    )


def test_constant_velocity_series_has_zero_acceleration_and_jerk():
    calculator = _calculator()
    episode = _dense_speed_episode(1.0)
    table = calculator.series_table(episode, run_id="run", robot_id="jackal_0")
    results = calculator.calculate(episode, {})

    assert table.height == 120  # the kinematics grid is thinned to the declared 20 Hz
    assert float(table["speed_mps"].min()) == pytest.approx(1.0)
    assert float(table["acceleration_mps2"].abs().max()) < 1e-9
    assert float(table["jerk_mps3"].abs().max()) < 1e-9
    assert results["acceleration_mean"] == pytest.approx(0.0, abs=1e-9)
    assert results["jerk_mean"] == pytest.approx(0.0, abs=1e-9)


def test_below_the_smoothing_window_acceleration_and_jerk_are_null_never_the_raw_difference():
    calculator = _calculator()
    episode = _episode(speed_mps=[0.0, 1.0, 2.0, 3.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    # The raw backward difference would publish 1.0 m/s² here: the one estimator is the smoothed fit, so NULL.
    assert table["acceleration_mps2"].is_null().all()
    assert table["jerk_mps3"].is_null().all()
    assert results["mean_velocity_mps"] == pytest.approx(1.5)
    assert results["velocity_max"] == pytest.approx(3.0)
    assert results["acceleration_mean"] is None
    assert results["max_acceleration_mps2"] is None
    assert results["jerk_mean"] is None
    assert results["jerk_max_mps3"] is None


def test_velocity_series_is_derived_from_ground_truth_without_a_twist_stream():
    table = _calculator().series_table(_episode(x=[0.0, 1.0, 2.0, 3.0]))

    assert_series_equal(
        table["speed_mps"],
        pl.Series("speed_mps", [None, 1.0, 1.0, 1.0], dtype=pl.Float64),
        check_names=False,
        rel_tol=1e-9,
    )
    assert table["acceleration_mps2"].is_null().all()
    assert table["jerk_mps3"].is_null().all()


def test_series_summaries_match_the_metric_row():
    calculator = _calculator()
    episode = _ramp_episode()

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert results["mean_velocity_mps"] == pytest.approx(
        float(table["speed_mps"].mean())
    )
    assert results["velocity_max"] == pytest.approx(float(table["speed_mps"].max()))
    assert results["max_acceleration_mps2"] == pytest.approx(
        float(table["acceleration_mps2"].abs().max())
    )
    assert results["jerk_max_mps3"] == pytest.approx(
        float(table["jerk_mps3"].abs().max())
    )


def test_a_single_sample_has_no_acceleration_series():
    calculator = _calculator()
    episode = _episode(speed_mps=[1.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert table.height == 1
    assert table["acceleration_mps2"].is_null().all()
    assert table["jerk_mps3"].is_null().all()
    assert results["mean_velocity_mps"] == pytest.approx(1.0)
    assert results["acceleration_mean"] is None
    assert results["jerk_mean"] is None


def test_two_samples_have_speed_but_no_derivative_series():
    calculator = _calculator()
    episode = _episode(speed_mps=[0.0, 2.0])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert_series_equal(
        table["speed_mps"],
        pl.Series("speed_mps", [0.0, 2.0], dtype=pl.Float64),
        check_names=False,
    )
    assert table["acceleration_mps2"].is_null().all()
    assert table["jerk_mps3"].is_null().all()
    assert results["mean_velocity_mps"] == pytest.approx(1.0)
    assert results["acceleration_mean"] is None
    assert results["jerk_mean"] is None


def test_without_an_aligned_frame_aggregates_and_series_are_empty():
    calculator = _calculator()
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(
        MotionMetricsCalculator.output_keys()
    )
    assert_frame_equal(
        calculator.series_table(episode),
        pl.DataFrame(schema=SERIES_SCHEMA),
        check_dtypes=True,
    )


# --- smoothed derivative estimator: physical plausibility ---------------------------------


def _ramp_episode(
    *,
    rate_hz: float = 50.0,
    duration_s: float = 6.0,
    accel_mps2: float = 1.0,
    ramp_s: float = 2.0,
) -> AlignedEpisodeBundle:
    """Analytic profile: constant acceleration for ramp_s, then constant velocity."""
    count = int(round(duration_s * rate_hz))
    time_s = np.arange(count, dtype=np.float64) / rate_hz
    ramp_speed_mps = accel_mps2 * ramp_s
    accelerating = time_s <= ramp_s
    x_m = np.where(
        accelerating,
        0.5 * accel_mps2 * time_s**2,
        0.5 * accel_mps2 * ramp_s**2 + ramp_speed_mps * (time_s - ramp_s),
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(
            {
                "t_ns": pl.Series(
                    "t_ns", (time_s * SEC).astype(np.int64), dtype=pl.Int64
                ),
                "x": pl.Series("x", x_m, dtype=pl.Float64),
                "y": pl.Series("y", np.zeros(count), dtype=pl.Float64),
                "yaw": pl.Series("yaw", np.zeros(count), dtype=pl.Float64),
            }
        ),
        start_pos=[],
        goal_pos=[],
    )


def test_a_constant_acceleration_ramp_has_zero_jerk_away_from_its_kinks():
    calculator = _calculator()
    episode = _ramp_episode()
    results = calculator.calculate(episode, {})
    table = calculator.series_table(episode)
    time_s = table["time_s"].to_numpy()
    jerk_mps3 = table["jerk_mps3"].to_numpy()

    for interior_s in (1.0, 4.0, 5.0):
        window = np.abs(time_s - interior_s) < 0.1
        assert np.all(np.abs(jerk_mps3[window]) < 0.05), interior_s

    assert results["velocity_max"] == pytest.approx(2.0, abs=0.05)
    assert results["max_acceleration_mps2"] == pytest.approx(1.0, abs=0.2)
    assert results["jerk_max_mps3"] is not None
    assert results["jerk_max_mps3"] < 25.0


def test_a_noisy_ground_truth_tracks_does_not_inflate_the_peaks():
    """A 1 cm position noise at 50 Hz is thousands of m/s^3 under raw differentiation, far less smoothed."""
    calculator = _calculator()
    rate_hz = 50.0
    count = int(10.0 * rate_hz)
    time_s = np.arange(count, dtype=np.float64) / rate_hz
    rng = np.random.default_rng(7)
    x_m = 1.0 * time_s + rng.normal(0.0, 0.01, count)
    y_m = rng.normal(0.0, 0.01, count)
    t_ns = (time_s * SEC).astype(np.int64)
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(
            {
                "t_ns": pl.Series("t_ns", t_ns, dtype=pl.Int64),
                "x": pl.Series("x", x_m, dtype=pl.Float64),
                "y": pl.Series("y", y_m, dtype=pl.Float64),
                "yaw": pl.Series("yaw", np.zeros(count), dtype=pl.Float64),
            }
        ),
        start_pos=[],
        goal_pos=[],
    )

    results = calculator.calculate(episode, {})
    time_s = t_ns / SEC
    raw_acceleration = np.diff(_speed_from_pose(x_m, y_m, t_ns)) / np.diff(time_s)
    raw_jerk = np.diff(raw_acceleration) / np.diff(time_s[1:])

    assert float(np.nanmax(np.abs(raw_jerk))) > 100.0
    assert results["jerk_max_mps3"] < 50.0
    assert results["max_acceleration_mps2"] < 2.5
    assert results["mean_velocity_mps"] == pytest.approx(1.0, abs=0.1)


def test_sample_order_and_duplicate_stamps_leave_the_peaks_unchanged():
    calculator = _calculator()
    episode = _ramp_episode()
    baseline = calculator.calculate(episode, {})

    duplicated = pl.concat([episode.data, episode.data.head(1)])
    reordered = duplicated.sort("t_ns", descending=True)
    variant = AlignedEpisodeBundle(
        episode_id=1, data=reordered, start_pos=[], goal_pos=[]
    )
    reordered_results = calculator.calculate(variant, {})

    assert reordered_results["max_acceleration_mps2"] == pytest.approx(
        baseline["max_acceleration_mps2"], rel=1e-9
    )
    assert reordered_results["jerk_max_mps3"] == pytest.approx(
        baseline["jerk_max_mps3"], rel=1e-9
    )


def test_smoothed_series_and_metric_row_cannot_disagree():
    calculator = _calculator()
    episode = _ramp_episode()
    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert results["mean_velocity_mps"] == pytest.approx(
        float(table["speed_mps"].mean())
    )
    assert results["velocity_max"] == pytest.approx(float(table["speed_mps"].max()))
    assert results["max_acceleration_mps2"] == pytest.approx(
        float(table["acceleration_mps2"].abs().max())
    )
    assert results["jerk_max_mps3"] == pytest.approx(
        float(table["jerk_mps3"].abs().max())
    )
