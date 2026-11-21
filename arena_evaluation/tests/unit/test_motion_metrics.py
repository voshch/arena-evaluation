"""Motion aggregates in the metric row per sample kinematics in the 1NF motion series table."""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.performance.motion_metrics import (
    SERIES_SCHEMA,
    MotionMetricsCalculator,
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
