"""Robot speed, acceleration and jerk over the scope window's ground truth.

Peak acceleration and jerk are estimated from the recorded ground-truth speed with a local
polynomial (Savitzky-Golay) fit over a declared **1.0 s** window and **polynomial order 2**
on a uniform grid of at most 20 Hz (a coarser recording keeps its own step). Differentiating the raw speed samples at the
recording rate amplifies sample noise into physically impossible peaks — a jackal with a
2.5 m/s² actuator limit showed median peak accelerations near 28 m/s² and median peak jerk
near 900 m/s³ — so the declared window is what the reported peaks measure. That fit is the only
derivative estimator: an episode with fewer samples than the smoothing fit needs publishes NULL
acceleration and jerk (its speed stays measured) instead of a raw finite difference taken at the
recording rate, which would make the two episodes' peaks incomparable.

``calculate`` keeps only the scalar aggregates in the metric row. The per-sample kinematics
are published as one flat 1NF table by :meth:`MotionMetricsCalculator.series_table`; both
come from the same derivation, so the series and its summaries cannot disagree.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Final

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from arena_evaluation.storage.registry.series import SeriesMotionMetrics
from arena_evaluation.processing.metrics.common.alignment import interpolate_values

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from arena_evaluation.processing.metrics.common.series import null_float_series

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
MIN_DT_S: typing.Final[float] = 1e-6

SERIES_TABLE_STEM = "motion"

# Declared smoothing of the derivative estimator: what the reported acceleration and jerk peaks measure.
SMOOTHING_WINDOW_S: Final[float] = 1.0
SMOOTHING_POLYORDER: Final[int] = 2
KINEMATICS_GRID_HZ: Final[float] = 20.0
MIN_SMOOTHING_SAMPLES: Final[int] = 5

SERIES_SCHEMA: typing.Final[pl.Schema] = SeriesMotionMetrics.polars_schema()

_EPISODE_KEYS: Final[tuple[str, ...]] = (
    "mean_velocity_mps",
    "velocity_max",
    "acceleration_mean",
    "max_acceleration_mps2",
    "jerk_mean",
    "jerk_max_mps3",
)

ACCELERATION_MEAN = define_metric(
    "acceleration_mean",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s²",
    description="Mean smoothed ground-truth acceleration magnitude: Savitzky-Golay derivative of the recorded speed, 1.0 s window, order 2, at most 20 Hz; NULL, never a raw difference, when the window does not fit the recording.",
    null_reason=True,
)
JERK_MEAN = define_metric(
    "jerk_mean",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s³",
    description="Mean smoothed ground-truth jerk magnitude: Savitzky-Golay derivative of the recorded speed, 1.0 s window, order 2, at most 20 Hz; NULL, never a raw difference, when the window does not fit the recording.",
    null_reason=True,
)
JERK_MAX_MPS3 = define_metric(
    "jerk_max_mps3",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s³",
    description="Peak smoothed ground-truth jerk magnitude: Savitzky-Golay derivative of the recorded speed, 1.0 s window, order 2, at most 20 Hz; NULL, never a raw difference, when the window does not fit the recording.",
    null_reason=True,
)
MAX_ACCELERATION_MPS2 = define_metric(
    "max_acceleration_mps2",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s²",
    description="Peak smoothed ground-truth acceleration magnitude: Savitzky-Golay derivative of the recorded speed, 1.0 s window, order 2, at most 20 Hz; NULL, never a raw difference, when the window does not fit the recording.",
    null_reason=True,
)
MEAN_VELOCITY_MPS = define_metric(
    "mean_velocity_mps",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s",
    description="Mean ground-truth speed over the scope window's resolved samples.",
    null_reason=True,
)
VELOCITY_MAX = define_metric(
    "velocity_max",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m/s",
    description="Peak ground-truth speed over the scope window's resolved samples.",
    null_reason=True,
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Kinematics:
    """Speed, tangential acceleration and jerk on one time axis; undefined samples are NaN."""

    t_ns: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray


def _empty_kinematics() -> Kinematics:
    empty = np.array([], dtype=np.float64)
    return Kinematics(
        t_ns=np.array([], dtype=np.int64),
        speed_mps=empty,
        acceleration_mps2=empty.copy(),
        jerk_mps3=empty.copy(),
    )


def _dt_s(t_ns: np.ndarray) -> np.ndarray:
    """Non-negative sample intervals in seconds, a degenerate interval clipped to ``MIN_DT_S``."""
    dt = np.diff(t_ns) / NANOSECONDS_PER_SECOND
    return np.where(dt <= 0.0, MIN_DT_S, dt)


def _pose_track(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Ground-truth position samples of the scope view: the aligned native /tf frame, else the native track."""
    frame = calculator.resolve_robot_frame(episode)
    if frame is not None and frame.height >= 2:
        return (
            frame.get_column("t_ns").to_numpy().astype(np.int64),
            frame.get_column("x").to_numpy().astype(np.float64),
            frame.get_column("y").to_numpy().astype(np.float64),
        )
    tracks = calculator.native_topics(episode).get("robot_tracks")
    if tracks is None or tracks.height < 2:
        return None
    tracks = tracks.sort("t_ns")
    return (
        tracks.get_column("t_ns").to_numpy().astype(np.int64),
        tracks.get_column("x").cast(pl.Float64).to_numpy().astype(np.float64),
        tracks.get_column("y").cast(pl.Float64).to_numpy().astype(np.float64),
    )


def _unique_track(
    t_ns: np.ndarray, x_m: np.ndarray, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Samples in ascending time with one row per stamp and finite positions, None when fewer than two remain."""
    order = np.argsort(t_ns, kind="stable")
    time_sorted = t_ns[order]
    x_sorted = x_m[order]
    y_sorted = y_m[order]
    keep = np.concatenate(([True], np.diff(time_sorted) > 0))
    finite = np.isfinite(x_sorted) & np.isfinite(y_sorted)
    keep &= finite
    time_sorted, x_sorted, y_sorted = time_sorted[keep], x_sorted[keep], y_sorted[keep]
    if len(time_sorted) < 2:
        return None
    return time_sorted, x_sorted, y_sorted


def _speed_only_kinematics(speed_mps: np.ndarray, t_ns: np.ndarray) -> Kinematics:
    """Measured speed with undefined acceleration and jerk, where the smoothing window does not fit."""
    undefined = np.full(len(speed_mps), np.nan, dtype=np.float64)
    return Kinematics(
        t_ns=t_ns,
        speed_mps=speed_mps,
        acceleration_mps2=undefined,
        jerk_mps3=undefined.copy(),
    )


def _speed_from_pose(x_m: np.ndarray, y_m: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    """Ground-truth speed from one-sided position differences; NaN on the first sample, where no step exists."""
    speed = np.full(len(x_m), np.nan, dtype=np.float64)
    if len(x_m) >= 2:
        speed[1:] = np.hypot(np.diff(x_m), np.diff(y_m)) / _dt_s(t_ns)
    return speed


def _smooth_position_kinematics(
    t_ns: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> Kinematics | None:
    """Savitzky-Golay kinematics of a position track on a uniform grid, None when the window does not fit.

    Used where the recording has no speed stream: fitting the positions first keeps sample noise
    out of the derived speed instead of amplifying it through finite differences.
    """
    unique = _unique_track(t_ns, x_m, y_m)
    if unique is None:
        return None
    grid_time_ns, grid_x_m, grid_y_m = unique
    step_s = max(
        1.0 / KINEMATICS_GRID_HZ,
        float(np.median(np.diff(grid_time_ns))) / NANOSECONDS_PER_SECOND,
    )
    step_ns = max(1, int(round(step_s * NANOSECONDS_PER_SECOND)))
    grid = np.arange(grid_time_ns[0], grid_time_ns[-1] + 1, step_ns, dtype=np.int64)
    if len(grid) < MIN_SMOOTHING_SAMPLES:
        return None
    sample_step_s = step_ns / NANOSECONDS_PER_SECOND
    window = int(round(SMOOTHING_WINDOW_S / sample_step_s)) | 1
    window = min(window, (len(grid) // 2) * 2 - 1)
    if window < MIN_SMOOTHING_SAMPLES:
        return None
    grid_x = interpolate_values(grid_time_ns, grid_x_m, grid)
    grid_y = interpolate_values(grid_time_ns, grid_y_m, grid)
    velocity_x = savgol_filter(
        grid_x, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    velocity_y = savgol_filter(
        grid_y, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    speed_mps = np.hypot(velocity_x, velocity_y)
    acceleration_mps2 = savgol_filter(
        speed_mps, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    jerk_mps3 = savgol_filter(
        speed_mps, window, SMOOTHING_POLYORDER, deriv=2, delta=sample_step_s
    )
    return Kinematics(
        t_ns=grid,
        speed_mps=speed_mps,
        acceleration_mps2=acceleration_mps2,
        jerk_mps3=jerk_mps3,
    )


def _smooth_speed_kinematics(
    t_ns: np.ndarray,
    speed_mps: np.ndarray,
) -> Kinematics | None:
    """Savitzky-Golay derivatives of the recorded speed on a uniform grid, None when the window does not fit."""
    order = np.argsort(t_ns, kind="stable")
    time_sorted = t_ns[order]
    speed_sorted = speed_mps[order]
    keep = np.concatenate(([True], np.diff(time_sorted) > 0)) & np.isfinite(
        speed_sorted
    )
    time_sorted, speed_sorted = time_sorted[keep], speed_sorted[keep]
    if len(time_sorted) < 2:
        return None
    # Never interpolate finer than the recorded samples: a coarser recording keeps its own step,
    # a finer one is thinned to the declared grid rate.
    step_s = max(
        1.0 / KINEMATICS_GRID_HZ,
        float(np.median(np.diff(time_sorted))) / NANOSECONDS_PER_SECOND,
    )
    step_ns = max(1, int(round(step_s * NANOSECONDS_PER_SECOND)))
    grid = np.arange(time_sorted[0], time_sorted[-1] + 1, step_ns, dtype=np.int64)
    if len(grid) < MIN_SMOOTHING_SAMPLES:
        return None
    sample_step_s = step_ns / NANOSECONDS_PER_SECOND
    window = int(round(SMOOTHING_WINDOW_S / sample_step_s)) | 1
    window = min(window, (len(grid) // 2) * 2 - 1)
    if window < MIN_SMOOTHING_SAMPLES:
        return None
    grid_speed_mps = interpolate_values(time_sorted, speed_sorted, grid)
    acceleration_mps2 = savgol_filter(
        grid_speed_mps, window, SMOOTHING_POLYORDER, deriv=1, delta=sample_step_s
    )
    jerk_mps3 = savgol_filter(
        grid_speed_mps, window, SMOOTHING_POLYORDER, deriv=2, delta=sample_step_s
    )
    return Kinematics(
        t_ns=grid,
        speed_mps=grid_speed_mps,
        acceleration_mps2=acceleration_mps2,
        jerk_mps3=jerk_mps3,
    )


def episode_speed_mps(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> tuple[np.ndarray, np.ndarray]:
    """Recorded speed (m/s) and sample times (ns) of an episode, the speed-only fallback stream.

    Used where the recording carries a speed stream but no ground-truth position track.
    """
    frame = episode.data
    if frame is None or frame.height == 0:
        return np.array([]), np.array([], dtype=np.int64)
    try:
        frame = frame.select("t_ns", "speed_mps")
    except pl.exceptions.ColumnNotFoundError:
        return np.array([]), np.array([], dtype=np.int64)
    valid = pl.col("speed_mps").is_not_null() & ~pl.col("speed_mps").is_nan()
    frame = frame.filter(valid)
    if frame.height == 0:
        return np.array([]), np.array([], dtype=np.int64)
    return (
        np.abs(frame["speed_mps"].to_numpy().astype(np.float64)),
        frame["t_ns"].to_numpy().astype(np.int64),
    )


def episode_kinematics(
    calculator: BaseMetricCalculator, episode: AlignedEpisodeBundle
) -> Kinematics:
    """Speed, acceleration and jerk of the scope view, smoothed over the declared window when it fits.

    The recorded ground-truth speed stream is the signal (the same stream the metric always
    reported); an episode without it derives the speed from the ground-truth pose track.
    """
    speed_mps, t_ns = episode_speed_mps(calculator, episode)
    if len(speed_mps) > 0:
        smoothed = (
            _smooth_speed_kinematics(t_ns, speed_mps) if len(speed_mps) > 1 else None
        )
        return (
            smoothed
            if smoothed is not None
            else _speed_only_kinematics(speed_mps, t_ns)
        )

    track = _pose_track(calculator, episode)
    if track is None:
        return _empty_kinematics()
    unique = _unique_track(*track)
    if unique is None:
        return _empty_kinematics()
    unique_time_ns, unique_x_m, unique_y_m = unique
    smoothed = _smooth_position_kinematics(unique_time_ns, unique_x_m, unique_y_m)
    if smoothed is not None:
        return smoothed
    derived_speed_mps = _speed_from_pose(unique_x_m, unique_y_m, unique_time_ns)
    return _speed_only_kinematics(derived_speed_mps, unique_time_ns)


def _summarize(kinematics: Kinematics) -> dict[str, float | None]:
    """Velocity, acceleration and jerk aggregates of one kinematics track; explicit nulls without samples."""
    result: dict[str, float | None] = dict.fromkeys(_EPISODE_KEYS)
    speed_mps = kinematics.speed_mps[np.isfinite(kinematics.speed_mps)]
    if len(speed_mps) == 0:
        return result
    result["mean_velocity_mps"] = float(np.mean(speed_mps))
    result["velocity_max"] = float(np.max(speed_mps))
    acceleration_mps2 = kinematics.acceleration_mps2[
        np.isfinite(kinematics.acceleration_mps2)
    ]
    if len(acceleration_mps2) > 0:
        result["acceleration_mean"] = float(np.mean(np.abs(acceleration_mps2)))
        result["max_acceleration_mps2"] = float(np.max(np.abs(acceleration_mps2)))
    jerk_mps3 = kinematics.jerk_mps3[np.isfinite(kinematics.jerk_mps3)]
    if len(jerk_mps3) > 0:
        result["jerk_mean"] = float(np.mean(np.abs(jerk_mps3)))
        result["jerk_max_mps3"] = float(np.max(np.abs(jerk_mps3)))
    return result


def _motion_series_frame(
    kinematics: Kinematics,
    *,
    run_id: str,
    episode_id: int,
    robot_id: str,
) -> pl.DataFrame:
    """1NF motion series of one kinematics track, schema-typed when empty."""
    sample_count = len(kinematics.speed_mps)
    if sample_count == 0:
        return pl.DataFrame(schema=SERIES_SCHEMA)
    time_s = (
        kinematics.t_ns.astype(np.float64) - float(kinematics.t_ns[0])
    ) / NANOSECONDS_PER_SECOND
    return pl.DataFrame(
        {
            "run_id": pl.Series("run_id", [run_id] * sample_count, dtype=pl.String),
            "episode_id": pl.Series(
                "episode_id", [int(episode_id)] * sample_count, dtype=pl.Int64
            ),
            "robot_id": pl.Series(
                "robot_id", [robot_id] * sample_count, dtype=pl.String
            ),
            "t_ns": pl.Series("t_ns", kinematics.t_ns, dtype=pl.Int64),
            "time_s": pl.Series("time_s", time_s, dtype=pl.Float64),
            "speed_mps": null_float_series(kinematics.speed_mps, "speed_mps"),
            "acceleration_mps2": null_float_series(
                kinematics.acceleration_mps2, "acceleration_mps2"
            ),
            "jerk_mps3": null_float_series(kinematics.jerk_mps3, "jerk_mps3"),
        },
        schema=SERIES_SCHEMA,
    )


class MotionMetricsCalculator(BaseMetricCalculator):
    """Robot velocity, acceleration and jerk over the scope window's ground truth.

    Acceleration and jerk are local-polynomial derivatives over a declared
    ``SMOOTHING_WINDOW_S`` (1.0 s) window at ``SMOOTHING_POLYORDER`` 2, so the peaks
    measure the smoothed trajectory rather than sample-rate noise. That is the only
    derivative estimator: a recording too short for the window publishes NULL acceleration
    and jerk, never a raw difference.
    """

    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]

    OUTPUTS = (
        MEAN_VELOCITY_MPS,
        VELOCITY_MAX,
        ACCELERATION_MEAN,
        MAX_ACCELERATION_MPS2,
        JERK_MEAN,
        JERK_MAX_MPS3,
    )
    DIRECTIONS = {
        "acceleration_mean": Direction.LOWER,
        "max_acceleration_mps2": Direction.LOWER,
        "jerk_mean": Direction.LOWER,
        "jerk_max_mps3": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = [
        "mean_velocity_mps",
        "velocity_max",
        "jerk_mean",
        "jerk_max_mps3",
    ]

    @classmethod
    def required_topics_for_scope(
        cls, scope: MetricScope
    ) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter.
        return (
            ()
            if scope == MetricScope.ENCOUNTER
            else super().required_topics_for_scope(scope)
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        return _summarize(episode_kinematics(self, episode))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        del prior_results
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return _summarize(episode_kinematics(self, episode))
        raise ValueError(
            f"motion_metrics does not serve the {scope_slice.scope.value} scope"
        )

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat motion series, one row per kinematics sample; schema-typed when empty."""
        return _motion_series_frame(
            episode_kinematics(self, episode),
            run_id=run_id,
            episode_id=int(episode.episode_id),
            robot_id=robot_id or (episode.robot_name or "robot"),
        )
