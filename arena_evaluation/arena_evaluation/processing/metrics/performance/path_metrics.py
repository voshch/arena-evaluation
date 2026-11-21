"""Trajectory geometry over the native /tf ground-truth pose track."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import polars as pl

from arena_evaluation.storage.registry.series import SeriesPathMetrics
from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from arena_evaluation.processing.metrics.common.series import null_float_series

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
MIN_STEP_M: typing.Final[float] = (
    1e-3  # consecutive samples closer than this are one point
)
MIN_CURVATURE_POINTS: typing.Final[int] = 3

SERIES_SCHEMA: typing.Final[pl.Schema] = SeriesPathMetrics.polars_schema()

_GEOMETRY_KEYS: typing.Final[tuple[str, ...]] = (
    "path_length_m",
    "curvature_mean",
    "curvature_max",
    "roughness_mean",
    "roughness_max",
    "angle_over_length",
)

ANGLE_OVER_LENGTH = define_metric(
    "angle_over_length",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="rad/m",
    description="Absolute net heading change over the travelled ground-truth path length; NULL when the path has no length.",
    null_reason=True,
)
CURVATURE_MAX = define_metric(
    "curvature_max",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1/m",
    description="Peak Menger curvature over the distinct ground-truth path triples.",
    null_reason=True,
)
CURVATURE_MEAN = define_metric(
    "curvature_mean",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1/m",
    description="Mean Menger curvature over the distinct ground-truth path triples; NULL without a valid triple.",
    null_reason=True,
)
PATH_LENGTH_M = define_metric(
    "path_length_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m",
    description="Travelled planar ground-truth path length over the scope window; 0.0 only for a recorded stationary track and NULL without a sample.",
    null_reason=True,
)
ROUGHNESS_MAX = define_metric(
    "roughness_max",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="rad/m",
    description="Peak triangle roughness over the distinct ground-truth path triples.",
    null_reason=True,
)
ROUGHNESS_MEAN = define_metric(
    "roughness_mean",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="rad/m",
    description="Mean triangle roughness over the distinct ground-truth path triples; NULL without a valid triple.",
    null_reason=True,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PathGeometry:
    """Per-sample path arrays of the native /tf track and the scalar aggregates they yield."""

    t_ns: np.ndarray
    pos_x_m: np.ndarray
    pos_y_m: np.ndarray
    yaw_rad: np.ndarray
    odom_x_m: np.ndarray
    odom_y_m: np.ndarray
    odom_yaw_rad: np.ndarray
    step_length_m: np.ndarray
    curvature_inv_m: np.ndarray
    normalized_curvature: np.ndarray
    roughness: np.ndarray
    path_length_m: float | None
    curvature_mean_inv_m: float | None
    curvature_max_inv_m: float | None
    roughness_mean: float | None
    roughness_max: float | None
    angle_over_length_rad_m: float | None


def _nan_array(length: int) -> np.ndarray:
    return np.full(length, np.nan, dtype=np.float64)


def _mean_max(values: np.ndarray) -> tuple[float | None, float | None]:
    """Mean and maximum of the defined samples, both None when there is no defined sample."""
    defined = values[~np.isnan(values)]
    if len(defined) == 0:
        return None, None
    return float(np.mean(defined)), float(np.max(defined))


def _angle_difference_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Signed shortest rotation between two headings, in radians."""
    return np.pi - np.abs(np.abs(first - second) - np.pi)


def _curvature_arrays(
    pos_x_m: np.ndarray, pos_y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample Menger curvature, normalized curvature and roughness, NaN where a triple is undefined.

    Curvature is defined at the middle sample of three consecutive distinct points, so the
    first and last samples stay NaN, as do samples that repeat the previous position.
    """
    sample_count = len(pos_x_m)
    curvature_inv_m = _nan_array(sample_count)
    normalized_curvature = _nan_array(sample_count)
    roughness = _nan_array(sample_count)
    if sample_count < MIN_CURVATURE_POINTS:
        return curvature_inv_m, normalized_curvature, roughness

    path_2d = np.column_stack((pos_x_m, pos_y_m))
    step_length_m = np.linalg.norm(np.diff(path_2d, axis=0), axis=1)
    keep = np.concatenate(([True], step_length_m > MIN_STEP_M))
    clean_path = path_2d[keep]
    if len(clean_path) < MIN_CURVATURE_POINTS:
        return curvature_inv_m, normalized_curvature, roughness

    p0, p1, p2 = clean_path[:-2], clean_path[1:-1], clean_path[2:]
    d01 = np.linalg.norm(p0 - p1, axis=1)
    d12 = np.linalg.norm(p1 - p2, axis=1)
    d20 = np.linalg.norm(p2 - p0, axis=1)
    v1 = p1 - p0
    v2 = p2 - p0
    triangle_area = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2.0
    divisor = d01 * d12 * d20
    with np.errstate(divide="ignore", invalid="ignore"):
        clean_curvature = np.where(divisor == 0.0, 0.0, 4.0 * triangle_area / divisor)
        clean_roughness = np.where(
            d20 == 0.0, 0.0, 2.0 * triangle_area / np.square(d20)
        )
    middle_samples = np.flatnonzero(keep)[1:-1]
    curvature_inv_m[middle_samples] = clean_curvature
    normalized_curvature[middle_samples] = clean_curvature * (d01 + d12)
    roughness[middle_samples] = clean_roughness
    return curvature_inv_m, normalized_curvature, roughness


def _build_geometry(
    *,
    t_ns: np.ndarray,
    pos_x_m: np.ndarray,
    pos_y_m: np.ndarray,
    yaw_rad: np.ndarray,
    odom_x_m: np.ndarray,
    odom_y_m: np.ndarray,
    odom_yaw_rad: np.ndarray,
) -> _PathGeometry:
    """Per-sample path arrays and their scalar aggregates, None wherever the geometry is undefined."""
    sample_count = len(pos_x_m)
    step_length_m = _nan_array(sample_count)
    if sample_count >= 2:
        step_length_m[1:] = np.hypot(np.diff(pos_x_m), np.diff(pos_y_m))
    curvature_inv_m, normalized_curvature, roughness = _curvature_arrays(
        pos_x_m, pos_y_m
    )

    path_length_m = float(np.nansum(step_length_m)) if sample_count > 0 else None
    angle_over_length_rad_m = None
    if sample_count >= 2 and path_length_m > 0.0:
        turn_rad = _angle_difference_rad(yaw_rad[:-1], yaw_rad[1:])
        angle_over_length_rad_m = float(np.abs(np.sum(turn_rad)) / path_length_m)
    curvature_mean_inv_m, curvature_max_inv_m = _mean_max(curvature_inv_m)
    roughness_mean, roughness_max = _mean_max(roughness)
    return _PathGeometry(
        t_ns=t_ns,
        pos_x_m=pos_x_m,
        pos_y_m=pos_y_m,
        yaw_rad=yaw_rad,
        odom_x_m=odom_x_m,
        odom_y_m=odom_y_m,
        odom_yaw_rad=odom_yaw_rad,
        step_length_m=step_length_m,
        curvature_inv_m=curvature_inv_m,
        normalized_curvature=normalized_curvature,
        roughness=roughness,
        path_length_m=path_length_m,
        curvature_mean_inv_m=curvature_mean_inv_m,
        curvature_max_inv_m=curvature_max_inv_m,
        roughness_mean=roughness_mean,
        roughness_max=roughness_max,
        angle_over_length_rad_m=angle_over_length_rad_m,
    )


class PathMetricsCalculator(BaseMetricCalculator):
    """Path length, curvature and roughness of the ground-truth trajectory.

    Metrics:
    - path_length_m: total travelled path length at the scope it is evaluated on
    - curvature_mean, curvature_max: Menger curvature at each distinct triple, aggregated
    - roughness_mean, roughness_max: triangle roughness, aggregated
    - angle_over_length: net heading change over the complete path

    The per-sample ``path``/``path_odom``/``path_length_values``/``curvature``/
    ``normalized_curvature``/``roughness`` keys live in :meth:`series_table`.

    References:
    - Math implementation ported from legacy scripts/metrics.py
    """

    NAME = "path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks"]

    OUTPUTS = (
        PATH_LENGTH_M,
        CURVATURE_MEAN,
        CURVATURE_MAX,
        ROUGHNESS_MEAN,
        ROUGHNESS_MAX,
        ANGLE_OVER_LENGTH,
    )
    DIRECTIONS = {
        "curvature_mean": Direction.LOWER,
        "path_length_m": Direction.LOWER,
        "roughness_mean": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["path_length_m", "curvature_mean", "roughness_mean"]

    def _path_geometry(self, episode: AlignedEpisodeBundle) -> _PathGeometry | None:
        """Per-sample geometry of the resolved track, None without any aligned episode frame."""
        if episode.data is None:
            return None
        frame = self.resolve_robot_frame(episode)
        pos_x_m, pos_y_m, yaw_rad, odom_x_m, odom_y_m, odom_yaw_rad = (
            self.resolve_robot_pose(episode)
        )
        t_ns = (
            np.array([], dtype=np.int64)
            if frame is None
            else frame["t_ns"].to_numpy().astype(np.int64)
        )
        if len(t_ns) != len(pos_x_m):
            raise ValueError(
                f"path_metrics: {len(t_ns)} sample times for {len(pos_x_m)} resolved poses"
            )
        if len(odom_x_m) != len(pos_x_m):
            # No wheel-odometry stream: the odom columns are null at every sample, not absent from the row axis.
            odom_x_m, odom_y_m, odom_yaw_rad = (
                _nan_array(len(pos_x_m)),
                _nan_array(len(pos_x_m)),
                _nan_array(len(pos_x_m)),
            )
        return _build_geometry(
            t_ns=t_ns,
            pos_x_m=pos_x_m,
            pos_y_m=pos_y_m,
            yaw_rad=yaw_rad,
            odom_x_m=odom_x_m,
            odom_y_m=odom_y_m,
            odom_yaw_rad=odom_yaw_rad,
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        geometry = self._path_geometry(episode)
        if geometry is None:
            return dict.fromkeys(_GEOMETRY_KEYS)
        return {
            "path_length_m": geometry.path_length_m,
            "curvature_mean": geometry.curvature_mean_inv_m,
            "curvature_max": geometry.curvature_max_inv_m,
            "roughness_mean": geometry.roughness_mean,
            "roughness_max": geometry.roughness_max,
            "angle_over_length": geometry.angle_over_length_rad_m,
        }

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
            geometry = self._path_geometry(episode)
            return {
                "path_length_m": None if geometry is None else geometry.path_length_m
            }
        raise ValueError(
            f"path_metrics does not serve the {scope_slice.scope.value} scope"
        )

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat path series, one row per resolved track sample; schema-typed when empty."""
        geometry = self._path_geometry(episode)
        if geometry is None or len(geometry.t_ns) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        sample_count = len(geometry.t_ns)
        time_s = (
            geometry.t_ns.astype(np.float64) - float(geometry.t_ns[0])
        ) / NANOSECONDS_PER_SECOND
        return pl.DataFrame(
            {
                "run_id": pl.Series("run_id", [run_id] * sample_count, dtype=pl.String),
                "episode_id": pl.Series(
                    "episode_id",
                    [int(episode.episode_id)] * sample_count,
                    dtype=pl.Int64,
                ),
                "robot_id": pl.Series(
                    "robot_id",
                    [robot_id or (episode.robot_name or "robot")] * sample_count,
                    dtype=pl.String,
                ),
                "t_ns": pl.Series("t_ns", geometry.t_ns, dtype=pl.Int64),
                "time_s": pl.Series("time_s", time_s, dtype=pl.Float64),
                "x_m": null_float_series(geometry.pos_x_m, "x_m"),
                "y_m": null_float_series(geometry.pos_y_m, "y_m"),
                "yaw_rad": null_float_series(geometry.yaw_rad, "yaw_rad"),
                "odom_x_m": null_float_series(geometry.odom_x_m, "odom_x_m"),
                "odom_y_m": null_float_series(geometry.odom_y_m, "odom_y_m"),
                "odom_yaw_rad": null_float_series(
                    geometry.odom_yaw_rad, "odom_yaw_rad"
                ),
                "step_length_m": null_float_series(
                    geometry.step_length_m, "step_length_m"
                ),
                "curvature_inv_m": null_float_series(
                    geometry.curvature_inv_m, "curvature_inv_m"
                ),
                "normalized_curvature": null_float_series(
                    geometry.normalized_curvature, "normalized_curvature"
                ),
                "roughness": null_float_series(geometry.roughness, "roughness"),
            },
            schema=SERIES_SCHEMA,
        )
