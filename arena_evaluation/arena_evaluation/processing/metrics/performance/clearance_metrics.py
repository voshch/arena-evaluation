from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.storage.registry.series import SeriesClearance
from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
    asof_join,
)
from arena_evaluation.processing.metrics.common.series import null_float_series
from arena_evaluation.processing.pedestrian_frames import distances_to_pedestrians
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_RANGE_EPSILON_M: typing.Final[float] = 1e-6

SERIES_SCHEMA: typing.Final[pl.Schema] = SeriesClearance.polars_schema()

MEAN_OBSTACLE_CLEARANCE = define_metric(
    "mean_obstacle_clearance",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m",
    description="Mean edge-to-edge clearance to the nearest detected obstacle over the scope window.",
    null_reason=True,
)
MEAN_PEDESTRIAN_CLEARANCE = define_metric(
    "mean_pedestrian_clearance",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m",
    description="Mean edge-to-edge clearance to the nearest pedestrian over the scope window, both recorded radii subtracted.",
    null_reason=True,
)
MIN_OBSTACLE_CLEARANCE_M = define_metric(
    "min_obstacle_clearance_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m",
    description="Closest edge-to-edge clearance to a detected obstacle over the scope window, robot radius subtracted.",
    null_reason=True,
)
MIN_PED_CLEARANCE_M = define_metric(
    "min_ped_clearance_m",
    pl.Float64,
    scopes=(MetricScope.ENCOUNTER, MetricScope.ROBOT),
    unit="m",
    description="Closest edge-to-edge clearance to the nearest pedestrian over the scope window, both recorded radii subtracted.",
    null_reason=True,
)


def _summary(values: np.ndarray) -> tuple[float | None, float | None]:
    """Minimum and mean of the non-NaN values, (None, None) when every value is NaN."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return None, None
    return float(np.min(valid)), float(np.mean(valid))


def _summaries(
    obstacle_clearance_m: np.ndarray, pedestrian_clearance_m: np.ndarray
) -> dict[str, float | None]:
    """Minimum and mean of the valid obstacle and pedestrian clearances, None without a single valid sample."""
    min_obstacle, mean_obstacle = _summary(obstacle_clearance_m)
    min_pedestrian, mean_pedestrian = _summary(pedestrian_clearance_m)
    return {
        "min_obstacle_clearance_m": min_obstacle,
        "mean_obstacle_clearance": mean_obstacle,
        "min_ped_clearance_m": min_pedestrian,
        "mean_pedestrian_clearance": mean_pedestrian,
    }


def _empty_clearance_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Empty times and combined, obstacle and pedestrian clearances for a window without a usable pose."""
    empty = np.array([], dtype=np.float64)
    return np.array([], dtype=np.int64), empty, empty.copy(), empty.copy()


class ClearanceMetricsCalculator(BaseMetricCalculator):
    """Minimum and mean clearance distances to obstacles and pedestrians over the scope window."""

    NAME = "clearance_metrics"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_scan_summary", "pedestrian_arena_peds")]

    OUTPUTS = (
        MIN_OBSTACLE_CLEARANCE_M,
        MEAN_OBSTACLE_CLEARANCE,
        MIN_PED_CLEARANCE_M,
        MEAN_PEDESTRIAN_CLEARANCE,
    )
    DIRECTIONS = {
        "min_obstacle_clearance_m": Direction.HIGHER,
        "mean_obstacle_clearance": Direction.HIGHER,
        "min_ped_clearance_m": Direction.HIGHER,
        "mean_pedestrian_clearance": Direction.HIGHER,
    }
    PRIMARY_OUTPUTS = ["min_obstacle_clearance_m", "min_ped_clearance_m"]

    SERIES_TABLE_STEM = "clearance"

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

    @staticmethod
    def _scan_arrays(
        scan_df: pl.DataFrame | None, t_ns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scan minimum and maximum range interpolated onto ``t_ns``, NaN where no scan sample is in tolerance."""
        if scan_df is None or len(scan_df) == 0:
            empty = np.full(len(t_ns), np.nan)
            return empty, empty.copy()
        scan = (
            scan_df.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("scan_min_m").cast(pl.Float64),
                pl.col("range_max_m").cast(pl.Float64),
            )
            .drop_nulls("t_ns")
            .sort("t_ns")
        )
        joined = asof_join(
            pl.DataFrame({"t_ns": t_ns.astype(np.int64)}),
            scan,
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        ).sort("t_ns")
        scan_min_m = joined["scan_min_m"].to_numpy().astype(np.float64)
        scan_max_m = joined["range_max_m"].to_numpy().astype(np.float64)
        return scan_min_m, scan_max_m

    def _pedestrian_clearance(
        self,
        t_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        peds_df: pl.DataFrame | None,
    ) -> np.ndarray:
        """Per-sample clearance to the nearest pedestrian, each body's recorded radius subtracted.

        A nearest pedestrian whose radius is null or NaN raises :class:`MetricInputError` because the
        clearance would be an invented number.
        """
        if peds_df is None or len(peds_df) == 0:
            return np.full(len(t_ns), np.nan)
        nearest = (
            distances_to_pedestrians(
                t_ns, pos_x, pos_y, peds_df, TRACK_ALIGN_TOLERANCE_NS
            )
            .sort(["row", "distance"])
            .group_by("row", maintain_order=True)
            .first()
        )
        radii = peds_df.select(
            ped_id=pl.col("ped_id").cast(pl.String),
            radius_m=pl.col("radius_m").cast(pl.Float64),
        ).unique(subset=["ped_id"], keep="first", maintain_order=True)
        nearest = nearest.join(radii, on="ped_id", how="left")
        unresolved = nearest.filter(
            pl.col("radius_m").is_null() | pl.col("radius_m").is_nan()
        )
        if unresolved.height > 0:
            missing = sorted({str(ped_id) for ped_id in unresolved["ped_id"]})
            raise MetricInputError(
                f"clearance_metrics: pedestrian_arena_peds has no recorded radius_m for the nearest pedestrians {missing}"
            )
        distance_m = np.full(len(t_ns), np.nan)
        radius_m = np.full(len(t_ns), np.nan)
        rows = nearest["row"].to_numpy()
        distance_m[rows] = nearest["distance"].to_numpy().astype(np.float64)
        radius_m[rows] = nearest["radius_m"].to_numpy().astype(np.float64)
        return np.where(
            np.isnan(distance_m),
            np.nan,
            np.maximum(0.0, distance_m - self.robot_params.robot_radius - radius_m),
        )

    def _clearance_arrays(
        self,
        t_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        scan_df: pl.DataFrame | None,
        peds_df: pl.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-sample obstacle and pedestrian clearance in metres, NaN where that source has no valid sample."""
        robot_radius_m = self.robot_params.robot_radius
        scan_min_m, scan_max_m = self._scan_arrays(scan_df, t_ns)
        undetected = np.isnan(scan_max_m) | (scan_min_m < scan_max_m - _RANGE_EPSILON_M)
        obstacle_valid = np.isfinite(scan_min_m) & (scan_min_m > 0.0) & undetected
        obstacle_clearance_m = np.where(
            obstacle_valid, np.maximum(0.0, scan_min_m - robot_radius_m), np.nan
        )
        pedestrian_clearance_m = self._pedestrian_clearance(t_ns, pos_x, pos_y, peds_df)
        return obstacle_clearance_m, pedestrian_clearance_m

    @staticmethod
    def _combined(
        obstacle_clearance_m: np.ndarray, pedestrian_clearance_m: np.ndarray
    ) -> np.ndarray:
        """Sample-wise minimum of both sources, whichever is valid, NaN without either."""
        return np.fmin(obstacle_clearance_m, pedestrian_clearance_m)

    def _episode_arrays(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Aligned sample times, ground-truth robot position and the two clearance series of the episode."""
        if episode.data is None or len(episode.data) == 0:
            return _empty_clearance_arrays()

        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(
            episode
        )
        frame = self.resolve_robot_frame(episode)
        if frame is None or len(frame) == 0 or len(pos_x) == 0:
            return _empty_clearance_arrays()

        aligned_times_ns = frame["t_ns"].to_numpy().astype(np.int64)
        topic_scan = self.native_topics(episode).get("robot_scan_summary")
        peds_df = self.native_ped_frame(episode)
        obstacle, pedestrian = self._clearance_arrays(
            aligned_times_ns, pos_x, pos_y, topic_scan, peds_df
        )
        return (
            aligned_times_ns,
            self._combined(obstacle, pedestrian),
            obstacle,
            pedestrian,
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        _times, _combined, obstacle, pedestrian = self._episode_arrays(episode)
        return _summaries(obstacle, pedestrian)

    def _native_arrays(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Native-clock sample times, ground-truth robot position and the two clearance series of a scope window."""
        pos_x, pos_y, _yaw, t_ns = self.resolve_native_pose(episode)
        if len(t_ns) == 0:
            return _empty_clearance_arrays()
        topics = self.native_topics(episode)
        obstacle, pedestrian = self._clearance_arrays(
            t_ns,
            pos_x,
            pos_y,
            topics.get("robot_scan_summary"),
            self.native_ped_frame(episode),
        )
        return t_ns, self._combined(obstacle, pedestrian), obstacle, pedestrian

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            _times, _combined, obstacle, pedestrian = self._native_arrays(episode)
            return _summaries(obstacle, pedestrian)
        raise ValueError(
            f"clearance_metrics does not serve the {scope_slice.scope.value} scope"
        )

    def series_table(
        self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = ""
    ) -> pl.DataFrame:
        """1NF flat clearance series, one row per sample, explicit nulls where a source has no valid sample."""
        times_ns, combined, obstacle, pedestrian = self._episode_arrays(episode)
        if len(times_ns) == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        return pl.DataFrame(
            {
                "run_id": pl.Series(
                    "run_id", [run_id] * len(times_ns), dtype=pl.String
                ),
                "episode_id": pl.Series(
                    "episode_id",
                    [int(episode.episode_id)] * len(times_ns),
                    dtype=pl.Int64,
                ),
                "robot_id": pl.Series(
                    "robot_id",
                    [robot_id or (episode.robot_name or "robot")] * len(times_ns),
                    dtype=pl.String,
                ),
                "t_ns": pl.Series("t_ns", times_ns, dtype=pl.Int64),
                "obstacle_clearance_m": null_float_series(
                    obstacle, "obstacle_clearance_m"
                ),
                "pedestrian_clearance_m": null_float_series(
                    pedestrian, "pedestrian_clearance_m"
                ),
                "combined_clearance_m": null_float_series(
                    combined, "combined_clearance_m"
                ),
            },
            schema=SERIES_SCHEMA,
        )
