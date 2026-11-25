"""Mutual gaze cone metrics on the robot sample axis, with per-sample counts as a 1NF series."""

from __future__ import annotations

import math
import typing
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, ScopeSlice
from arena_evaluation.processing.metrics.social.interaction_common import (
    native_robot_frame,
)
from arena_evaluation.processing.pedestrian_frames import pedestrians_at
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.registry.series import SeriesGaze
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

GAZE_CONE_RAD: Final[float] = math.radians(5.0)
FULL_TURN_RAD: Final[float] = math.pi
SAMPLE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "t_ns": pl.Int64,
        "looking_at_peds": pl.Int64,
        "looked_at_by_peds": pl.Int64,
    }
)

ROBOT_SAMPLE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "t_ns": pl.Int64,
        "robot_x_m": pl.Float64,
        "robot_y_m": pl.Float64,
        "robot_yaw_rad": pl.Float64,
    }
)

SERIES_SCHEMA: Final[pl.Schema] = SeriesGaze.polars_schema()

TOTAL_TIME_LOOKING_AT_PEDESTRIANS = define_metric(
    "total_time_looking_at_pedestrians",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="s",
    description="Seconds of robot time whose 5 degree gaze cone contained at least one pedestrian.",
    null_reason=True,
)
TOTAL_TIME_LOOKED_AT_BY_PEDESTRIANS = define_metric(
    "total_time_looked_at_by_pedestrians",
    pl.Float64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="s",
    description="Seconds of robot time at which at least one pedestrian's 5 degree gaze cone contained the robot.",
    null_reason=True,
)


def _angle_difference(left_rad: pl.Expr, right_rad: pl.Expr) -> pl.Expr:
    """Absolute angular distance of two headings in [0, pi]."""
    return FULL_TURN_RAD - ((left_rad - right_rad).abs() - FULL_TURN_RAD).abs()


def gaze_counts(robot: pl.DataFrame, peds: pl.DataFrame | None) -> pl.DataFrame:
    """Per robot sample: how many pedestrians it looks at, and how many look back."""
    if peds is None or peds.height == 0 or robot.height == 0:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    matched = pedestrians_at(robot.get_column("t_ns"), peds).join(robot, on="t_ns", how="inner")
    if matched.height == 0:
        return pl.DataFrame(schema=SAMPLE_SCHEMA)
    directions = matched.with_columns(
        dx_m=pl.col("x").cast(pl.Float64) - pl.col("robot_x_m"),
        dy_m=pl.col("y").cast(pl.Float64) - pl.col("robot_y_m"),
    ).with_columns(
        angle_to_ped_rad=pl.arctan2(pl.col("dy_m"), pl.col("dx_m")),
        angle_to_robot_rad=pl.arctan2(-pl.col("dy_m"), -pl.col("dx_m")),
    )
    looking_at = _angle_difference(pl.col("robot_yaw_rad"), pl.col("angle_to_ped_rad")) <= GAZE_CONE_RAD
    looked_at = pl.col("yaw").is_not_null() & (_angle_difference(pl.col("yaw").cast(pl.Float64), pl.col("angle_to_robot_rad")) <= GAZE_CONE_RAD)
    return (
        directions.with_columns(looking_at=looking_at, looked_at=looked_at)
        .group_by("t_ns")
        .agg(
            looking_at_peds=pl.col("looking_at").sum().cast(pl.Int64),
            looked_at_by_peds=pl.col("looked_at").sum().cast(pl.Int64),
        )
        .select(
            t_ns=pl.col("t_ns"),
            looking_at_peds=pl.col("looking_at_peds"),
            looked_at_by_peds=pl.col("looked_at_by_peds"),
        )
    )


def time_with_cone_contact_s(samples: pl.DataFrame, column: str) -> float | None:
    """Seconds of robot time whose count in ``column`` is positive, over the forward sample steps."""
    if samples.height == 0:
        return None
    steps = samples.sort("t_ns").with_columns(step_s=(pl.col("t_ns").shift(-1) - pl.col("t_ns")).cast(pl.Float64) / 1e9)
    return float(steps.select(pl.when((pl.col(column) > 0) & pl.col("step_s").is_not_null()).then(pl.col("step_s")).otherwise(0.0).sum()).item())


class GazeMetricsCalculator(BaseMetricCalculator):
    """Robot-to-pedestrian and pedestrian-to-robot gaze cone dwell."""

    NAME = "gaze_metrics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_arena_peds"]

    OUTPUTS = (TOTAL_TIME_LOOKING_AT_PEDESTRIANS, TOTAL_TIME_LOOKED_AT_BY_PEDESTRIANS)
    PRIMARY_OUTPUTS = [
        "total_time_looking_at_pedestrians",
        "total_time_looked_at_by_pedestrians",
    ]

    def _robot_rows(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Ground-truth robot rows of the episode view, one per aligned sample."""
        frame = self.resolve_robot_frame(episode)
        if frame is None or frame.height == 0:
            return pl.DataFrame(schema=ROBOT_SAMPLE_SCHEMA)
        return frame.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            robot_x_m=pl.col("x").cast(pl.Float64),
            robot_y_m=pl.col("y").cast(pl.Float64),
            robot_yaw_rad=pl.col("yaw").cast(pl.Float64),
        ).drop_nulls(["t_ns", "robot_x_m", "robot_y_m", "robot_yaw_rad"])

    def _native_robot_rows(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """The window's own robot samples in the native topic frame, one row per sample."""
        frame = native_robot_frame(episode)
        if frame is None or frame.height == 0:
            return pl.DataFrame(schema=ROBOT_SAMPLE_SCHEMA)
        return frame.select(
            t_ns=pl.col("t_ns"),
            robot_x_m=pl.col("x"),
            robot_y_m=pl.col("y"),
            robot_yaw_rad=pl.col("yaw"),
        )

    @staticmethod
    def _samples_on_robot_axis(robot: pl.DataFrame, peds: pl.DataFrame | None) -> pl.DataFrame:
        """Per robot sample gaze counts, zero where the sample matched no pedestrian of the population."""
        if robot.height == 0:
            return pl.DataFrame(schema=SAMPLE_SCHEMA)
        counts = gaze_counts(robot, peds)
        return (
            robot.select(pl.col("t_ns").alias("t_ns"))
            .join(counts, on="t_ns", how="left")
            .select(
                t_ns=pl.col("t_ns"),
                looking_at_peds=pl.coalesce(pl.col("looking_at_peds"), pl.lit(0, dtype=pl.Int64)).cast(pl.Int64),
                looked_at_by_peds=pl.coalesce(pl.col("looked_at_by_peds"), pl.lit(0, dtype=pl.Int64)).cast(pl.Int64),
            )
        )

    def _samples(self, episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Per robot sample gaze counts over the episode view's own pedestrian stream."""
        return self._samples_on_robot_axis(self._robot_rows(episode), self.native_ped_frame(episode))

    def _metrics_from_samples(self, samples: pl.DataFrame) -> dict[str, typing.Any]:
        """Both gaze dwell totals over one per-sample count frame; empty samples stay null."""
        if samples.height == 0:
            return dict.fromkeys(self.output_keys())
        return {
            "total_time_looking_at_pedestrians": time_with_cone_contact_s(samples, "looking_at_peds"),
            "total_time_looked_at_by_pedestrians": time_with_cone_contact_s(samples, "looked_at_by_peds"),
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Compute the episode-scope gaze dwell over the bundle's own pedestrian frames."""
        del prior_results
        return self._metrics_from_samples(self._samples(episode))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE and ROBOT measure the whole episode on the ground-truth axis; an ENCOUNTER slice measures its own window telemetry."""
        del prior_results
        if scope_slice is None:
            return self._metrics_from_samples(self._samples(episode))
        match scope_slice.scope:
            case MetricScope.EPISODE | MetricScope.ROBOT:
                return self._metrics_from_samples(self._samples(episode))
            case MetricScope.ENCOUNTER:
                window = scope_slice.window_bundle
                return self._metrics_from_samples(self._samples_on_robot_axis(self._native_robot_rows(window), self.native_ped_frame(window)))
            case _:
                return dict.fromkeys(self.output_keys())

    SERIES_TABLE_STEM = "gaze"

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-sample gaze counts, zero where the robot had no pedestrian frame."""
        samples = self._samples(episode)
        if samples.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(samples.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return samples.select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / 1e9,
            looking_at_peds=pl.col("looking_at_peds"),
            looked_at_by_peds=pl.col("looked_at_by_peds"),
        )
