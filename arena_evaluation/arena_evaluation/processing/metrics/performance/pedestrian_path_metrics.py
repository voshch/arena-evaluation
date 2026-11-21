"""Pedestrian trajectories as a flat 1NF series table."""

from __future__ import annotations

import typing

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.registry.series import SeriesPedestrianPathMetrics
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9

SERIES_SCHEMA: typing.Final[pl.Schema] = SeriesPedestrianPathMetrics.polars_schema()

_REQUIRED_TRACK_COLUMNS: typing.Final[frozenset[str]] = frozenset({"t_ns", "ped_id", "x", "y"})


class PedestrianPathMetricsCalculator(BaseMetricCalculator):
    """Time-synchronized pedestrian trajectories, published only as the ``series_pedestrian_path`` table."""

    NAME = "pedestrian_path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["pedestrian_arena_peds"]
    REQUIRES_PEDSIM = True

    OUTPUTS = ()

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del episode, prior_results
        return {}

    @staticmethod
    def _optional_track_column(tracks: pl.DataFrame, column: str) -> pl.Expr:
        """Optional telemetry column cast to Float64, null when the recording predates it."""
        # agent-contracts: ignore MCAP schema evolution
        if column not in tracks.columns:
            return pl.lit(None, dtype=pl.Float64)
        return pl.col(column).cast(pl.Float64)

    def series_table(self, episode: AlignedEpisodeBundle, run_id: str = "", robot_id: str = "") -> pl.DataFrame:
        """1NF flat pedestrian path series, one row per recorded pedestrian sample; schema-typed when empty.

        The series holds no held last position: consumers must use an as-of join to read a
        pedestrian at another time. Duplicate recorded telemetry samples for the same key
        collapse to the last row, so the published natural key is honest.
        """
        tracks = self.native_ped_frame(episode)
        if tracks is None or tracks.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        missing = sorted(_REQUIRED_TRACK_COLUMNS - set(tracks.columns))
        if missing:
            raise ValueError(f"pedestrian_arena_peds lacks {missing}; the telemetry contract is (t_ns, ped_id, x, y, z, yaw, vx, vy, speed_mps)")

        present = tracks.filter(pl.col("x").is_not_null() & ~pl.col("x").is_nan() & pl.col("y").is_not_null() & ~pl.col("y").is_nan()).sort(["t_ns", "ped_id"])
        if present.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)

        present = present.unique(subset=["t_ns", "ped_id"], keep="last", maintain_order=True)
        min_t = present["t_ns"].min()
        if not isinstance(min_t, (int, float)):
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(min_t)
        return present.select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(robot_id or (episode.robot_name or "robot"), dtype=pl.String),
            t_ns=pl.col("t_ns").cast(pl.Int64),
            time_s=(pl.col("t_ns").cast(pl.Float64) - first_t_ns) / NANOSECONDS_PER_SECOND,
            ped_id=pl.col("ped_id").cast(pl.String),
            x_m=pl.col("x").cast(pl.Float64),
            y_m=pl.col("y").cast(pl.Float64),
            z_m=self._optional_track_column(tracks, "z"),
            yaw_rad=self._optional_track_column(tracks, "yaw"),
            vx_mps=self._optional_track_column(tracks, "vx"),
            vy_mps=self._optional_track_column(tracks, "vy"),
            speed_mps=self._optional_track_column(tracks, "speed_mps"),
        )
