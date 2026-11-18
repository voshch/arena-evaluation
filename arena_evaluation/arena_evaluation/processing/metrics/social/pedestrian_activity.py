"""Pedestrian lifetime activity: distance travelled, mean speed and completed patrol rounds."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, ScopeSlice
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

ROUND_TRIP_REACH_SHARE: Final[float] = 0.75
ROUND_TRIP_RETURN_SHARE: Final[float] = 0.25
ROUND_TRIP_MIN_DISPLACEMENT_M: Final[float] = 0.5

TOTAL_DISTANCE_M = define_metric(
    "total_distance_m",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="m",
    description="Planar distance the pedestrian travelled over the recorded track.",
    null_reason=True,
)
MEAN_SPEED_MPS = define_metric(
    "mean_speed_mps",
    pl.Float64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="m/s",
    description="Mean recorded speed of the pedestrian over the samples that carry one.",
    null_reason=True,
)
ROUND_TRIPS_COMPLETED = define_metric(
    "round_trips_completed",
    pl.Int64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="",
    description="Completed patrol rounds: an away reach of three quarters of the farthest displacement followed by a return within one quarter.",
    null_reason=True,
)


class PedestrianActivityCalculator(BaseMetricCalculator):
    """Pedestrian lifetime activity metrics: distance traveled, mean speed and patrol rounds."""

    NAME = "pedestrian_activity"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["pedestrian_arena_peds"]

    OUTPUTS = (TOTAL_DISTANCE_M, MEAN_SPEED_MPS, ROUND_TRIPS_COMPLETED)
    PRIMARY_OUTPUTS = ["total_distance_m", "mean_speed_mps"]

    def _calculate_for_target(
        self,
        episode: AlignedEpisodeBundle,
        target_id: str | None = None,
    ) -> dict[str, typing.Any]:
        ped_tracks = self.native_topics(episode).get("pedestrian_arena_peds")
        if isinstance(ped_tracks, pl.LazyFrame):
            ped_tracks = ped_tracks.collect()

        if ped_tracks is None or len(ped_tracks) == 0:
            return dict.fromkeys(self.output_keys())

        resolved_target = (
            target_id
            if target_id is not None
            else self.metrics_config.get("target_agent_id")
        )
        if resolved_target is not None:
            ped_tracks = ped_tracks.filter(pl.col("ped_id") == str(resolved_target))

        if len(ped_tracks) == 0:
            return dict.fromkeys(self.output_keys())

        tracks = ped_tracks.sort("t_ns").with_columns(
            derived_speed_mps=pl.col("speed_mps").fill_null(
                (pl.col("vx").pow(2) + pl.col("vy").pow(2)).sqrt()
            )
        )
        # A step exists only between two samples that both carry a position: dropping x and y
        # independently would pair one sample's dx with another sample's dy.
        steps = tracks.select(
            pl.col("x").diff().alias("dx"), pl.col("y").diff().alias("dy")
        ).drop_nulls()
        step_dist = np.hypot(steps["dx"].to_numpy(), steps["dy"].to_numpy())
        total_distance_m = float(np.sum(step_dist)) if len(step_dist) > 0 else 0.0

        speeds = tracks.get_column("derived_speed_mps").drop_nulls()
        mean_speed_mps = float(speeds.mean()) if len(speeds) > 0 else None

        spawn_x = float(tracks.get_column("x")[0])
        spawn_y = float(tracks.get_column("y")[0])
        x_arr = tracks.get_column("x").to_numpy().astype(np.float64)
        y_arr = tracks.get_column("y").to_numpy().astype(np.float64)
        displacements = np.hypot(x_arr - spawn_x, y_arr - spawn_y)
        max_disp = float(np.max(displacements))

        round_trips = 0
        if max_disp > ROUND_TRIP_MIN_DISPLACEMENT_M:
            reach_m = ROUND_TRIP_REACH_SHARE * max_disp
            return_m = ROUND_TRIP_RETURN_SHARE * max_disp
            disp_df = (
                pl.DataFrame({"disp": displacements})
                .filter((pl.col("disp") >= reach_m) | (pl.col("disp") <= return_m))
                .with_columns(is_away=pl.col("disp") >= reach_m)
                .with_columns(next_away=pl.col("is_away").shift(-1))
            )
            completed = disp_df.filter(
                pl.col("is_away") & ~pl.col("next_away").fill_null(True)
            )
            round_trips = len(completed)

        return {
            "total_distance_m": total_distance_m,
            "mean_speed_mps": mean_speed_mps,
            "round_trips_completed": round_trips,
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        return self._calculate_for_target(episode)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        target_id = (
            scope_slice.slice_id
            if scope_slice is not None and scope_slice.scope == MetricScope.PEDESTRIAN
            else None
        )
        del prior_results
        return self._calculate_for_target(episode, target_id=target_id)
