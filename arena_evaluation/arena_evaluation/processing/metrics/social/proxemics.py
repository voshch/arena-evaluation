"""Personal-space occupancy of the robot, and per-pedestrian zone-intrusion counts."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.storage.registry.series import SeriesProxemics
from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    native_robot_frame,
)
from arena_evaluation.processing.pedestrian_frames import (
    NS_PER_S,
    distances_to_pedestrians,
)
from arena_evaluation.processing.windows import (
    DEFAULT_DETECTOR_PARAMS,
    pedestrian_pairs_lf,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

INTIMATE_CLEARANCE_M: Final[float] = 0.45
# Three personal-space boundaries with different meanings, each named for what it measures: this one is the
# edge clearance of the personal zone (1.20 m); PROBST_PERSONAL_SPACE_RADIUS_M (probst_responsibility.py) is the
# body radius of Probst's pair (1.0 m); the edge-case intrusion boundary is PERSONAL_SPACE (0.5 m) in
# edge_case/criticality.py.
PERSONAL_ZONE_EDGE_CLEARANCE_M: Final[float] = 1.20

PERSONAL_SPACE_SERIES_SCHEMA: Final[pl.Schema] = SeriesProxemics.polars_schema()

NUM_PEDESTRIANS = define_metric(
    "num_pedestrians",
    pl.Int64,
    scopes=(MetricScope.EPISODE, MetricScope.ENCOUNTER),
    unit="",
    description="Pedestrians that took part in the episode or encounter window.",
    null_reason=True,
)
TOTAL_TIME_IN_PERSONAL_SPACE = define_metric(
    "total_time_in_personal_space",
    pl.Float64,
    scopes=(MetricScope.EPISODE, MetricScope.ENCOUNTER),
    unit="s",
    description="Seconds whose robot sample had at least one pedestrian inside 1.20 m edge clearance.",
    null_reason=True,
)
AVG_VELOCITY_IN_PERSONAL_SPACE = define_metric(
    "avg_velocity_in_personal_space",
    pl.Float64,
    scopes=(MetricScope.EPISODE, MetricScope.ENCOUNTER),
    unit="m/s",
    description="Mean robot speed over the samples with a pedestrian inside 1.20 m edge clearance, from the ground-truth position difference in both scopes.",
    null_reason=True,
)
TOTAL_INTRUSIONS_INTIMATE = define_metric(
    "total_intrusions_intimate",
    pl.Int64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="",
    description="Entries of the pedestrian into the robot's intimate zone, 0.45 m edge clearance.",
    null_reason=True,
)
TOTAL_INTRUSIONS_PERSONAL = define_metric(
    "total_intrusions_personal",
    pl.Int64,
    scopes=(MetricScope.PEDESTRIAN,),
    unit="",
    description="Entries of the pedestrian into the robot's personal zone, 1.20 m edge clearance.",
    null_reason=True,
)


def _pedestrian_radii(peds: pl.DataFrame) -> pl.DataFrame:
    """Each pedestrian's recorded body radius; a missing or non-positive one is refused."""
    radii = peds.select(
        ped_id=pl.col("ped_id").cast(pl.String),
        radius_m=pl.col("radius_m").cast(pl.Float64),
    ).unique(subset=["ped_id"], keep="first")
    invalid = radii.filter(
        pl.col("radius_m").is_null()
        | pl.col("radius_m").is_nan()
        | (pl.col("radius_m") <= 0.0)
    )
    if invalid.height > 0:
        raise MetricInputError(
            "proxemics needs each pedestrian's recorded radius_m; a pedestrian has none or a non-positive one"
        )
    return radii


def personal_space_counts(
    robot_frame: pl.DataFrame,
    peds: pl.DataFrame,
    clearance_m: float,
    robot_radius_m: float,
) -> pl.DataFrame:
    """One row per robot ground-truth sample with the number of pedestrians inside ``clearance_m`` edge clearance."""
    t_ns = robot_frame.get_column("t_ns").to_numpy().astype(np.int64)
    pos_x = robot_frame.get_column("x").to_numpy().astype(np.float64)
    pos_y = robot_frame.get_column("y").to_numpy().astype(np.float64)
    radii = _pedestrian_radii(peds)
    within = (
        distances_to_pedestrians(t_ns, pos_x, pos_y, peds)
        .join(radii, on="ped_id", how="left")
        .with_columns(
            edge_clearance_m=pl.col("distance")
            - (pl.lit(float(robot_radius_m), dtype=pl.Float64) + pl.col("radius_m"))
        )
        .filter(pl.col("edge_clearance_m") < clearance_m)
        .group_by("row")
        .len()
    )
    counts = np.zeros(len(t_ns), dtype=np.int64)
    if within.height > 0:
        counts[within["row"].to_numpy()] = within["len"].to_numpy()
    return pl.DataFrame(
        {"t_ns": t_ns, "count": counts},
        schema={"t_ns": pl.Int64, "count": pl.Int64},
    )


def _masked_speed_mean(speed_mps: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean of the measured speeds under ``mask``, None when no masked sample carries a speed."""
    measured = speed_mps[mask & np.isfinite(speed_mps)]
    return float(np.mean(measured)) if measured.size > 0 else None


class ProxemicsCalculator(BaseMetricCalculator):
    """Calculates proxemics and personal space intrusion metrics."""

    NAME = "proxemics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_arena_peds"]

    OUTPUTS = (
        NUM_PEDESTRIANS,
        TOTAL_TIME_IN_PERSONAL_SPACE,
        AVG_VELOCITY_IN_PERSONAL_SPACE,
    )
    PRIMARY_OUTPUTS = ["total_time_in_personal_space"]
    DIRECTIONS = {
        "total_time_in_personal_space": Direction.LOWER,
    }

    def _episode_inputs(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[pl.DataFrame, pl.DataFrame] | None:
        """The episode's robot ground-truth frame and flat pedestrian rows, None when either is unusable."""
        robot_frame = self.resolve_robot_frame(episode)
        peds_df = self.native_ped_frame(episode)
        if (
            robot_frame is None
            or robot_frame.height == 0
            or peds_df is None
            or peds_df.height == 0
        ):
            return None
        return robot_frame, peds_df

    def _occupancy_counts(
        self, robot_frame: pl.DataFrame, peds_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Personal-space occupancy counts against the calculator's own radius parameters."""
        return personal_space_counts(
            robot_frame,
            peds_df,
            PERSONAL_ZONE_EDGE_CLEARANCE_M,
            float(self.robot_params.robot_radius),
        )

    def _personal_space_metrics(
        self,
        robot_frame: pl.DataFrame,
        peds_df: pl.DataFrame,
        num_pedestrians: int | None,
    ) -> dict[str, typing.Any]:
        """Dwell, mean in-space speed and pedestrian count over the given robot frame and pedestrian set."""
        counts = self._occupancy_counts(robot_frame, peds_df)
        t_ns = counts.get_column("t_ns").to_numpy()
        occupancy = counts.get_column("count").to_numpy()
        step_s = np.append(np.diff(t_ns) / NS_PER_S, 0.0)
        inside = occupancy > 0
        speeds = self.speed_from_pose(
            robot_frame.get_column("x").to_numpy().astype(np.float64),
            robot_frame.get_column("y").to_numpy().astype(np.float64),
            t_ns,
        )
        return {
            "num_pedestrians": num_pedestrians,
            "total_time_in_personal_space": float(np.sum(step_s[inside])),
            "avg_velocity_in_personal_space": _masked_speed_mean(speeds, inside),
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Dwell of the robot inside its personal space, and the mean velocity while inside."""
        del prior_results
        inputs = self._episode_inputs(episode)
        if inputs is None:
            return dict.fromkeys(self.output_keys())
        robot_frame, peds_df = inputs
        return self._personal_space_metrics(
            robot_frame, peds_df, episode.num_pedestrians
        )

    def _window_metrics(self, window: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Personal-space dwell over the window's own robot samples and pedestrian set."""
        robot = native_robot_frame(window)
        peds = self.native_ped_frame(window)
        if robot is None or robot.height == 0 or peds is None or peds.height == 0:
            return dict.fromkeys(self.output_keys())
        return self._personal_space_metrics(
            robot, peds, int(peds.get_column("ped_id").n_unique())
        )

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE keeps the aligned population; an ENCOUNTER slice measures its own window telemetry."""
        scope = MetricScope.EPISODE if scope_slice is None else scope_slice.scope
        match scope:
            case MetricScope.EPISODE:
                return self.calculate(episode, prior_results)
            case MetricScope.ENCOUNTER:
                return self._window_metrics(scope_slice.window_bundle)
            case _:
                return dict.fromkeys(self.output_keys())

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-sample personal-space occupancy, one row per robot sample."""
        inputs = self._episode_inputs(episode)
        if inputs is None:
            return pl.DataFrame(schema=PERSONAL_SPACE_SERIES_SCHEMA)
        robot_frame, peds_df = inputs
        counts = self._occupancy_counts(robot_frame, peds_df)
        first_t_ns = int(counts.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return counts.select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / NS_PER_S,
            peds_in_personal_space=pl.col("count"),
        )


class PedestrianIntrusionCalculator(BaseMetricCalculator):
    """Pedestrian-level zone intrusion counts on edge clearance (intimate <0.45 m, personal <1.20 m)."""

    NAME = "pedestrian_intrusions"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_tracks", "pedestrian_arena_peds"]

    OUTPUTS = (TOTAL_INTRUSIONS_INTIMATE, TOTAL_INTRUSIONS_PERSONAL)
    PRIMARY_OUTPUTS = ["total_intrusions_intimate", "total_intrusions_personal"]
    DIRECTIONS = {
        "total_intrusions_intimate": Direction.LOWER,
        "total_intrusions_personal": Direction.LOWER,
    }

    def _calculate_for_target(
        self,
        episode: AlignedEpisodeBundle,
        target_id: str | None = None,
    ) -> dict[str, typing.Any]:
        topics = self.native_topics(episode)
        robot_tracks = topics.get("robot_tracks")
        ped_tracks = topics.get("pedestrian_arena_peds")

        if isinstance(robot_tracks, pl.LazyFrame):
            robot_tracks = robot_tracks.collect()
        if isinstance(ped_tracks, pl.LazyFrame):
            ped_tracks = ped_tracks.collect()

        if (
            robot_tracks is None
            or len(robot_tracks) == 0
            or ped_tracks is None
            or len(ped_tracks) == 0
        ):
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

        robot_radius_m = float(self.robot_params.robot_radius)
        pairs = (
            pedestrian_pairs_lf(
                robot_tracks, ped_tracks, robot_radius_m, DEFAULT_DETECTOR_PARAMS
            )
            .collect()
            .sort(["actor_id", "t_ns"])
        )
        if len(pairs) == 0:
            return {"total_intrusions_intimate": 0, "total_intrusions_personal": 0}

        entries = pairs.with_columns(
            intimate=pl.col("edge_clearance_m") < INTIMATE_CLEARANCE_M,
            personal=pl.col("edge_clearance_m") < PERSONAL_ZONE_EDGE_CLEARANCE_M,
        ).with_columns(
            intimate_entry=pl.col("intimate")
            & ~pl.col("intimate").shift(1, fill_value=False).over("actor_id"),
            personal_entry=pl.col("personal")
            & ~pl.col("personal").shift(1, fill_value=False).over("actor_id"),
        )
        return {
            "total_intrusions_intimate": int(
                entries.get_column("intimate_entry").sum()
            ),
            "total_intrusions_personal": int(
                entries.get_column("personal_entry").sum()
            ),
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
