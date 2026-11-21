"""Hall proxemic-zone dwell, zone-entry events, closing time and personal-space intrusion integral."""

from __future__ import annotations

import typing
from typing import Final

import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import (
    TRACK_ALIGN_TOLERANCE_NS,
)
from arena_evaluation.processing.metrics.social.proxemics import (
    INTIMATE_CLEARANCE_M,
    PERSONAL_ZONE_EDGE_CLEARANCE_M,
)
from arena_evaluation.processing.pedestrian_frames import (
    NS_PER_S,
    FrameGeometry,
    frame_geometry,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.registry.series import SeriesProxemicsExtended
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

#: Edge-to-edge (surface) clearance, metres, at or beyond which a pair sits in Hall's public zone.
#: The intimate and personal boundaries it stacks on are edge clearances too: each zone bound is
#: the centre distance minus the robot and pedestrian radii.
SOCIAL_CLEARANCE_M: Final[float] = 3.6
TTI_LOOKAHEAD_S: Final[float] = 5.0
TTI_MIN_REL_SPEED_MPS: Final[float] = 0.05
#: Wall-clock gap, nanoseconds, that still merges two stays in one zone into a single entry event.
#: A time gap, never a distance: the zone bounds above are the distances.
ZONE_EVENT_MERGE_GAP_NS: Final[int] = 2 * 10**9

SERIES_SCHEMA: Final[pl.Schema] = SeriesProxemicsExtended.polars_schema()

ZONE_SCOPES: Final[tuple[MetricScope, ...]] = (
    MetricScope.EPISODE,
    MetricScope.ENCOUNTER,
)

#: Hall zones in boundary order, from the tightest to the public outside.
_ZONES: Final[tuple[str, ...]] = ("intimate", "personal", "social", "public")
_ZONE_ENTRY_ZONES: Final[tuple[str, ...]] = ("intimate", "personal", "social")

TIME_IN_INTIMATE_ZONE = define_metric(
    "time_in_intimate_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Seconds whose nearest pedestrian sat inside 0.45 m edge clearance.",
    null_reason=True,
)
TIME_IN_PERSONAL_ZONE = define_metric(
    "time_in_personal_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Seconds whose nearest pedestrian sat inside 1.20 m edge clearance.",
    null_reason=True,
)
TIME_IN_SOCIAL_ZONE = define_metric(
    "time_in_social_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Seconds whose nearest pedestrian sat inside 3.60 m edge clearance.",
    null_reason=True,
)
TIME_IN_PUBLIC_ZONE = define_metric(
    "time_in_public_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Seconds whose nearest pedestrian sat at or beyond 3.60 m edge clearance.",
    null_reason=True,
)
PSI_INTIMATE_EVENTS = define_metric(
    "psi_intimate_events",
    pl.Int64,
    scopes=ZONE_SCOPES,
    unit="",
    description="Contiguous stays of the nearest pedestrian inside 0.45 m edge clearance, nearby stays merged.",
    null_reason=True,
)
PSI_PERSONAL_EVENTS = define_metric(
    "psi_personal_events",
    pl.Int64,
    scopes=ZONE_SCOPES,
    unit="",
    description="Contiguous stays of the nearest pedestrian inside 1.20 m edge clearance, nearby stays merged.",
    null_reason=True,
)
PSI_SOCIAL_EVENTS = define_metric(
    "psi_social_events",
    pl.Int64,
    scopes=ZONE_SCOPES,
    unit="",
    description="Contiguous stays of the nearest pedestrian inside 3.60 m edge clearance, nearby stays merged.",
    null_reason=True,
)
MAX_SPEED_INTIMATE_ZONE = define_metric(
    "max_speed_intimate_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="m/s",
    description="Fastest robot speed while the nearest pedestrian sat inside 0.45 m edge clearance.",
    null_reason=True,
)
MAX_SPEED_PERSONAL_ZONE = define_metric(
    "max_speed_personal_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="m/s",
    description="Fastest robot speed while the nearest pedestrian sat inside 1.20 m edge clearance.",
    null_reason=True,
)
MAX_SPEED_SOCIAL_ZONE = define_metric(
    "max_speed_social_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="m/s",
    description="Fastest robot speed while the nearest pedestrian sat inside 3.60 m edge clearance.",
    null_reason=True,
)
MAX_SPEED_PUBLIC_ZONE = define_metric(
    "max_speed_public_zone",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="m/s",
    description="Fastest robot speed while the nearest pedestrian sat at or beyond 3.60 m edge clearance.",
    null_reason=True,
)
MOVEMENT_TOWARDS_PEDS_RATIO = define_metric(
    "movement_towards_peds_ratio",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="1",
    description="Share of resolved frames whose robot velocity pointed at the nearest pedestrian.",
    null_reason=True,
)
TTI_MIN = define_metric(
    "tti_min",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Shortest telemetry-only time to collision of a pedestrian closing on the robot.",
    null_reason=True,
)
TTI_MEAN = define_metric(
    "tti_mean",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="s",
    description="Mean telemetry-only time to collision of the pedestrians closing on the robot.",
    null_reason=True,
)
PERSONAL_SPACE_INTRUSION_INTEGRAL = define_metric(
    "personal_space_intrusion_integral",
    pl.Float64,
    scopes=ZONE_SCOPES,
    unit="m·s",
    description="Time integral of the penetration depth inside 1.20 m edge clearance.",
    null_reason=True,
)


def _zone_expression() -> pl.Expr:
    """Hall zone of an edge-to-edge clearance, null while the clearance is unknown."""
    clearance_m = pl.col("clearance_m")
    return pl.when(clearance_m.is_null()).then(pl.lit(None, dtype=pl.String)).when(clearance_m < INTIMATE_CLEARANCE_M).then(pl.lit("intimate")).when(clearance_m < PERSONAL_ZONE_EDGE_CLEARANCE_M).then(pl.lit("personal")).when(clearance_m < SOCIAL_CLEARANCE_M).then(pl.lit("social")).otherwise(pl.lit("public"))


def _zone_value_expression(zone: str, value: str, alias: str) -> pl.Expr:
    """``value`` on the frames whose closest pedestrian sat in ``zone``, null elsewhere; the caller reduces it."""
    return pl.when(pl.col("zone") == zone).then(pl.col(value)).otherwise(None).alias(alias)


def zone_entry_events(
    cleared: pl.DataFrame,
    zone: str,
    merge_gap_ns: int = ZONE_EVENT_MERGE_GAP_NS,
) -> int:
    """Number of contiguous zone stays, bridging stays that resume within ``merge_gap_ns`` of the previous one."""
    runs = (
        cleared.with_columns(in_zone=pl.col("zone") == zone)
        .select("t_ns", "in_zone")
        .with_columns(run_id=pl.col("in_zone").rle_id())
        .group_by("run_id")
        .agg(
            in_zone=pl.col("in_zone").first(),
            start_ns=pl.col("t_ns").min(),
            end_ns=pl.col("t_ns").max(),
        )
        .filter(pl.col("in_zone"))
        .sort("start_ns")
    )
    if runs.height == 0:
        return 0
    previous_end_ns = pl.col("end_ns").shift(1)
    starts_event = previous_end_ns.is_null() | ((pl.col("start_ns") - previous_end_ns) >= merge_gap_ns)
    return int(runs.select(starts_event.sum()).item())


class ProxemicsExtendedCalculator(BaseMetricCalculator):
    """Hall proxemic zone dwell, zone-entry events and closing-time metrics on the pedestrian frame axis."""

    NAME = "proxemics_extended"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics", "path_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_arena_peds"]

    OUTPUTS = (
        TIME_IN_INTIMATE_ZONE,
        TIME_IN_PERSONAL_ZONE,
        TIME_IN_SOCIAL_ZONE,
        TIME_IN_PUBLIC_ZONE,
        PSI_INTIMATE_EVENTS,
        PSI_PERSONAL_EVENTS,
        PSI_SOCIAL_EVENTS,
        MAX_SPEED_INTIMATE_ZONE,
        MAX_SPEED_PERSONAL_ZONE,
        MAX_SPEED_SOCIAL_ZONE,
        MAX_SPEED_PUBLIC_ZONE,
        MOVEMENT_TOWARDS_PEDS_RATIO,
        TTI_MIN,
        TTI_MEAN,
        PERSONAL_SPACE_INTRUSION_INTEGRAL,
    )
    PRIMARY_OUTPUTS = [
        "time_in_intimate_zone",
        "psi_intimate_events",
        "tti_min",
    ]
    DIRECTIONS = {
        "time_in_intimate_zone": Direction.LOWER,
        "time_in_personal_zone": Direction.LOWER,
        "psi_intimate_events": Direction.LOWER,
        "max_speed_intimate_zone": Direction.LOWER,
        "max_speed_personal_zone": Direction.LOWER,
        "tti_min": Direction.HIGHER,
        "personal_space_intrusion_integral": Direction.LOWER,
    }

    def _combined_radius_expression(self) -> pl.Expr:
        """Robot plus pedestrian radius removed from the centre distance to reach the edge clearance."""
        return pl.lit(float(self.robot_params.robot_radius), dtype=pl.Float64) + pl.col("radius_m")

    def _edge_clearance_expression(self) -> pl.Expr:
        """Surface clearance of one pedestrian row, from its own recorded radius."""
        return pl.col("distance_m") - self._combined_radius_expression()

    def _geometry(self, episode: AlignedEpisodeBundle) -> FrameGeometry:
        """Robot state on the pedestrian frame axis, with every pedestrian's relative geometry and own radius."""
        peds = self.native_ped_frame(episode)
        geometry = frame_geometry(
            peds,
            self.native_topics(episode).get("robot_tracks"),
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        )
        if peds is None or peds.height == 0:
            return geometry
        invalid = peds.filter(pl.col("radius_m").is_null() | pl.col("radius_m").is_nan() | (pl.col("radius_m") <= 0.0))
        if invalid.height > 0:
            raise MetricInputError("proxemics_extended needs each pedestrian's recorded radius_m; a pedestrian has none or a non-positive one")
        radii = peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            radius_m=pl.col("radius_m").cast(pl.Float64),
        ).unique(subset=["t_ns", "ped_id"], keep="first")
        pedestrians = geometry.pedestrians.join(radii, on=["t_ns", "ped_id"], how="left")
        return FrameGeometry(frames=geometry.frames, pedestrians=pedestrians)

    def _nearest_frame_pedestrians(self, geometry: FrameGeometry) -> pl.DataFrame:
        """The pedestrian with the smallest edge clearance of every resolved frame, with the frame's robot state."""
        if geometry.pedestrians.height == 0:
            return pl.DataFrame()
        ranked = geometry.pedestrians.filter(pl.col("distance_m").is_not_null()).with_columns(clearance_m=self._edge_clearance_expression())
        ranked = ranked.with_columns(clearance_rank=pl.col("clearance_m").rank("ordinal").over("frame_ix"))
        nearest = ranked.filter(pl.col("clearance_rank") == 1).drop("clearance_rank")
        return nearest.join(
            geometry.frames.select("frame_ix", "dt_s", "robot_vx_mps", "robot_vy_mps", "robot_speed_mps"),
            on="frame_ix",
            how="inner",
        )

    def _cleared_frames(self, geometry: FrameGeometry) -> pl.DataFrame:
        """The closest pedestrian of every resolved frame, its edge clearance, Hall zone and integration step."""
        if geometry.pedestrians.height == 0:
            return pl.DataFrame()
        return self._nearest_frame_pedestrians(geometry).sort("t_ns").with_columns(zone=_zone_expression())

    def _closing_times(self, geometry: FrameGeometry) -> pl.DataFrame:
        """Telemetry-only time-to-collision of every pedestrian closing on the robot inside the lookahead."""
        clearance_m = self._edge_clearance_expression()
        closing_m2_s2 = pl.col("dx_m") * pl.col("rel_vx_mps") + pl.col("dy_m") * pl.col("rel_vy_mps")
        return (
            geometry.pedestrians.with_columns(clearance_m=clearance_m)
            .filter((pl.col("rel_speed_mps") > TTI_MIN_REL_SPEED_MPS) & (closing_m2_s2 < 0.0) & (pl.col("clearance_m") > 0.0))
            .with_columns(tti_s=pl.col("clearance_m") / pl.col("rel_speed_mps"))
            .filter((pl.col("tti_s") > 0.0) & (pl.col("tti_s") < TTI_LOOKAHEAD_S))
            .select("tti_s")
        )

    def _episode_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Every proxemic output over the given episode view, all null without a resolved robot-pedestrian frame."""
        geometry = self._geometry(episode)
        cleared = self._cleared_frames(geometry)
        if cleared.height == 0:
            return dict.fromkeys(self.output_keys())

        summary = cleared.select(
            *(_zone_value_expression(zone, "dt_s", f"time_in_{zone}_zone_s").sum() for zone in _ZONES),
            *(_zone_value_expression(zone, "robot_speed_mps", f"max_speed_{zone}_zone_mps").max() for zone in _ZONES),
            personal_space_intrusion_integral_m_s=(pl.when(pl.col("clearance_m") < PERSONAL_ZONE_EDGE_CLEARANCE_M).then((PERSONAL_ZONE_EDGE_CLEARANCE_M - pl.col("clearance_m").clip(lower_bound=0.0)) * pl.col("dt_s")).otherwise(0.0).sum()),
            approaching_frames=((pl.col("robot_vx_mps") * pl.col("dx_m") + pl.col("robot_vy_mps") * pl.col("dy_m")) > 0.0).sum(),
            sampled_frames=pl.len(),
        ).row(0, named=True)

        closing = self._closing_times(geometry)
        tti_min = closing.select(pl.col("tti_s").min()).item()
        tti_mean = closing.select(pl.col("tti_s").mean()).item()
        sampled_frames = int(summary["sampled_frames"])
        approaching_frames = int(summary["approaching_frames"])

        return {
            **{f"time_in_{zone}_zone": float(summary[f"time_in_{zone}_zone_s"]) for zone in _ZONES},
            **{f"psi_{zone}_events": zone_entry_events(cleared, zone) for zone in _ZONE_ENTRY_ZONES},
            **{f"max_speed_{zone}_zone": summary[f"max_speed_{zone}_zone_mps"] for zone in _ZONES},
            "movement_towards_peds_ratio": approaching_frames / sampled_frames if sampled_frames > 0 else None,
            "tti_min": tti_min,
            "tti_mean": tti_mean,
            "personal_space_intrusion_integral": float(summary["personal_space_intrusion_integral_m_s"]),
        }

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Compute the episode-scope proxemics set over the bundle's own pedestrian frames."""
        del prior_results
        return self._episode_metrics(episode)

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        """EPISODE keeps the aligned population; an ENCOUNTER slice measures its own window telemetry."""
        del prior_results
        if scope_slice is None:
            return self._episode_metrics(episode)
        match scope_slice.scope:
            case MetricScope.EPISODE:
                return self._episode_metrics(episode)
            case MetricScope.ENCOUNTER:
                return self._episode_metrics(scope_slice.window_bundle)
            case _:
                return dict.fromkeys(self.output_keys())

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-frame clearance series: one row per pedestrian frame, null clearance where unresolved."""
        geometry = self._geometry(episode)
        if geometry.frames.height == 0 or geometry.pedestrians.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(geometry.frames.select(pl.col("t_ns").min()).item())
        clearances = self._nearest_frame_pedestrians(geometry).select("frame_ix", "clearance_m")
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return (
            geometry.frames.join(clearances, on="frame_ix", how="left")
            .sort("t_ns")
            .select(
                run_id=pl.lit(run_id, dtype=pl.String),
                episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
                robot_id=pl.lit(resolved_robot, dtype=pl.String),
                t_ns=pl.col("t_ns"),
                time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / NS_PER_S,
                min_ped_clearance_m=pl.col("clearance_m"),
            )
        )
