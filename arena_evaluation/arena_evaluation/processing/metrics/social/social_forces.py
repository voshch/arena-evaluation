"""Helbing-Molnar social force and personal-space index on the pedestrian frame axis."""

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
from arena_evaluation.processing.pedestrian_frames import (
    NS_PER_S,
    FrameGeometry,
    frame_geometry,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.registry.series import SeriesSocialForces
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SFM_INTERACTION_STRENGTH_N: Final[float] = 2.1
SFM_INTERACTION_RANGE_M: Final[float] = 0.3
SFM_CUTOFF_M: Final[float] = 5.0
SFM_MAX_FORCE_N: Final[float] = 100.0
ESFM_ANISOTROPY: Final[float] = 0.5
CI_SIGMA_FRONT_M: Final[float] = 1.0
CI_SIGMA_SIDE_M: Final[float] = 0.5
CI_SIGMA_X_M: Final[float] = 0.28
CI_SIGMA_Y_M: Final[float] = 0.28

SERIES_SCHEMA: Final[pl.Schema] = SeriesSocialForces.polars_schema()

FORCE_SCOPES: Final[tuple[MetricScope, ...]] = (
    MetricScope.EPISODE,
    MetricScope.ENCOUNTER,
)

SFM_CUMULATIVE_FORCE = define_metric(
    "sfm_cumulative_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N·s",
    description="Time integral of the Helbing social force from the robot onto the pedestrians.",
    null_reason=True,
)
SFM_PEAK_FORCE = define_metric(
    "sfm_peak_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Peak Helbing social force from the robot onto the pedestrians.",
    null_reason=True,
)
SFM_MEAN_FORCE = define_metric(
    "sfm_mean_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Mean Helbing social force from the robot onto the pedestrians over the resolved frames.",
    null_reason=True,
)
ESFM_CUMULATIVE_FORCE = define_metric(
    "esfm_cumulative_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N·s",
    description="Time integral of the extended-SFM force with robot-heading anisotropy.",
    null_reason=True,
)
ESFM_PEAK_FORCE = define_metric(
    "esfm_peak_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Peak extended-SFM force with robot-heading anisotropy.",
    null_reason=True,
)
ESFM_MEAN_FORCE = define_metric(
    "esfm_mean_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Mean extended-SFM force with robot-heading anisotropy.",
    null_reason=True,
)
ESFM_PED_CUMULATIVE_FORCE = define_metric(
    "esfm_ped_cumulative_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N·s",
    description="Time integral of the extended-SFM force with pedestrian-heading anisotropy.",
    null_reason=True,
)
ESFM_PED_PEAK_FORCE = define_metric(
    "esfm_ped_peak_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Peak extended-SFM force with pedestrian-heading anisotropy.",
    null_reason=True,
)
ESFM_PED_MEAN_FORCE = define_metric(
    "esfm_ped_mean_force",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="N",
    description="Mean extended-SFM force with pedestrian-heading anisotropy.",
    null_reason=True,
)
CI_MAX = define_metric(
    "ci_max",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="1",
    description="Peak Gaussian personal-space index of the pedestrians.",
    null_reason=True,
)
CI_MEAN = define_metric(
    "ci_mean",
    pl.Float64,
    scopes=FORCE_SCOPES,
    unit="1",
    description="Mean Gaussian personal-space index over the resolved frames.",
    null_reason=True,
)


def _anisotropy(angle_rad: pl.Expr) -> pl.Expr:
    """Extended-SFM anisotropy weight in [0, 1] of a heading angle difference."""
    return (ESFM_ANISOTROPY + (1.0 - ESFM_ANISOTROPY) * (1.0 + angle_rad.cos()) / 2.0).clip(0.0, 1.0)


def _force_expression(robot_radius_m: float) -> pl.Expr:
    """Helbing repulsive magnitude of one pedestrian, clamped, zero beyond the cutoff, null while unresolved.

    The contact distance is this robot's radius plus the pedestrian's own recorded ``radius_m``;
    a pedestrian without one raises in :meth:`SocialForcesCalculator._geometry` instead of taking
    a fleet-wide body size.
    """
    distance_m = pl.col("distance_m")
    combined_radius_m = robot_radius_m + pl.col("radius_m")
    return pl.when(distance_m.is_null()).then(pl.lit(None, dtype=pl.Float64)).when(distance_m <= SFM_CUTOFF_M).then((SFM_INTERACTION_STRENGTH_N * ((combined_radius_m - distance_m) / SFM_INTERACTION_RANGE_M).exp()).clip(upper_bound=SFM_MAX_FORCE_N)).otherwise(0.0)


def _personal_space_index_expression() -> pl.Expr:
    """Gaussian personal-space index of one pedestrian, elliptical along its heading when that heading is recorded."""
    heading_cos = pl.col("ped_yaw_rad").cos()
    heading_sin = pl.col("ped_yaw_rad").sin()
    d_front_m = pl.col("dx_m") * heading_cos + pl.col("dy_m") * heading_sin
    d_side_m = -pl.col("dx_m") * heading_sin + pl.col("dy_m") * heading_cos
    positional_exponent = -(pl.col("dx_m").pow(2) / (2.0 * CI_SIGMA_X_M**2) + pl.col("dy_m").pow(2) / (2.0 * CI_SIGMA_Y_M**2))
    heading_exponent = -(d_front_m.pow(2) / (2.0 * CI_SIGMA_FRONT_M**2) + d_side_m.pow(2) / (2.0 * CI_SIGMA_SIDE_M**2))
    return pl.when(pl.col("ped_yaw_rad").is_null()).then(positional_exponent).otherwise(heading_exponent).exp()


def _frame_value(column: str, resolved: str) -> pl.Expr:
    """A frame's pedestrian-derived value: zero without pedestrians, null while the robot state is unresolved."""
    return pl.when(~pl.col("has_pedestrians")).then(pl.lit(0.0)).when(~pl.col(resolved)).then(pl.lit(None, dtype=pl.Float64)).otherwise(pl.col(column))


def _observed(value: str) -> pl.Expr:
    """Per-frame value over the frames that observed it, null frames skipped."""
    return pl.col(value).drop_nulls()


class SocialForcesCalculator(BaseMetricCalculator):
    """Helbing-Molnar social force, extended-SFM anisotropy and personal-space index metrics."""

    NAME = "social_forces"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_arena_peds"]

    OUTPUTS = (
        SFM_CUMULATIVE_FORCE,
        SFM_PEAK_FORCE,
        SFM_MEAN_FORCE,
        ESFM_CUMULATIVE_FORCE,
        ESFM_PEAK_FORCE,
        ESFM_MEAN_FORCE,
        ESFM_PED_CUMULATIVE_FORCE,
        ESFM_PED_PEAK_FORCE,
        ESFM_PED_MEAN_FORCE,
        CI_MAX,
        CI_MEAN,
    )
    PRIMARY_OUTPUTS = ["sfm_mean_force", "ci_mean", "esfm_ped_peak_force"]
    DIRECTIONS = {
        "sfm_mean_force": Direction.LOWER,
        "ci_mean": Direction.LOWER,
    }

    def _geometry(self, episode: AlignedEpisodeBundle) -> FrameGeometry:
        """Robot state on the pedestrian frame axis, with every pedestrian's relative geometry and own radius."""
        peds = self.native_ped_frame(episode)
        geometry = frame_geometry(
            peds,
            self.native_topics(episode).get("robot_tracks"),
            tolerance_ns=TRACK_ALIGN_TOLERANCE_NS,
        )
        if peds is None:
            return geometry
        radii = peds.select(
            t_ns=pl.col("t_ns").cast(pl.Int64),
            ped_id=pl.col("ped_id").cast(pl.String),
            radius_m=pl.col("radius_m").cast(pl.Float64),
        ).unique(subset=["t_ns", "ped_id"], keep="first")
        pedestrians = geometry.pedestrians.join(radii, on=["t_ns", "ped_id"], how="left")
        resolved = pedestrians.filter(pl.col("distance_m").is_not_null())
        if resolved.filter(pl.col("radius_m").is_null() | pl.col("radius_m").is_nan()).height > 0:
            raise MetricInputError("social_forces needs each pedestrian's recorded radius_m; the pedestrian track has a body without one")
        return FrameGeometry(frames=geometry.frames, pedestrians=pedestrians)

    def _frame_forces(self, geometry: FrameGeometry) -> pl.DataFrame:
        """Per-frame social force totals and personal-space index, one row per pedestrian frame."""
        force_n = _force_expression(float(self.robot_params.robot_radius))
        robot_anisotropy = _anisotropy(pl.col("robot_yaw_rad") - pl.arctan2(pl.col("dy_m"), pl.col("dx_m")))
        pedestrian_anisotropy = pl.when(pl.col("ped_yaw_rad").is_null()).then(pl.lit(1.0)).otherwise(_anisotropy(pl.col("ped_yaw_rad") - pl.arctan2(-pl.col("dy_m"), -pl.col("dx_m"))))
        contributions = geometry.pedestrians.with_columns(
            force_n=force_n,
            esfm_force_n=force_n * robot_anisotropy,
            esfm_ped_force_n=force_n * pedestrian_anisotropy,
            ci=_personal_space_index_expression(),
        )
        totals = contributions.group_by("frame_ix").agg(
            has_pedestrians=(pl.len() > 0),
            resolved=pl.col("distance_m").is_not_null().any(),
            esfm_resolved=pl.col("esfm_force_n").is_not_null().any(),
            sfm_force_n=pl.col("force_n").drop_nulls().sum(),
            esfm_force_n=pl.col("esfm_force_n").drop_nulls().sum(),
            esfm_ped_force_n=pl.col("esfm_ped_force_n").drop_nulls().sum(),
            ci=pl.col("ci").drop_nulls().max(),
        )
        return (
            geometry.frames.select("frame_ix", "t_ns", "dt_s")
            .join(totals, on="frame_ix", how="left")
            .with_columns(
                has_pedestrians=pl.col("has_pedestrians").fill_null(False),
                resolved=pl.col("resolved").fill_null(False),
                esfm_resolved=pl.col("esfm_resolved").fill_null(False),
                sfm_force_n=_frame_value("sfm_force_n", "resolved"),
                esfm_force_n=_frame_value("esfm_force_n", "esfm_resolved"),
                esfm_ped_force_n=_frame_value("esfm_ped_force_n", "resolved"),
                ci=_frame_value("ci", "resolved"),
            )
            .sort("t_ns")
        )

    def _aggregate(self, frame_forces: pl.DataFrame) -> dict[str, float | None]:
        """Force and index summaries over the frames that observed a resolved robot-pedestrian pair."""
        keys: Final[tuple[str, ...]] = tuple(self.output_keys())
        if frame_forces.height == 0 or frame_forces.select(pl.col("sfm_force_n").is_not_null().any()).item() is not True:
            return dict.fromkeys(keys)
        row = frame_forces.select(
            # Row by row: a null force adds nothing and keeps its own frame's duration.
            sfm_cumulative_force=(pl.col("sfm_force_n") * pl.col("dt_s")).sum(),
            esfm_cumulative_force=(pl.col("esfm_force_n") * pl.col("dt_s")).sum(),
            esfm_ped_cumulative_force=(pl.col("esfm_ped_force_n") * pl.col("dt_s")).sum(),
            sfm_peak_force=_observed("sfm_force_n").max(),
            sfm_mean_force=_observed("sfm_force_n").mean(),
            esfm_peak_force=_observed("esfm_force_n").max(),
            esfm_mean_force=_observed("esfm_force_n").mean(),
            esfm_ped_peak_force=_observed("esfm_ped_force_n").max(),
            esfm_ped_mean_force=_observed("esfm_ped_force_n").mean(),
            ci_max=pl.col("ci").drop_nulls().max(),
            ci_mean=pl.col("ci").drop_nulls().mean(),
        ).row(0, named=True)
        return {key: None if row[key] is None else float(row[key]) for key in keys}

    def _episode_metrics(self, episode: AlignedEpisodeBundle) -> dict[str, float | None]:
        """Every social-force output over the given episode view."""
        geometry = self._geometry(episode)
        if not geometry.observed:
            return dict.fromkeys(self.output_keys())
        return self._aggregate(self._frame_forces(geometry))

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Compute the episode-scope social forces over the bundle's own pedestrian frames."""
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
        scope = MetricScope.EPISODE if scope_slice is None else scope_slice.scope
        match scope:
            case MetricScope.EPISODE:
                return self._episode_metrics(episode)
            case MetricScope.ENCOUNTER:
                return self._episode_metrics(scope_slice.window_bundle)
            case _:
                return dict.fromkeys(self.output_keys())

    def series_table(
        self,
        episode: AlignedEpisodeBundle,
        *,
        run_id: str = "",
        robot_id: str = "",
    ) -> pl.DataFrame:
        """1NF per-frame force and index series: one row per pedestrian frame, null where the state is unresolved."""
        geometry = self._geometry(episode)
        if geometry.frames.height == 0:
            return pl.DataFrame(schema=SERIES_SCHEMA)
        first_t_ns = int(geometry.frames.select(pl.col("t_ns").min()).item())
        resolved_robot = robot_id or (episode.robot_name or "robot")
        return self._frame_forces(geometry).select(
            run_id=pl.lit(run_id, dtype=pl.String),
            episode_id=pl.lit(int(episode.episode_id), dtype=pl.Int64),
            robot_id=pl.lit(resolved_robot, dtype=pl.String),
            t_ns=pl.col("t_ns"),
            time_s=(pl.col("t_ns") - first_t_ns).cast(pl.Float64) / NS_PER_S,
            sfm_force_n=pl.col("sfm_force_n"),
            ci=pl.col("ci"),
        )
