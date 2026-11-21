"""Holistic cross-domain metrics: acoustic exposure in personal space and cost of transport."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    BaseMetricCalculator,
    Direction,
)
from arena_evaluation.storage.registry import MetricScope, NullReason, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PERSONAL_SPACE_CLEARANCE_M: Final[float] = 1.2
APA_MIN_DISTANCE_M: Final[float] = 0.3
COLLISION_PENALTY: Final[float] = 2.0
INTRUSION_WEIGHT: Final[float] = 0.5
COT_DECIMALS: Final[int] = 6
# The registry refuses an empty unit for a float metric and defines no dimensionless marker;
# "1" is the UCUM code for a dimensionless ratio.
DIMENSIONLESS_UNIT: Final[str] = "1"


def _aligned_values(t_ns: np.ndarray, frame: pl.DataFrame, column: str) -> np.ndarray:
    """`frame[column]` reindexed onto the episode's sample times, NaN where the sample is absent."""
    if frame.height == 0:
        return np.full(len(t_ns), np.nan, dtype=np.float64)
    values = frame.select(
        pl.col("t_ns").cast(pl.Int64), pl.col(column).cast(pl.Float64)
    ).drop_nulls("t_ns")
    values = values.unique(subset=["t_ns"], keep="last")
    joined = pl.DataFrame({"t_ns": t_ns.astype(np.int64)}).join(
        values, on="t_ns", how="left"
    )
    return joined.get_column(column).to_numpy().astype(np.float64, copy=False)


AEPS_LINEAR_S = define_metric(
    "aeps_linear_s",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Acoustic exposure inside personal space: integral of the linear sound power over the frames whose nearest pedestrian stayed inside 1.2 m. Bounded below by the measured levels; no nominal 45 dBA floor is substituted, so an absent acoustic-exposure series gives NULL.",
    null_reason=True,
)
E_COT = define_metric(
    "e_cot",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit=DIMENSIONLESS_UNIT,
    description="Effective cost of transport: specific cost of transport times (1 + 2 * collisions + 0.5 * personal-space intrusion integral / episode duration). NULL without a positive specific cost of transport.",
    null_reason=True,
)
ACOUSTIC_PROXEMIC_ANNOYANCE = define_metric(
    "acoustic_proxemic_annoyance",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s/m2",
    description="Acoustic proxemic annoyance: integral of linear sound power divided by squared nearest-pedestrian distance clamped at 0.3 m.",
    null_reason=True,
)
RELATIVE_SOCIAL_COMPLIANCE_COST = define_metric(
    "relative_social_compliance_cost",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit=DIMENSIONLESS_UNIT,
    description="Relative social compliance cost: fractional total-energy overhead relative to the planner's unobstructed reference run, computed in Layer 3.",
    null_reason=True,
)


class HolisticMetricsCalculator(BaseMetricCalculator):
    """Acoustic exposure inside personal space, acoustic proxemic annoyance, RSCC, and effective cost of transport."""

    NAME = "holistic_metrics"
    CATEGORY = "holistic"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = [
        "energy",
        "collision_metrics",
        "proxemics_extended",
        "time_metrics",
        "acoustic_exposure",
    ]
    SERIES_DEPENDS_ON = ("acoustic_exposure", "proxemics_extended")
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom"), "pedestrian_arena_peds"]

    OUTPUTS = (
        AEPS_LINEAR_S,
        E_COT,
        ACOUSTIC_PROXEMIC_ANNOYANCE,
        RELATIVE_SOCIAL_COMPLIANCE_COST,
    )
    DIRECTIONS = {
        "aeps_linear_s": Direction.LOWER,
        "e_cot": Direction.LOWER,
        "acoustic_proxemic_annoyance": Direction.LOWER,
        "relative_social_compliance_cost": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["e_cot"]

    def _level_series(
        self, episode: AlignedEpisodeBundle, t_ns: np.ndarray
    ) -> np.ndarray | None:
        """Per-frame dBA: the loudest pedestrian exposure at each sample time from the published series."""
        frame = self.series_frame(episode, "acoustic_exposure")
        if frame is None:
            return None
        per_frame = (
            frame.group_by("t_ns")
            .agg(pl.col("exposure_dba").max().alias("level_dba"))
            .sort("t_ns")
        )
        levels = _aligned_values(t_ns, per_frame, "level_dba")
        return levels if np.isfinite(levels).any() else None

    def _clearance_series(
        self, episode: AlignedEpisodeBundle, t_ns: np.ndarray
    ) -> np.ndarray | None:
        """Per-frame nearest-pedestrian clearance from the published proxemics-extended series."""
        frame = self.series_frame(episode, "proxemics_extended")
        if frame is None:
            return None
        clearances = _aligned_values(t_ns, frame, "min_ped_clearance_m")
        return clearances if np.isfinite(clearances).any() else None

    @staticmethod
    def _t_ns_axis(episode: AlignedEpisodeBundle) -> np.ndarray | None:
        """Aligned sample timestamps of the episode view, none without aligned rows."""
        if episode.data is None or episode.data.height == 0:
            return None
        return episode.data.get_column("t_ns").to_numpy().astype(np.int64)

    def _acoustic_proxemic_integrals(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[float | None, float | None]:
        """AEPS and APA from one aligned clearance + acoustic-exposure pass."""
        t_ns = self._t_ns_axis(episode)
        if t_ns is None or len(t_ns) == 0:
            return None, None
        clearance = self._clearance_series(episode, t_ns)
        levels = self._level_series(episode, t_ns)
        if clearance is None or levels is None:
            return None, None
        step_s = np.diff(t_ns, prepend=t_ns[0]) / 1e9
        step_s[step_s < 0.0] = 0.0
        finite = np.isfinite(levels) & np.isfinite(clearance)
        if not np.any(finite):
            return 0.0, 0.0
        linear_power = np.power(10.0, levels[finite] / 10.0)
        d_finite = clearance[finite]
        dt_finite = step_s[finite]
        intruded = d_finite < PERSONAL_SPACE_CLEARANCE_M
        aeps = (
            float(np.sum(linear_power[intruded] * dt_finite[intruded]))
            if np.any(intruded)
            else 0.0
        )
        d_clamped = np.maximum(d_finite, APA_MIN_DISTANCE_M)
        apa = float(np.sum((linear_power / (d_clamped * d_clamped)) * dt_finite))
        return aeps, apa

    @staticmethod
    def _effective_cost_of_transport(
        prior_results: typing.Mapping[str, typing.Any],
    ) -> float | None:
        """Cost of transport inflated by the collision count and the personal-space intrusion rate; NULL unless every input is measured."""
        cost_of_transport = prior_results.get("specific_cost_of_transport")
        collisions = prior_results.get("collision_amount")
        intrusion_integral = prior_results.get("personal_space_intrusion_integral")
        time_to_goal = prior_results.get("time_to_goal")
        if (
            cost_of_transport is None
            or collisions is None
            or intrusion_integral is None
            or time_to_goal is None
        ):
            return None
        if float(cost_of_transport) <= 0.0 or float(time_to_goal) <= 0.0:
            return None
        value = float(cost_of_transport) * (
            1.0
            + COLLISION_PENALTY * int(collisions)
            + INTRUSION_WEIGHT * float(intrusion_integral) / float(time_to_goal)
        )
        return round(value, COT_DECIMALS)

    def _episode_metrics(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: typing.Mapping[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """Holistic outputs over the given episode view; RSCC is populated in Layer 3."""
        aeps, apa = self._acoustic_proxemic_integrals(episode)
        return {
            "aeps_linear_s": aeps,
            "e_cot": self._effective_cost_of_transport(prior_results),
            "acoustic_proxemic_annoyance": apa,
            "relative_social_compliance_cost": None,
            f"relative_social_compliance_cost{NULL_REASON_SUFFIX}": NullReason.REFERENCE_MISSING.value,
        }

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Holistic values over the bundle's own samples, at whichever scope the bundle was sliced for."""
        return self._episode_metrics(episode, prior_results)
