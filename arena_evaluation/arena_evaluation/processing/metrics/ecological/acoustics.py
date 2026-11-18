"""Ecological acoustic metrics on the native robot_acoustics time base."""

from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.storage.registry import Metric, MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

ACOUSTIC_REFERENCE_PRESSURE_PA: Final[float] = 20e-6
ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2: Final[float] = (
    ACOUSTIC_REFERENCE_PRESSURE_PA * ACOUSTIC_REFERENCE_PRESSURE_PA
)
NANOSECONDS_PER_SECOND: Final[float] = 1e9
MIN_DBA: Final[float] = 1.0
STARTLE_GATE_DBA: Final[float] = 65.0
SURGE_WINDOW_S: Final[float] = 1.0
MIN_PATH_LENGTH_M: Final[float] = 0.1
MIN_DT_S: Final[float] = 1e-6

_SCOPES: Final[tuple[MetricScope, ...]] = (MetricScope.ROBOT, MetricScope.ENCOUNTER)

L_AEQ_T = define_metric(
    "l_aeq_t",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA",
    description="Equivalent continuous A-weighted sound level over the scope.",
    null_reason=True,
)
L_A10 = define_metric(
    "l_a10",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA",
    description="A-weighted sound level exceeded 10 percent of the scope.",
    null_reason=True,
)
L_A50 = define_metric(
    "l_a50",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA",
    description="Median A-weighted sound level over the scope.",
    null_reason=True,
)
L_A90 = define_metric(
    "l_a90",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA",
    description="A-weighted sound level exceeded 90 percent of the scope.",
    null_reason=True,
)
ACOUSTIC_STARTLE_FACTOR = define_metric(
    "acoustic_startle_factor",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA/s",
    description="Maximum positive rise rate of 3-sample median-filtered level gated at 65 dBA.",
    null_reason=True,
)
ACOUSTIC_SURGE_INDEX = define_metric(
    "acoustic_surge_index",
    pl.Float64,
    scopes=_SCOPES,
    unit="dBA²",
    description="Variance of 1 s energy-equivalent A-weighted levels over the scope.",
    null_reason=True,
)
ACOUSTIC_COST = define_metric(
    "acoustic_cost",
    pl.Float64,
    scopes=_SCOPES,
    unit="Pa²·s/m",
    description="Emitted sound pressure-squared exposure per metre of robot path over the scope.",
    null_reason=True,
)

_METRICS: Final[tuple[Metric, ...]] = (
    L_AEQ_T,
    L_A10,
    L_A50,
    L_A90,
    ACOUSTIC_STARTLE_FACTOR,
    ACOUSTIC_SURGE_INDEX,
    ACOUSTIC_COST,
)


class AcousticsCalculator(BaseMetricCalculator):
    """Ecological acoustic metrics on the native robot_acoustics time base."""

    NAME = "acoustics"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["robot_acoustics"]
    PARAMS = ()

    OUTPUTS = _METRICS
    DIRECTIONS = {metric.name: Direction.LOWER for metric in _METRICS}
    PRIMARY_OUTPUTS = ["l_aeq_t", "acoustic_startle_factor"]

    @classmethod
    def required_topics_for_scope(
        cls, scope: MetricScope
    ) -> tuple[str | tuple[str, ...], ...]:
        return (
            ()
            if scope == MetricScope.ENCOUNTER
            else super().required_topics_for_scope(scope)
        )

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return self.calculate(
                episode, {"path_length_m": self._window_path_length_m(episode)}
            )
        raise ValueError(
            f"acoustics does not serve the {scope_slice.scope.value} scope"
        )

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        acous = self.native_topics(episode).get("robot_acoustics")
        if acous is None or len(acous) < 2:
            return dict.fromkeys(self.output_keys())

        frame = (
            acous.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("total_level_af_dba")
                .cast(pl.Float64)
                .forward_fill()
                .alias("dba"),
            )
            .drop_nulls("dba")
            .sort("t_ns")
        )
        if len(frame) < 2:
            return dict.fromkeys(self.output_keys())

        dba = np.maximum(frame["dba"].to_numpy(), MIN_DBA)
        t_ns = frame["t_ns"].to_numpy()
        if np.all(np.isnan(dba)):
            return dict.fromkeys(self.output_keys())

        dt = np.diff(t_ns) / NANOSECONDS_PER_SECOND
        dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.01)
        dt = np.where(dt <= 0.0, MIN_DT_S, dt)

        linear_energy = 10.0 ** (dba / 10.0)
        total_duration_s = float(np.sum(dt))

        with np.errstate(divide="ignore", invalid="ignore"):
            l_aeq_t_val = 10.0 * np.log10(np.sum(linear_energy * dt) / total_duration_s)
        l_aeq_t = (
            None
            if (np.isnan(l_aeq_t_val) or np.isinf(l_aeq_t_val))
            else float(l_aeq_t_val)
        )

        l_a10 = float(np.percentile(dba, 90))
        l_a50 = float(np.percentile(dba, 50))
        l_a90 = float(np.percentile(dba, 10))

        dba_filt = self._median_filter_3(dba)
        d_level = np.diff(dba_filt)
        dt_diff = np.where(
            np.diff(t_ns) / NANOSECONDS_PER_SECOND <= 0.0,
            MIN_DT_S,
            np.diff(t_ns) / NANOSECONDS_PER_SECOND,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = d_level / dt_diff

        gated = np.where(dba_filt[1:] >= STARTLE_GATE_DBA, rates, 0.0)
        positive_rates = gated[gated > 0]
        acoustic_startle_factor = (
            float(np.max(positive_rates)) if len(positive_rates) > 0 else 0.0
        )

        acoustic_surge_index = self._surge_index(dt, linear_energy, total_duration_s)

        path_length_m = prior_results.get("path_length_m")
        if path_length_m is not None and path_length_m >= MIN_PATH_LENGTH_M:
            dose_pa2s = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * float(
                np.sum(linear_energy * dt)
            )
            acoustic_cost = float(dose_pa2s / path_length_m)
        else:
            acoustic_cost = None

        return {
            "l_aeq_t": l_aeq_t,
            "l_a10": l_a10,
            "l_a50": l_a50,
            "l_a90": l_a90,
            "acoustic_startle_factor": acoustic_startle_factor,
            "acoustic_surge_index": acoustic_surge_index,
            "acoustic_cost": acoustic_cost,
        }

    @staticmethod
    def _surge_index(
        dt: np.ndarray, linear_energy: np.ndarray, total_duration_s: float
    ) -> float:
        """Variance of 1 s Leq windows over the scope."""
        if total_duration_s < SURGE_WINDOW_S or len(linear_energy) < 2:
            return 0.0
        cumsum_t = np.cumsum(dt)
        window_lecs: list[float] = []
        t_start = 0.0
        while t_start + SURGE_WINDOW_S <= total_duration_s:
            mask = (cumsum_t >= t_start) & (cumsum_t < t_start + SURGE_WINDOW_S)
            if np.sum(mask) > 0:
                window_le = float(np.mean(linear_energy[mask]))
                if window_le > 0:
                    window_lecs.append(10.0 * np.log10(window_le))
            t_start += SURGE_WINDOW_S
        return float(np.var(window_lecs)) if len(window_lecs) >= 2 else 0.0

    def _window_path_length_m(self, episode: AlignedEpisodeBundle) -> float | None:
        """Sum of the window's native /tf robot steps in metres, None without a usable track."""
        tracks = self.native_topics(episode).get("robot_tracks")
        if tracks is None or len(tracks) < 2:
            return None
        frame = (
            tracks.select(
                pl.col("t_ns").cast(pl.Int64),
                pl.col("x").cast(pl.Float64),
                pl.col("y").cast(pl.Float64),
            )
            .drop_nulls(["x", "y"])
            .sort("t_ns")
        )
        if len(frame) < 2:
            return None
        steps = (
            (pl.col("x").diff().pow(2) + pl.col("y").diff().pow(2))
            .sqrt()
            .alias("step_m")
        )
        return float(frame.select(steps).get_column("step_m").drop_nulls().sum())

    @staticmethod
    def _median_filter_3(signal: np.ndarray) -> np.ndarray:
        """3-sample median filter with mirrored edges."""
        if len(signal) < 3:
            return signal.copy()
        padded = np.pad(signal, 1, mode="edge")
        windows = np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=1)
        return np.median(windows, axis=1)
