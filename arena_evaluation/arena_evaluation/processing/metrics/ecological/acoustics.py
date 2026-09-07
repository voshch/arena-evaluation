from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class AcousticsCalculator(BaseMetricCalculator):
    """Ecological acoustic metrics on the native /acoustics time base.

    Metrics:
    - L_Aeq,T (ISO 1996-1:2016): equivalent continuous A-weighted level.
    - L_A10 / L_A50 / L_A90: levels exceeded 10/50/90 % of the time.
    - ASF / LTR: max positive rise rate of the 3-sample median filtered
      A-weighted Fast level, gated on L_AF >= 65 dBA. A proxy for transients
      from jerky torque bursts, not the biological startle reflex (which
      needs >80 dBA and millisecond rise times).
    - ASI: variance of 1 s Leq windows.
    - acoustic_cost: linearized acoustic energy per meter.
    """

    NAME = "acoustics"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["acoustics"]

    UNITS = {
        "l_aeq_t": "dBA",
        "l_a10": "dBA",
        "l_a50": "dBA",
        "l_a90": "dBA",
        "acoustic_startle_factor": "dBA/s",
        "acoustic_surge_index": "dBA²",
        "acoustic_cost": "s/m",
    }

    PRIMARY_OUTPUTS = ["l_aeq_t", "acoustic_startle_factor"]
    OUTPUT_DIRECTIONS = {
        "l_aeq_t": "lower",
        "acoustic_startle_factor": "lower",
        "acoustic_cost": "lower",
        "acoustic_surge_index": "lower",
        "l_a10": "lower",
        "l_a50": "lower",
        "l_a90": "lower",
    }

    _DEFAULT_DBA = 42.0  # dBA, idle noise floor
    _MIN_DBA = 1.0       # dBA, clamp before log
    _L_GATE_DBA = 65.0   # dBA, ASF level gate

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "l_aeq_t",
            "l_a10",
            "l_a50",
            "l_a90",
            "acoustic_startle_factor",
            "acoustic_surge_index",
            "acoustic_cost",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        topics = self.native_topics(episode)
        acous = topics.get("acoustics")
        if (
            acous is None
            or "total_level_af_dba" not in acous.columns
            or "time_ns" not in acous.columns
        ):
            return {k: None for k in self.output_keys()}

        acous = acous.sort("time_ns")
        try:
            dba = (
                acous["total_level_af_dba"]
                .cast(pl.Float64)
                .fill_null(strategy="forward")
                .fill_null(self._DEFAULT_DBA)
                .to_numpy()
            )
        except Exception:
            return {k: None for k in self.output_keys()}

        if len(dba) == 0 or np.all(np.isnan(dba)):
            return {k: None for k in self.output_keys()}

        dba = np.maximum(dba, self._MIN_DBA)
        time_ns = acous["time_ns"].to_numpy()
        if len(time_ns) < 2:
            return {k: None for k in self.output_keys()}

        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.01)
        dt = np.where(dt <= 0.0, 1e-6, dt)

        linear_energy = 10.0 ** (dba / 10.0)
        T = float(np.sum(dt))

        with np.errstate(divide="ignore", invalid="ignore"):
            l_aeq_t = 10.0 * np.log10(np.sum(linear_energy * dt) / T)
        if np.isnan(l_aeq_t) or np.isinf(l_aeq_t):
            l_aeq_t = None
        else:
            l_aeq_t = float(l_aeq_t)

        # L_AN convention: L_A10 is the level exceeded 10 % of the time
        l_a10 = float(np.percentile(dba, 90))
        l_a50 = float(np.percentile(dba, 50))
        l_a90 = float(np.percentile(dba, 10))

        dba_filt = self._median_filter_3(dba)
        dL = np.diff(dba_filt)
        dt_diff = np.diff(time_ns) / 1e9
        dt_diff = np.where(dt_diff <= 0.0, 1e-6, dt_diff)
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = dL / dt_diff

        # Only rises ending at or above the gate count as transients
        gated = np.where(dba_filt[1:] >= self._L_GATE_DBA, rates, 0.0)
        positive_rates = gated[gated > 0]
        acoustic_startle_factor = float(np.max(positive_rates)) if len(positive_rates) > 0 else 0.0

        window_s = 1.0
        if T >= window_s and len(dba) >= 2:
            cumsum_t = np.cumsum(dt)
            window_lecs = []
            t_start = 0.0
            while t_start + window_s <= T:
                t_end = t_start + window_s
                mask = (cumsum_t >= t_start) & (cumsum_t < t_end)
                if np.sum(mask) > 0:
                    window_le = np.mean(linear_energy[mask])
                    if window_le > 0:
                        window_lecs.append(10.0 * np.log10(window_le))
                t_start += window_s
            acoustic_surge_index = float(np.var(window_lecs)) if len(window_lecs) >= 2 else 0.0
        else:
            acoustic_surge_index = 0.0

        path_length = prior_results.get("path_length")
        if path_length is not None and path_length >= 0.1:
            tee_acoustic = float(np.sum(linear_energy * dt))
            acoustic_cost = float(tee_acoustic / path_length)
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
    def _median_filter_3(signal: np.ndarray) -> np.ndarray:
        """3-sample median filter, edges mirrored."""
        if len(signal) < 3:
            return signal.copy()
        result = np.zeros_like(signal)
        for i in range(len(signal)):
            if i == 0:
                result[i] = np.median([signal[0], signal[0], signal[1]])
            elif i == len(signal) - 1:
                result[i] = np.median([signal[i - 1], signal[i], signal[i]])
            else:
                result[i] = np.median([signal[i - 1], signal[i], signal[i + 1]])
        return result
