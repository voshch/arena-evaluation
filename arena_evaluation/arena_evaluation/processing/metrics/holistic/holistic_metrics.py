from __future__ import annotations

import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class HolisticMetricsCalculator(BaseMetricCalculator):
    """Holistic cross-domain metrics synthesizing energy, safety, and acoustic exposure."""

    NAME = "holistic_metrics"
    CATEGORY = "holistic"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["energy_extended", "collision_metrics", "proxemics_extended", "time_metrics", "acoustic_exposure"]
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    UNITS = {
        "aeps_linear_s": "s",
        "e_cot": "",
    }

    PRIMARY_OUTPUTS = ["e_cot"]
    OUTPUT_DIRECTIONS = {"e_cot": "lower", "aeps_linear_s": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["aeps_linear_s", "e_cot"]

    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        result: dict[str, typing.Any] = {"aeps_linear_s": None, "e_cot": None}

        # --- AEPS: Acoustic Exposure in Personal Space ---
        # integral of 10^(Lp(t)/10) dt for frames where d_eff < 1.2m
        ts_min_clearance = prior_results.get("timeseries_min_ped_clearance")
        ts_acoustic = prior_results.get("timeseries_acoustic_exposure_dba") or prior_results.get("timeseries_total_level_dba")
        
        if ts_min_clearance is not None:
            clearance = np.array([c if c is not None else float('inf') for c in ts_min_clearance], dtype=float)
            if ts_acoustic is not None and len(ts_acoustic) > 0:
                # If nested (list of per-ped exposures), take maximum exposure received by any ped at each frame
                if isinstance(ts_acoustic[0], (list, tuple, np.ndarray)):
                    acoustic = np.array([max(f) if len(f) > 0 else 45.0 for f in ts_acoustic], dtype=float)
                else:
                    acoustic = np.array(ts_acoustic, dtype=float)
            elif episode.data is not None and "total_level_af_dba" in episode.data.columns:
                acoustic = episode.data["total_level_af_dba"].to_numpy().astype(float)
            else:
                acoustic = np.full(len(clearance), 45.0, dtype=float)

            min_len = min(len(clearance), len(acoustic))
            if min_len > 0:
                clearance = clearance[:min_len]
                acoustic = acoustic[:min_len]
                # Default aeps to 0.0 s when no intrusions occur
                aeps = 0.0
                if episode.data is not None and "time_ns" in episode.data.columns:
                    t_ns = episode.data["time_ns"].to_numpy()[:min_len]
                    dt = np.diff(t_ns, prepend=t_ns[0]) / 1e9
                    dt[dt < 0] = 0.0
                    mask = clearance < 1.2
                    if np.any(mask):
                        linear_power = np.power(10.0, acoustic[mask] / 10.0)
                        aeps = float(np.sum(linear_power * dt[mask]))
                result["aeps_linear_s"] = float(aeps)

        # --- E-CoT: Effective Cost of Transport ---
        # CoT * (1.0 + 2.0 * N_coll + 0.5 * PSII / T)
        cot = prior_results.get("specific_cost_of_transport")
        n_coll = prior_results.get("collision_amount", 0)
        psii = prior_results.get("personal_space_intrusion_integral", 0.0)
        time_to_goal = prior_results.get("time_to_goal", 0.0)

        if cot is not None and cot > 0:
            n_coll = n_coll if n_coll is not None else 0
            psii = psii if psii is not None else 0.0
            t = max(float(time_to_goal), 0.01) if time_to_goal else 0.01
            e_cot = float(cot) * (1.0 + 2.0 * int(n_coll) + 0.5 * float(psii) / t)
            result["e_cot"] = round(e_cot, 6)

        return result
