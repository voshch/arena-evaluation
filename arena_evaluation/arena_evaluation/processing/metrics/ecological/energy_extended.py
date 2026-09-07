from __future__ import annotations

import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


def _median_filter_window3(arr: np.ndarray) -> np.ndarray:
    """3-sample moving median, to reject single-frame outliers."""
    n = len(arr)
    if n < 3:
        return arr.copy()
    result = np.empty_like(arr)
    result[0] = np.median(arr[:2])
    for i in range(1, n - 1):
        result[i] = np.median(arr[i - 1 : i + 2])
    result[-1] = np.median(arr[-2:])
    return result


class EnergyExtendedCalculator(BaseMetricCalculator):
    """Extended ecological energy metrics.

    Computes Specific Cost of Transport, Energy per Meter, Peak-to-Mean Power
    Ratio, and Standstill Energy Penalty from prior energy, path, and motion
    results. All energy quantities are reported in watt-hours.
    """

    NAME = "energy_extended"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["energy", "path_metrics", "motion_metrics"]
    REQUIRED_TOPICS = [("power", "energy", "odom")]

    UNITS = {
        "specific_cost_of_transport": "",
        "energy_per_meter": "Wh/m",
        "peak_to_mean_power_ratio": "",
        "standstill_energy_penalty_wh": "Wh",
        "kinetic_energy_demand_j": "J",
        "friction_dissipation_j": "J",
    }

    PRIMARY_OUTPUTS = ["specific_cost_of_transport", "energy_per_meter"]
    OUTPUT_DIRECTIONS = {
        "specific_cost_of_transport": "lower",
        "energy_per_meter": "lower",
        "peak_to_mean_power_ratio": "lower",
        "standstill_energy_penalty_wh": "lower",
        "kinetic_energy_demand_j": "lower",
        "friction_dissipation_j": "lower",
    }

    def __init__(self, robot_params):
        super().__init__(robot_params)
        model = getattr(robot_params, 'model', None) if robot_params else None
        self._rolling_resistance = self._load_rolling_resistance(model) if model else 0.015

    @staticmethod
    def _load_rolling_resistance(model: str) -> float:
        import os
        import yaml
        try:
            from ament_index_python.packages import get_package_share_directory
            power_path = os.path.join(
                get_package_share_directory("arena_robots"),
                "robots", model, "telemetry", "power.yaml"
            )
            with open(power_path) as f:
                data = yaml.safe_load(f)
            return float(data.get("power_system", {}).get("rolling_resistance_crr", 0.015))
        except Exception:
            return 0.015

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "specific_cost_of_transport",
            "energy_per_meter",
            "peak_to_mean_power_ratio",
            "standstill_energy_penalty_wh",
            "kinetic_energy_demand_j",
            "friction_dissipation_j",
        ]

    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None:
            return {k: None for k in self.output_keys()}

        g = 9.81         # m/s^2
        v_thresh = 0.01  # m/s, standstill threshold
        mass = self.robot_params.mass

        path_length: float | None = prior_results.get("path_length")
        energy_total_wh: float | None = prior_results.get("energy_total_wh")
        velocity_list: list | None = prior_results.get("velocity")
        power_ts: list | None = prior_results.get("timeseries_power_total_w")
        time_ts: list | None = prior_results.get("timeseries_time_s")

        result: dict[str, typing.Any] = {}

        # Cost of transport over total energy, i.e. the whole navigation
        # task including static and compute overhead, not just locomotion.
        # An undeclared mass leaves it unreported rather than guessed.
        cot = None
        if (
            mass > 0
            and energy_total_wh is not None
            and path_length is not None
            and path_length > 0.1
            and energy_total_wh > 0
        ):
            e_total_j = float(energy_total_wh) * 3600.0
            with np.errstate(divide="ignore", invalid="ignore"):
                cot = e_total_j / (mass * g * float(path_length))
                if not np.isfinite(cot):
                    cot = None
        result["specific_cost_of_transport"] = float(cot) if cot is not None else None

        epm = None
        if (
            energy_total_wh is not None
            and path_length is not None
            and path_length > 0.1
            and energy_total_wh > 0
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                epm = float(energy_total_wh) / float(path_length)
                if not np.isfinite(epm):
                    epm = None
        result["energy_per_meter"] = float(epm) if epm is not None else None

        pmpr = None
        if power_ts is not None and len(power_ts) > 0:
            p_arr = np.array(power_ts, dtype=float)
            p_filt = _median_filter_window3(p_arr)
            p_mean = float(np.mean(p_filt))
            if p_mean > 0.01:
                p_max = float(np.max(p_filt))
                with np.errstate(divide="ignore", invalid="ignore"):
                    pmpr = p_max / p_mean
                    if not np.isfinite(pmpr):
                        pmpr = None
        result["peak_to_mean_power_ratio"] = float(pmpr) if pmpr is not None else None

        vel = (
            np.array(velocity_list, dtype=float)
            if velocity_list is not None and len(velocity_list) > 0
            else np.array([])
        )
        p_ts = (
            np.array(power_ts, dtype=float)
            if power_ts is not None and len(power_ts) > 0
            else np.array([])
        )
        t_ts = (
            np.array(time_ts, dtype=float)
            if time_ts is not None and len(time_ts) > 0
            else np.array([])
        )

        lengths = [len(vel), len(p_ts), len(t_ts)]
        min_len = min(lengths) if lengths else 0

        if min_len > 0:
            vel_trim = vel[:min_len]
            p_ts_trim = p_ts[:min_len]
            t_ts_trim = t_ts[:min_len]

            dt = np.diff(t_ts_trim, prepend=0.0)
            dt[dt < 0] = 0.0

            is_stationary = np.abs(vel_trim) < v_thresh
            sep_wh = 0.0
            if np.any(is_stationary):
                padded = np.concatenate([[False], is_stationary, [False]])
                edges = np.diff(padded.astype(np.int8))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]

                for s, e in zip(starts, ends):
                    block_duration = float(np.sum(dt[s:e]))
                    if block_duration >= 0.5:  # ignore micro-jitter blocks
                        sep_wh += float(np.sum(p_ts_trim[s:e] * dt[s:e])) / 3600.0

            result["standstill_energy_penalty_wh"] = float(sep_wh)
        else:
            result["standstill_energy_penalty_wh"] = None

        # Kinetic Energy Demand
        ked = None
        if mass > 0 and len(vel) > 1 and len(t_ts) > 1:
            inertia = 0.5 * mass * self.robot_params.robot_radius ** 2
            omega = np.array(prior_results.get("angular_velocity", []), dtype=float)
            if len(omega) == 0:
                omega = np.zeros_like(vel)
            min_n = min(len(vel), len(omega), len(t_ts))
            vel_k = vel[:min_n]
            omega_k = omega[:min_n]
            t_k = t_ts[:min_n]
            ke = 0.5 * mass * vel_k**2 + 0.5 * inertia * omega_k**2
            dke = np.abs(np.diff(ke))
            ked = float(np.sum(dke))
        result["kinetic_energy_demand_j"] = ked

        # Friction Dissipation
        fd = None
        if mass > 0 and len(vel) > 1 and len(t_ts) > 1:
            min_n = min(len(vel), len(t_ts))
            vel_f = vel[:min_n]
            t_f = t_ts[:min_n]
            dt_f = np.diff(t_f, prepend=0.0)
            dt_f[dt_f < 0] = 0.0
            fd = float(np.sum(self._rolling_resistance * mass * 9.81 * np.abs(vel_f) * dt_f))
        result["friction_dissipation_j"] = fd

        return result
