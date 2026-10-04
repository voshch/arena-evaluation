from __future__ import annotations

import typing

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class EnergyMetricCalculator(BaseMetricCalculator):
    NAME = "energy"
    CATEGORY = "ecological"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("power", "energy", "odom")]

    UNITS = {
        "energy_static_wh": "Wh",
        "energy_mechanical_wh": "Wh",
        "energy_thermal_wh": "Wh",
        "energy_total_wh": "Wh",
        "power_peak_w": "W",
        "battery_soc_final": "%",
        "battery_soc_drop_pct": "%",
    }

    PRIMARY_OUTPUTS = ["energy_total_wh", "energy_mechanical_wh", "power_peak_w"]

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            # Scalars (aggregates for the episode)
            "energy_static_wh",
            "energy_mechanical_wh",
            "energy_thermal_wh",
            "energy_total_wh",
            "power_peak_w",
            "battery_soc_final",
            "battery_soc_drop_pct",
            # Timeseries (arrays)
            "timeseries_power_total_w",
            "timeseries_power_static_w",
            "timeseries_power_mechanical_w",
            "timeseries_power_thermal_w",
            "timeseries_battery_soc",
            "timeseries_velocity_linear",
            "timeseries_time_s",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, dependencies: dict[str, typing.Any]) -> dict[str, typing.Any]:
        df = episode.data
        if df is None or df.is_empty():
            return {k: None for k in self.output_keys()}

        # We need a time axis in seconds relative to start
        if "time_ns" in df.columns:
            t_ns = df["time_ns"].to_numpy()
            t_s = (t_ns - t_ns[0]) * 1e-9
        else:
            t_s = np.zeros(len(df))

        # We need to fill nulls which happen if the 'power' topic was joined but had missing data at odom timestamps (backward join leaves nulls at the start)
        def fill_nulls(arr: np.ndarray) -> np.ndarray:
            mask = np.isnan(arr)
            # Forward fill, then backward fill
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            out = arr[idx]

            # backward fill remaining
            mask = np.isnan(out)
            if np.any(mask):
                valid_idx = np.where(~np.isnan(arr))[0]
                out[mask] = arr[valid_idx[0]]
            return out

        def column_or_none(name: str) -> np.ndarray | None:
            if name not in df.columns:
                return None
            arr = df[name].to_numpy().astype(float)
            if np.all(np.isnan(arr)):
                return None
            return fill_nulls(arr)

        p_total = column_or_none("total_power_w")
        p_static = column_or_none("static_power_w")
        p_mech = column_or_none("total_mechanical_power_w")
        p_therm = column_or_none("total_thermal_power_w")

        # Velocity timeseries
        vel = column_or_none("vel_linear")

        # Battery timeseries - normalized to start at 100.0% per episode
        batt = column_or_none("battery_soc_percent")
        if batt is not None:
            batt_initial = batt[0]
            batt_final = float(batt[-1])
            batt_drop = max(float(batt_initial - batt_final), 0.0)
            batt_normalized = np.clip(100.0 - (batt_initial - batt), 0.0, 100.0)
        else:
            batt_final = None
            batt_drop = None
            batt_normalized = None

        # Integration for total energy consumption over the episode
        # Energy = integral of Power dt
        dt = np.diff(t_s, prepend=0.0)
        e_static = float(np.sum(p_static * dt) / 3600.0) if p_static is not None else None
        e_mech = float(np.sum(p_mech * dt) / 3600.0) if p_mech is not None else None
        e_therm = float(np.sum(p_therm * dt) / 3600.0) if p_therm is not None else None

        # Alternative: use the final value from the /energy topic
        # The /energy topic publishes cumulative energy since node start.
        # The energy used in THIS episode is the final value minus the initial value.
        energy_arr = column_or_none("total_energy_consumed_wh")
        if energy_arr is not None:
            e_total = float(energy_arr[-1] - energy_arr[0])
        elif p_total is not None:
            e_total = float(np.sum(p_total * dt) / 3600.0)
        else:
            e_total = None
        power_peak = float(np.max(p_total)) if p_total is not None else None

        return {
            "energy_static_wh": e_static,
            "energy_mechanical_wh": e_mech,
            "energy_thermal_wh": e_therm,
            "energy_total_wh": e_total,
            "power_peak_w": power_peak,
            "battery_soc_final": batt_final,
            "battery_soc_drop_pct": batt_drop,
            "timeseries_power_total_w": p_total.tolist() if p_total is not None else None,
            "timeseries_power_static_w": p_static.tolist() if p_static is not None else None,
            "timeseries_power_mechanical_w": p_mech.tolist() if p_mech is not None else None,
            "timeseries_power_thermal_w": p_therm.tolist() if p_therm is not None else None,
            "timeseries_battery_soc": batt_normalized.tolist() if batt_normalized is not None else None,
            "timeseries_velocity_linear": vel.tolist() if vel is not None else None,
            "timeseries_time_s": t_s.tolist(),
        }
