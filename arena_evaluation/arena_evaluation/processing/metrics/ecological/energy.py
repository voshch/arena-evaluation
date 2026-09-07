from __future__ import annotations

import typing
import polars as pl
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

    def calculate(self, episode: "AlignedEpisodeBundle", dependencies: dict[str, typing.Any]) -> dict[str, typing.Any]:
        df = episode.data
        if df is None or df.is_empty():
            return {k: None for k in self.output_keys()}
            
        # We need a time axis in seconds relative to start
        if "time_ns" in df.columns:
            t_ns = df["time_ns"].to_numpy()
            t_s = (t_ns - t_ns[0]) * 1e-9
        else:
            t_s = np.zeros(len(df))

        # Power timeseries
        p_total = df["total_power_w"].to_numpy() if "total_power_w" in df.columns else np.zeros_like(t_s)
        p_static = df["static_power_w"].to_numpy() if "static_power_w" in df.columns else np.zeros_like(t_s)
        p_mech = df["total_mechanical_power_w"].to_numpy() if "total_mechanical_power_w" in df.columns else np.zeros_like(t_s)
        p_therm = df["total_thermal_power_w"].to_numpy() if "total_thermal_power_w" in df.columns else np.zeros_like(t_s)
        
        # We need to fill nulls which happen if the 'power' topic was joined but had missing data at odom timestamps (backward join leaves nulls at the start)
        def fill_nulls(arr):
            arr = np.array(arr, dtype=float)
            mask = np.isnan(arr)
            if np.all(mask):
                return np.zeros_like(arr)
            # Forward fill, then backward fill
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            out = arr[idx]
            
            # backward fill remaining
            mask = np.isnan(out)
            if np.any(mask):
                valid_idx = np.where(~np.isnan(arr))[0]
                if len(valid_idx) > 0:
                    out[mask] = arr[valid_idx[0]]
                else:
                    out[mask] = 0.0
            return out

        p_total = fill_nulls(p_total)
        p_static = fill_nulls(p_static)
        p_mech = fill_nulls(p_mech)
        p_therm = fill_nulls(p_therm)
        
        # Velocity timeseries
        vel = df["vel_linear"].to_numpy() if "vel_linear" in df.columns else np.zeros_like(t_s)
        vel = fill_nulls(vel)
        
        # Battery timeseries - normalized to start at 100.0% per episode
        batt = df["battery_soc_percent"].to_numpy() if "battery_soc_percent" in df.columns else np.zeros_like(t_s)
        batt = fill_nulls(batt)
        if len(batt) > 0:
            batt_initial = batt[0]
            batt_final = batt[-1]
            batt_drop = max(float(batt_initial - batt_final), 0.0)
            batt_normalized = np.clip(100.0 - (batt_initial - batt), 0.0, 100.0)
        else:
            batt_initial = 0.0
            batt_final = 0.0
            batt_drop = 0.0
            batt_normalized = batt

        # Integration for total energy consumption over the episode
        # Energy = integral of Power dt
        dt = np.diff(t_s, prepend=0.0)
        e_static = np.sum(p_static * dt) / 3600.0
        e_mech = np.sum(p_mech * dt) / 3600.0
        e_therm = np.sum(p_therm * dt) / 3600.0
        
        # Alternative: use the final value from the /energy topic
        # The /energy topic publishes cumulative energy since node start. 
        # The energy used in THIS episode is the final value minus the initial value.
        if "total_energy_consumed_wh" in df.columns:
            energy_arr = fill_nulls(df["total_energy_consumed_wh"].to_numpy())
            if len(energy_arr) > 0:
                e_total = energy_arr[-1] - energy_arr[0]
            else:
                e_total = np.sum(p_total * dt) / 3600.0
        else:
            e_total = np.sum(p_total * dt) / 3600.0
        power_peak = float(np.max(p_total)) if len(p_total) > 0 else 0.0

        return {
            "energy_static_wh": float(e_static),
            "energy_mechanical_wh": float(e_mech),
            "energy_thermal_wh": float(e_therm),
            "energy_total_wh": float(e_total),
            "power_peak_w": power_peak,
            "battery_soc_final": float(batt_final),
            "battery_soc_drop_pct": float(batt_drop),
            "timeseries_power_total_w": p_total.tolist(),
            "timeseries_power_static_w": p_static.tolist(),
            "timeseries_power_mechanical_w": p_mech.tolist(),
            "timeseries_power_thermal_w": p_therm.tolist(),
            "timeseries_battery_soc": batt_normalized.tolist(),
            "timeseries_velocity_linear": vel.tolist(),
            "timeseries_time_s": t_s.tolist(),
        }
