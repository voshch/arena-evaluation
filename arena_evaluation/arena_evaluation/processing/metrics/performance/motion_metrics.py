from __future__ import annotations
import typing
import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class MotionMetricsCalculator(BaseMetricCalculator):
    """Calculates robot velocity, acceleration, and jerk metrics."""
    
    NAME = "motion_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom"]
    
    UNITS = {
        "velocity": "m/s",
        "velocity_mean": "m/s",
        "velocity_max": "m/s",
        "acceleration": "m/s²",
        "acceleration_mean": "m/s²",
        "acceleration_max": "m/s²",
        "jerk": "m/s³",
        "jerk_mean": "m/s³",
        "jerk_max": "m/s³",
    }

    PRIMARY_OUTPUTS = ["velocity_mean", "velocity_max", "jerk_mean", "jerk_max"]
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "velocity",
            "velocity_mean",
            "velocity_max",
            "acceleration",
            "acceleration_mean",
            "acceleration_max",
            "jerk",
            "jerk_mean",
            "jerk_max",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy() if (episode.data is not None and "time_ns" in episode.data.columns) else np.arange(len(pos_x)) * 100_000_000

        if episode.data is not None and "vel_linear" in episode.data.columns:
            valid = pl.col("vel_linear").is_not_null() & ~pl.col("vel_linear").is_nan()
            v_max = self.robot_params.max_linear_velocity if self.robot_params else 0.0
            if v_max > 0.0:
                valid = valid & (pl.col("vel_linear").abs() <= v_max)
            episode.data = episode.data.filter(valid)
            vel_abs = np.abs(episode.data["vel_linear"].to_numpy())
            time_ns = episode.data["time_ns"].to_numpy()
        elif len(pos_x) > 1:
            vel_abs = self.speed_from_pose(pos_x, pos_y, time_ns)
        else:
            return {k: None for k in self.output_keys()}
        
        N = len(vel_abs)
        if N < 2:
            return {
                "velocity": vel_abs.tolist(),
                "velocity_mean": float(np.mean(vel_abs)) if N > 0 else 0.0,
                "velocity_max": float(np.max(vel_abs)) if N > 0 else 0.0,
                "acceleration": [],
                "acceleration_mean": 0.0,
                "acceleration_max": 0.0,
                "jerk": [],
                "jerk_mean": 0.0,
                "jerk_max": 0.0,
            }
            
        dt = np.diff(time_ns) / 1e9
        dt = np.where(dt <= 0.0, 1e-6, dt)

        acceleration = np.diff(vel_abs) / dt
        
        if N < 3:
            return {
                "velocity": vel_abs.tolist(),
                "velocity_mean": float(np.mean(vel_abs)),
                "velocity_max": float(np.max(vel_abs)),
                "acceleration": acceleration.tolist(),
                "acceleration_mean": float(np.mean(np.abs(acceleration))),
                "acceleration_max": float(np.max(np.abs(acceleration))),
                "jerk": [],
                "jerk_mean": 0.0,
                "jerk_max": 0.0,
            }
            
        jerk = np.diff(acceleration) / dt[:-1]
        
        return {
            "velocity": vel_abs.tolist(),
            "velocity_mean": float(np.mean(vel_abs)),
            "velocity_max": float(np.max(vel_abs)),
            "acceleration": acceleration.tolist(),
            "acceleration_mean": float(np.mean(np.abs(acceleration))),
            "acceleration_max": float(np.max(np.abs(acceleration))),
            "jerk": jerk.tolist(),
            "jerk_mean": float(np.mean(np.abs(jerk))),
            "jerk_max": float(np.max(np.abs(jerk))),
        }

