from __future__ import annotations
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class ProxemicsCalculator(BaseMetricCalculator):
    """Calculates proxemics and personal space intrusion metrics."""
    
    NAME = "proxemics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["odom", "peds"]
    
    UNITS = {
        "num_pedestrians": "",
        "time_in_personal_space": "",
        "total_time_in_personal_space": "s",
        "avg_velocity_in_personal_space": "m/s",
    }

    PRIMARY_OUTPUTS = ["total_time_in_personal_space"]
    
    PERSONAL_SPACE_RADIUS = 1.2
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "num_pedestrians",
            "time_in_personal_space",
            "total_time_in_personal_space",
            "avg_velocity_in_personal_space",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "pos_x" not in episode.data.columns or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        pos_x, pos_y, _, _, _, _ = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy()
        peds_positions = episode.data["peds_positions"].to_list()
        num_pedestrians_col = (
            episode.data["num_pedestrians"].to_numpy()
            if "num_pedestrians" in episode.data.columns
            else None
        )
        
        velocity = prior_results.get("velocity", [])
        
        N = len(pos_x)
        if N == 0:
            return {
                "num_pedestrians": 0,
                "time_in_personal_space": [],
                "total_time_in_personal_space": 0.0,
                "avg_velocity_in_personal_space": 0.0,
            }
            
        in_personal_space_steps = []
        in_personal_space_time_s = 0.0
        vel_in_space = []
        
        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, 0.0)
        
        for i in range(N):
            peds = peds_positions[i]
            if not peds or len(peds) == 0:
                in_personal_space_steps.append(0)
                continue
                
            rx, ry = pos_x[i], pos_y[i]

            peds_arr = self._parse_peds(
                peds,
                num_pedestrians_col[i] if num_pedestrians_col is not None else None,
            )
            if peds_arr.shape[0] == 0:
                in_personal_space_steps.append(0)
                continue


            dx = peds_arr[:, 0] - rx
            dy = peds_arr[:, 1] - ry
            distances = np.sqrt(dx**2 + dy**2)
            
            in_space_count = np.sum(distances < self.PERSONAL_SPACE_RADIUS)
            
            in_personal_space_steps.append(int(in_space_count))
            
            if in_space_count > 0:
                in_personal_space_time_s += dt[i]
                if velocity and i < len(velocity):
                    vel_in_space.append(velocity[i])
                    
        if vel_in_space:
            avg_vel = float(np.nanmean(vel_in_space))
            if np.isnan(avg_vel) or np.isinf(avg_vel):
                avg_vel = 0.0
        else:
            avg_vel = 0.0        
        return {
            "num_pedestrians": episode.num_pedestrians,
            "time_in_personal_space": in_personal_space_steps,
            "total_time_in_personal_space": in_personal_space_time_s,
            "avg_velocity_in_personal_space": avg_vel,
        }
