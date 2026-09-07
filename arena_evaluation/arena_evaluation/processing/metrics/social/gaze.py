from __future__ import annotations

import ast
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class GazeMetricsCalculator(BaseMetricCalculator):
    """Calculates mutual and directional gaze metrics between robot and pedestrians."""
    
    NAME = "gaze_metrics"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom", "peds"]
    
    UNITS = {
        "time_looking_at_pedestrians": "",
        "total_time_looking_at_pedestrians": "s",
        "time_looked_at_by_pedestrians": "",
        "total_time_looked_at_by_pedestrians": "s",
    }

    PRIMARY_OUTPUTS = [
        "total_time_looking_at_pedestrians",
        "total_time_looked_at_by_pedestrians",
    ]
    
    GAZE_CONE_DEG = 5.0
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_looking_at_pedestrians",
            "total_time_looking_at_pedestrians",
            "time_looked_at_by_pedestrians",
            "total_time_looked_at_by_pedestrians",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "pos_x" not in episode.data.columns or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)
        time_ns = episode.data["time_ns"].to_numpy()
        
        peds_positions = episode.data["peds_positions"].to_list()
        num_pedestrians_col = (
            episode.data["num_pedestrians"].to_numpy()
            if "num_pedestrians" in episode.data.columns
            else None
        )
        if "peds_headings" in episode.data.columns:
            peds_headings = episode.data["peds_headings"].to_list()
        else:
            peds_headings = [None] * len(pos_x)
            
        N = len(pos_x)
        
        dt = np.diff(time_ns) / 1e9
        dt = np.append(dt, 0.0)
        
        looking_at = []
        looking_at_time = 0.0
        
        looked_at = []
        looked_at_time = 0.0
        
        cone_rad = np.radians(self.GAZE_CONE_DEG)
        
        def angle_diff(a1, a2):
            return np.pi - np.abs(np.abs(a1 - a2) - np.pi)
            
        for i in range(N):
            peds = peds_positions[i]
            headings = peds_headings[i]
            
            if not peds or len(peds) == 0:
                looking_at.append(0)
                looked_at.append(0)
                continue
                
            rx, ry, ryaw = pos_x[i], pos_y[i], yaw[i]
            
            n_peds = num_pedestrians_col[i] if num_pedestrians_col is not None else None

            peds_arr = self._parse_peds(peds, n_peds)
            n_resolved_peds = peds_arr.shape[0]

            if isinstance(peds, str):
                try:
                    head_arr = np.array(ast.literal_eval(headings)) if headings else np.zeros(n_resolved_peds)
                except:
                    head_arr = np.zeros(n_resolved_peds)
            else:
                head_arr = np.array(headings) if headings is not None else np.zeros(n_resolved_peds)
                
            if peds_arr.ndim != 2 or peds_arr.shape[1] < 2:
                looking_at.append(0)
                looked_at.append(0)
                continue
                
            dx = peds_arr[:, 0] - rx
            dy = peds_arr[:, 1] - ry
            
            angle_to_ped = np.arctan2(dy, dx)
            diff_robot_to_ped = angle_diff(ryaw, angle_to_ped)
            
            is_looking = np.sum(diff_robot_to_ped <= cone_rad)
            looking_at.append(int(is_looking))
            if is_looking > 0:
                looking_at_time += dt[i]
                
            angle_to_robot = np.arctan2(-dy, -dx)
            if len(head_arr) == len(peds_arr):
                diff_ped_to_robot = angle_diff(head_arr, angle_to_robot)
                is_looked_at = np.sum(diff_ped_to_robot <= cone_rad)
                looked_at.append(int(is_looked_at))
                if is_looked_at > 0:
                    looked_at_time += dt[i]
            else:
                looked_at.append(0)
                
        return {
            "time_looking_at_pedestrians": looking_at,
            "total_time_looking_at_pedestrians": looking_at_time,
            "time_looked_at_by_pedestrians": looked_at,
            "total_time_looked_at_by_pedestrians": looked_at_time,
        }
