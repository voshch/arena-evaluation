from __future__ import annotations
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PedestrianPathMetricsCalculator(BaseMetricCalculator):
    """Calculates time-synchronized pedestrian trajectory paths."""
    
    NAME = "pedestrian_path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["peds"]
    REQUIRES_PEDSIM = True
    
    UNITS = {
        "pedestrian_path": "m"
    }
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return ["pedestrian_path"]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "peds_positions" not in episode.data.columns:
            return {"pedestrian_path": []}
            
        peds_positions = episode.data["peds_positions"].to_list()
        
        T = len(peds_positions)
        if T == 0:
            return {"pedestrian_path": []}
            
        max_peds = 0
        for step_peds in peds_positions:
            if isinstance(step_peds, list):
                max_peds = max(max_peds, len(step_peds) // 3)
                
        if max_peds == 0:
            return {"pedestrian_path": []}
            
        paths = [[[float('nan'), float('nan'), float('nan')] for _ in range(T)] for _ in range(max_peds)]
        
        for t, step_peds in enumerate(peds_positions):
            if not isinstance(step_peds, list):
                if t > 0:
                    for k in range(max_peds):
                        paths[k][t] = paths[k][t-1]
                continue
                
            num_peds = len(step_peds) // 3
            for k in range(num_peds):
                paths[k][t] = [step_peds[3*k], step_peds[3*k + 1], step_peds[3*k + 2]]
                
            for k in range(num_peds, max_peds):
                if t > 0:
                    paths[k][t] = paths[k][t-1]
                
        valid_paths = []
        for path in paths:
            if any(not np.isnan(pt[0]) for pt in path):
                valid_paths.append(path)
                
        return {"pedestrian_path": valid_paths}
