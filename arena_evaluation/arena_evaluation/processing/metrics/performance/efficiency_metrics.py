from __future__ import annotations
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PathEfficiencyCalculator(BaseMetricCalculator):
    """Calculates ratio of Theta* optimal geodesic distance over actual path length."""
    
    NAME = "path_efficiency"
    CATEGORY = "performance"
    DEPENDS_ON = ["path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = [("tf_gt", "odom")]
    UNITS = {"path_efficiency": ""}

    PRIMARY_OUTPUTS = ["path_efficiency"]
    OUTPUT_DIRECTIONS = {"path_efficiency": "higher"}
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "path_efficiency",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        path_length = prior_results.get("path_length", 0.0) or 0.0
        
        if path_length <= 1e-9 or not episode.start_pos or not episode.goal_pos:
            return {"path_efficiency": 0.0}
            
        l0 = prior_results.get("theta_star_length")
        if l0 is None or l0 <= 0:
            start = np.array(episode.start_pos[:2])
            goal = np.array(episode.goal_pos[:2])
            l0 = float(np.linalg.norm(goal - start))
        
        if np.isnan(l0) or np.isinf(l0) or np.isnan(path_length):
            return {"path_efficiency": 0.0}
            
        try:
            efficiency = float(l0 / max(path_length, 0.001))
            if np.isnan(efficiency):
                efficiency = 0.0
            else:
                efficiency = min(max(efficiency, 0.0), 1.0)
        except ZeroDivisionError:
            efficiency = 0.0

        return {
            "path_efficiency": efficiency,
        }

