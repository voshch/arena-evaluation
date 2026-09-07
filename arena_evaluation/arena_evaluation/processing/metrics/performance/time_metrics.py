from __future__ import annotations
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class TimeMetricsCalculator(BaseMetricCalculator):
    """
    Calculates time-related metrics.
    
    Metrics:
    - time_to_goal: Total duration of the episode in seconds
    - idling_time: Total time spent with velocity near zero (< 0.01 m/s)
    """
    
    NAME = "time_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["odom"]
    
    UNITS = {
        "time": "ns",
        "time_diff": "ns",
        "time_to_goal": "s",
        "idling_time": "s",
    }

    PRIMARY_OUTPUTS = ["time_to_goal", "idling_time"]
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time",
            "time_diff",
            "time_to_goal",
            "idling_time",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "time_ns" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        time_ns = episode.data["time_ns"].to_numpy()
        N = len(time_ns)
        
        if N < 2:
            return {
                "time": [],
                "time_diff": 0,
                "time_to_goal": 0.0,
                "idling_time": 0.0,
            }
            
        duration_s = float(time_ns[-1] - time_ns[0]) / 1e9
        
        idling_time = 0.0
        velocity = prior_results.get("velocity", [])
        if velocity and len(velocity) == N:
            vel_arr = np.array(velocity)
            idle_mask = vel_arr < 0.01
            
            dt = np.diff(time_ns) / 1e9
            idling_time = float(np.sum(dt[idle_mask[:-1]]))
            
        return {
            "time": time_ns.tolist(),
            "time_diff": int(time_ns[-1] - time_ns[0]),
            "time_to_goal": duration_s,
            "idling_time": idling_time,
        }
