from __future__ import annotations
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PathMetricsCalculator(BaseMetricCalculator):
    """
    Calculates path-related metrics based on odometry.
    
    Metrics:
    - path: List of [x, y, yaw] coordinates
    - path_length_values: Step-by-step path length
    - path_length: Total path length
    - curvature: Menger curvature at each step
    - curvature_mean: Average curvature
    - normalized_curvature: Curvature normalized by step distance
    - roughness: Roughness of the path
    - roughness_mean: Average roughness
    - angle_over_length: Mean change of the angle over the complete path
    
    References:
    - Math implementation ported from legacy scripts/metrics.py
    """
    
    NAME = "path_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["odom"]
    
    UNITS = {
        "path": "m",
        "path_odom": "m",
        "path_length_values": "m",
        "path_length": "m",
        "curvature": "1/m",
        "curvature_mean": "1/m",
        "curvature_max": "1/m",
        "normalized_curvature": "",
        "roughness": "rad/m",
        "roughness_mean": "rad/m",
        "roughness_max": "rad/m",
        "angle_over_length": "rad/m",
    }

    PRIMARY_OUTPUTS = ["path_length", "curvature_mean", "roughness_mean"]
    
    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "path",
            "path_odom",
            "path_length_values",
            "path_length",
            "curvature",
            "curvature_mean",
            "curvature_max",
            "normalized_curvature",
            "roughness",
            "roughness_mean",
            "roughness_max",
            "angle_over_length",
        ]
        
    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        
        if episode.data is None or "pos_x" not in episode.data.columns:
            return {k: None for k in self.output_keys()}
            
        pos_x, pos_y, yaw, odom_x_trans, odom_y_trans, odom_yaw_trans = self.resolve_robot_pose(episode)
        path_odom_3d = np.column_stack((odom_x_trans, odom_y_trans, odom_yaw_trans))
            
        path_3d = np.column_stack((pos_x, pos_y, yaw))
        path_2d = np.column_stack((pos_x, pos_y))
        
        N = len(path_2d)
        if N < 2:
            return {
                "path": path_3d.tolist(),
                "path_odom": path_odom_3d.tolist(),
                "path_length_values": [],
                "path_length": 0.0,
                "curvature": [],
                "curvature_mean": 0.0,
                "normalized_curvature": [],
                "roughness": [],
                "roughness_mean": 0.0,
                "angle_over_length": 0.0,
            }
            
        path_length_values = np.linalg.norm(path_2d[1:] - path_2d[:-1], axis=1)
        path_length = float(np.sum(path_length_values))
        
        def angle_difference(x1, x2):
            return np.pi - np.abs(np.abs(x1 - x2) - np.pi)
            
        turn = angle_difference(yaw[:-1], yaw[1:])
        angle_over_length = float(np.abs(np.sum(turn) / path_length)) if path_length > 0 else 0.0
        
        diffs = np.linalg.norm(path_2d[1:] - path_2d[:-1], axis=1)
        keep_mask = np.concatenate(([True], diffs > 0.001))
        path_2d_clean = path_2d[keep_mask]
        
        N_clean = len(path_2d_clean)
        
        if N_clean < 3:
            return {
                "path": path_3d.tolist(),
                "path_odom": path_odom_3d.tolist(),
                "path_length_values": path_length_values.tolist(),
                "path_length": path_length,
                "curvature": [],
                "curvature_mean": 0.0,
                "curvature_max": 0.0,
                "normalized_curvature": [],
                "roughness": [],
                "roughness_mean": 0.0,
                "roughness_max": 0.0,
                "angle_over_length": angle_over_length,
            }
            
        p0 = path_2d_clean[:-2]
        p1 = path_2d_clean[1:-1]
        p2 = path_2d_clean[2:]
        
        d01 = np.linalg.norm(p0 - p1, axis=1)
        d12 = np.linalg.norm(p1 - p2, axis=1)
        d20 = np.linalg.norm(p2 - p0, axis=1)
        
        v1 = p1 - p0
        v2 = p2 - p0
        triangle_area = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]) / 2.0

        divisor = d01 * d12 * d20
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.where(divisor == 0, 0, 4 * triangle_area / divisor)
            normalized_curvature = curvature * (d01 + d12)

            roughness = np.where(d20 == 0, 0, 2 * triangle_area / np.square(d20))
            
        return {
            "path": path_3d.tolist(),
            "path_odom": path_odom_3d.tolist(),
            "path_length_values": path_length_values.tolist(),
            "path_length": path_length,
            "curvature": curvature.tolist(),
            "curvature_mean": float(np.mean(curvature)) if len(curvature) > 0 else 0.0,
            "curvature_max": float(np.max(curvature)) if len(curvature) > 0 else 0.0,
            "normalized_curvature": normalized_curvature.tolist(),
            "roughness": roughness.tolist(),
            "roughness_mean": float(np.mean(roughness)) if len(roughness) > 0 else 0.0,
            "roughness_max": float(np.max(roughness)) if len(roughness) > 0 else 0.0,
            "angle_over_length": angle_over_length,
        }
