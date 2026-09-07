from __future__ import annotations

import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class MutualAccommodationCalculator(BaseMetricCalculator):
    """
    Mutual Accommodation Ratio (MAR) and multi-agent reference metrics.
    
    Evaluates the allocation of collision-avoidance responsibility by comparing:
    1. Dynamic Robot vs. Unobstructed Robot Reference (P_unobstructed)
    2. Dynamic Pedestrians vs. Unhindered Pedestrians Reference (P_unhindered)
    """

    NAME = "mutual_accommodation"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics_extended", "path_metrics", "energy_extended"]
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    UNITS = {
        "mar": "",
        "pfi": "m",
        "robot_path_deviation_m": "m",
        "relative_throughput": "",
        "kcsc_wh": "Wh",
    }

    PRIMARY_OUTPUTS = ["mar", "pfi", "relative_throughput", "kcsc_wh"]
    OUTPUT_DIRECTIONS = {
        "mar": "lower",
        "pfi": "lower",
        "robot_path_deviation_m": "lower",
        "relative_throughput": "higher",
        "kcsc_wh": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "mar",
            "pfi",
            "robot_path_deviation_m",
            "relative_throughput",
            "kcsc_wh",
        ]

    @staticmethod
    def compute_orthogonal_deviation(
        pts: np.ndarray | list, ref_polyline: np.ndarray | list
    ) -> tuple[float, float]:
        """
        Compute (max_dev, mean_dev) of trajectory points orthogonally projected onto reference polyline.
        Decouples lateral detour from temporal/speed differences (Frenet-Serret frame).
        """
        if pts is None or ref_polyline is None:
            return 0.0, 0.0

        p_arr = np.asarray(pts, dtype=np.float64)
        if p_arr.ndim == 3:
            p_arr = p_arr[0]
        r_arr = np.asarray(ref_polyline, dtype=np.float64)
        if r_arr.ndim == 3:
            r_arr = r_arr[0]

        if len(p_arr) == 0 or len(r_arr) < 2:
            return 0.0, 0.0

        displacements = []
        for p in p_arr:
            min_d = float("inf")
            for i in range(len(r_arr) - 1):
                a = r_arr[i, :2]
                b = r_arr[i + 1, :2]
                ab = b - a
                ab_len2 = float(np.dot(ab, ab))
                if ab_len2 <= 1e-12:
                    d = float(np.linalg.norm(p[:2] - a))
                else:
                    t = float(np.clip(np.dot(p[:2] - a, ab) / ab_len2, 0.0, 1.0))
                    proj = a + t * ab
                    d = float(np.linalg.norm(p[:2] - proj))
                if d < min_d:
                    min_d = d
            displacements.append(min_d)

        return float(np.max(displacements)), float(np.mean(displacements))

    @staticmethod
    def compute_pfi_dtw(actual_paths: list, ref_paths: list) -> float | None:
        """
        Compute Pedestrian Flow Interference (PFI) via Dynamic Time Warping (DTW).
        """
        if not actual_paths or not ref_paths:
            return None

        errors: list[float] = []
        n = min(len(actual_paths), len(ref_paths))
        for i in range(n):
            a, r = actual_paths[i], ref_paths[i]
            if a is None or r is None or len(a) < 2 or len(r) < 2:
                continue
            a_arr = np.asarray(a, dtype=np.float64)
            r_arr = np.asarray(r, dtype=np.float64)
            if a_arr.ndim != 2 or r_arr.ndim != 2 or a_arr.shape[1] < 2 or r_arr.shape[1] < 2:
                continue
            if len(a_arr) > 300:
                idx_a = np.linspace(0, len(a_arr) - 1, 300, dtype=int)
                a_arr = a_arr[idx_a]
            if len(r_arr) > 300:
                idx_r = np.linspace(0, len(r_arr) - 1, 300, dtype=int)
                r_arr = r_arr[idx_r]
            try:
                from scipy.spatial.distance import cdist
                dist_m = cdist(a_arr[:, :2], r_arr[:, :2])
                nr, nc = dist_m.shape
                cost = np.full((nr, nc), np.inf)
                cost[0, 0] = dist_m[0, 0]
                for row_i in range(1, nr):
                    cost[row_i, 0] = cost[row_i - 1, 0] + dist_m[row_i, 0]
                for col_j in range(1, nc):
                    cost[0, col_j] = cost[0, col_j - 1] + dist_m[0, col_j]
                for row_i in range(1, nr):
                    for col_j in range(1, nc):
                        cost[row_i, col_j] = dist_m[row_i, col_j] + min(
                            cost[row_i - 1, col_j], cost[row_i, col_j - 1], cost[row_i - 1, col_j - 1]
                        )
                errors.append(float(cost[-1, -1] / max(nr, nc)))
            except Exception:
                continue

        return float(np.mean(errors)) if errors else None

    @classmethod
    def reconcile_stage_references(
        cls,
        dynamic_row: dict[str, typing.Any],
        ref_robot_row: dict[str, typing.Any] | None,
        ref_ped_row: dict[str, typing.Any] | None,
    ) -> dict[str, typing.Any]:
        """
        Reconcile dynamic episode metrics against stage reference runs (unobstructed robot and unhindered peds).
        """
        results: dict[str, typing.Any] = {k: None for k in cls.output_keys()}

        actual_ped_paths = dynamic_row.get("pedestrian_path")
        actual_robot_path = dynamic_row.get("path")

        # 1. Pedestrian Flow Interference (PFI) & Deflection
        pfi_val = None
        if actual_ped_paths and ref_ped_row and ref_ped_row.get("pedestrian_path"):
            ref_ped_paths = ref_ped_row.get("pedestrian_path")
            pfi_val = cls.compute_pfi_dtw(actual_ped_paths, ref_ped_paths)
            if pfi_val is not None:
                results["pfi"] = round(float(pfi_val), 4)

            try:
                actual_pts = (
                    np.concatenate([np.array(p) for p in actual_ped_paths if len(p) > 0], axis=0)
                    if isinstance(actual_ped_paths, list)
                    else np.array(actual_ped_paths)
                )
                ref_pts = (
                    np.concatenate([np.array(p) for p in ref_ped_paths if len(p) > 0], axis=0)
                    if isinstance(ref_ped_paths, list)
                    else np.array(ref_ped_paths)
                )
                if len(actual_pts) > 0 and len(ref_pts) > 0 and actual_pts.ndim == 2 and ref_pts.ndim == 2:
                    from .pedestrian_disturbance import PedestrianDisturbanceCalculator
                    deflect = PedestrianDisturbanceCalculator._compute_trajectory_deflection(actual_pts, ref_pts)
                    results["ped_path_deflection_m"] = round(float(deflect), 3)
            except Exception:
                pass

        # 2. Robot Lateral Detour & MAR Computation
        if ref_robot_row:
            ref_robot_path = ref_robot_row.get("path")
            robot_lat_dev = None
            if actual_robot_path is not None and ref_robot_path is not None:
                try:
                    max_d, _ = cls.compute_orthogonal_deviation(actual_robot_path, ref_robot_path)
                    robot_lat_dev = float(max_d)
                    results["robot_path_deviation_m"] = round(robot_lat_dev, 3)
                except Exception:
                    pass

            if pfi_val is not None:
                if robot_lat_dev is not None and robot_lat_dev > 0:
                    results["mar"] = round(float(pfi_val / max(robot_lat_dev, 0.05)), 4)
                elif dynamic_row.get("path_length") is not None and ref_robot_row.get("path_length") is not None:
                    ref_len = float(ref_robot_row["path_length"])
                    robot_dev = max(float(dynamic_row["path_length"]) - ref_len, 0.01)
                    results["mar"] = round(float(pfi_val / robot_dev), 4)

            # 3. Relative Throughput (Speed under congestion vs free run)
            ref_time = ref_robot_row.get("time_to_goal")
            ep_time = dynamic_row.get("time_to_goal")
            if ref_time is not None and ep_time is not None and float(ep_time) > 0:
                results["relative_throughput"] = round(float(ref_time) / float(ep_time), 4)

            # 4. Kinematic Cost of Social Compliance (KCSC Wh)
            ref_energy = ref_robot_row.get("energy_total_wh")
            ep_energy = dynamic_row.get("energy_total_wh")
            if ref_energy is not None and ep_energy is not None:
                results["kcsc_wh"] = round(max(float(ep_energy) - float(ref_energy), 0.0), 4)

        return results

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del episode, prior_results
        return {k: None for k in self.output_keys()}
