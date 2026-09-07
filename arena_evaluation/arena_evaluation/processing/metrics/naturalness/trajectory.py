from __future__ import annotations

import typing
import numpy as np
import polars as pl
from scipy.spatial.distance import cdist

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.path.theta_star import (
    compute_theta_star_for_episode,
    compute_theta_star_path,
)

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


def _discrete_frechet(p: np.ndarray, q: np.ndarray) -> float:
    """Discrete Fréchet Distance via dynamic programming with <=500 point decimation."""
    n, m = len(p), len(q)
    if n == 0 or m == 0:
        return 0.0
    if n > 500:
        idx = np.linspace(0, n - 1, 500, dtype=int)
        p = p[idx]
        n = len(p)
    if m > 500:
        idx = np.linspace(0, m - 1, 500, dtype=int)
        q = q[idx]
        m = len(q)

    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(p[0] - q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(p[i] - q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(p[0] - q[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                np.linalg.norm(p[i] - q[j]),
            )
    return float(ca[n - 1, m - 1])


class TrajectoryMetricsCalculator(BaseMetricCalculator):
    NAME = "trajectory_naturalness"
    CATEGORY = "naturalness"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("tf_gt", "odom")]

    UNITS = {
        "ade": "m",
        "fde": "m",
        "adtw": "m",
        "mhd": "m",
        "frechet_distance": "m",
        "path_irregularity": "rad/m",
        "topological_complexity": "",
        "theta_star_length": "m",
    }

    PRIMARY_OUTPUTS = ["ade", "fde", "mhd", "frechet_distance"]
    OUTPUT_DIRECTIONS = {
        "ade": "lower",
        "fde": "lower",
        "adtw": "lower",
        "mhd": "lower",
        "frechet_distance": "lower",
        "path_irregularity": "lower",
        "topological_complexity": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ade",
            "fde",
            "adtw",
            "mhd",
            "frechet_distance",
            "path_irregularity",
            "topological_complexity",
            "theta_star_length",
        ]

    def calculate(
        self,
        episode: "AlignedEpisodeBundle",
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        del prior_results
        pos_x, pos_y, yaw, _, _, _ = self.resolve_robot_pose(episode)

        if len(pos_x) == 0:
            return {k: None for k in self.output_keys()}

        results: dict[str, typing.Any] = {k: None for k in self.output_keys()}

        # 1. Path Irregularity
        if len(pos_x) > 1:
            dx = np.diff(pos_x)
            dy = np.diff(pos_y)
            distances = np.sqrt(dx**2 + dy**2)
            L = np.sum(distances)

            if L > 0.1 and episode.goal_pos and len(episode.goal_pos) >= 2:
                dyaw = np.diff(yaw)
                dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
                sum_abs_dyaw = np.sum(np.abs(dyaw))

                start_x, start_y = pos_x[0], pos_y[0]
                goal_x, goal_y = episode.goal_pos[0], episode.goal_pos[1]
                target_angle = np.arctan2(goal_y - start_y, goal_x - start_x)

                rot_to_target = (target_angle - yaw[0] + np.pi) % (2 * np.pi) - np.pi
                delta_theta_min = np.abs(rot_to_target)

                pi_val = (sum_abs_dyaw - delta_theta_min) / L
                results["path_irregularity"] = float(pi_val)

        # 2. Topological Complexity (Signed net winding loop W = 1/(2*pi) * sum_j |sum Delta theta_rel_j|)
        total_winding = 0.0
        peds_df = self.native_ped_frame(episode)
        if peds_df is not None and "peds_positions" in peds_df.columns and len(pos_x) > 1:
            try:
                peds_time_ns = peds_df["time_ns"].to_numpy()
                peds_positions = peds_df["peds_positions"].to_list()
                _, _, _, t_odom_base = self.resolve_native_pose(episode)
                rpx, rpy, _ = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom_base)

                ped_series: dict[int, list[tuple[float, float, float, float]]] = {}
                for i in range(len(peds_time_ns)):
                    raw_p = peds_positions[i]
                    if raw_p is None:
                        continue
                    parsed = self._parse_peds(raw_p)
                    for p_idx in range(parsed.shape[0]):
                        if p_idx not in ped_series:
                            ped_series[p_idx] = []
                        ped_series[p_idx].append((float(rpx[i]), float(rpy[i]), float(parsed[p_idx, 0]), float(parsed[p_idx, 1])))

                for p_idx, samples in ped_series.items():
                    if len(samples) > 1:
                        s_arr = np.array(samples)
                        rel_x = s_arr[:, 0] - s_arr[:, 2]
                        rel_y = s_arr[:, 1] - s_arr[:, 3]
                        angles = np.arctan2(rel_y, rel_x)
                        d_angles = np.diff(angles)
                        d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
                        net_ped_winding = np.abs(np.sum(d_angles)) / (2.0 * np.pi)
                        total_winding += float(net_ped_winding)
            except Exception:
                pass
        results["topological_complexity"] = float(total_winding)

        # 3. Reference Path Computation using Geometric Theta* (Synthetic Human Demonstration)
        theta_res = None
        start_pt = (pos_x[0], pos_y[0]) if len(pos_x) > 0 else (episode.start_pos[0], episode.start_pos[1]) if episode.start_pos else None
        goal_pt = (episode.goal_pos[0], episode.goal_pos[1]) if episode.goal_pos and len(episode.goal_pos) >= 2 else None

        if start_pt and goal_pt and episode.map:
            try:
                robot_radius = self.robot_params.robot_radius if self.robot_params else 0.25
                theta_res = compute_theta_star_for_episode(
                    episode.map,
                    start_pt,
                    goal_pt,
                    robot_radius=robot_radius,
                )
            except Exception:
                theta_res = None

        if theta_res is not None and theta_res.success and len(theta_res.path_x) > 0:
            ref_x, ref_y = theta_res.path_x, theta_res.path_y
            results["theta_star_length"] = float(theta_res.geodesic_length)

            # Final Displacement Error (FDE) vs goal or reference end
            if goal_pt:
                fde = np.sqrt((pos_x[-1] - goal_pt[0])**2 + (pos_y[-1] - goal_pt[1])**2)
            else:
                fde = np.sqrt((pos_x[-1] - ref_x[-1])**2 + (pos_y[-1] - ref_y[-1])**2)
            results["fde"] = float(fde)

            # Average Displacement Error (ADE) via arc-length parameterization
            curr_dist = np.insert(np.cumsum(np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2)), 0, 0)
            ref_dist = np.insert(np.cumsum(np.sqrt(np.diff(ref_x)**2 + np.diff(ref_y)**2)), 0, 0)

            if curr_dist[-1] > 0 and ref_dist[-1] > 0:
                curr_frac = curr_dist / curr_dist[-1]
                ref_frac = ref_dist / ref_dist[-1]

                interp_ref_x = np.interp(curr_frac, ref_frac, ref_x)
                interp_ref_y = np.interp(curr_frac, ref_frac, ref_y)

                ade = np.mean(np.sqrt((pos_x - interp_ref_x)**2 + (pos_y - interp_ref_y)**2))
                results["ade"] = float(ade)

            # Modified Hausdorff Distance (MHD) on 100% full-resolution paths via cKDTree (sub-millisecond O(N log M))
            pts_robot = np.column_stack((pos_x, pos_y))
            pts_ref = np.column_stack((ref_x, ref_y))

            try:
                from scipy.spatial import cKDTree
                tree_robot = cKDTree(pts_robot)
                tree_ref = cKDTree(pts_ref)
                d_r2ref, _ = tree_ref.query(pts_robot, k=1)
                d_ref2r, _ = tree_robot.query(pts_ref, k=1)
                mhd_val = max(float(np.mean(d_r2ref)), float(np.mean(d_ref2r)))
            except Exception:
                d_mat = cdist(pts_robot, pts_ref)
                mhd_val = max(np.mean(np.min(d_mat, axis=1)), np.mean(np.min(d_mat, axis=0)))
            results["mhd"] = float(mhd_val)

            # Discrete Fréchet Distance & Asymmetric DTW (ADTW)
            pts_r_sub = pts_robot
            pts_ref_sub = pts_ref
            if len(pts_robot) > 1000:
                idx_r = np.linspace(0, len(pts_robot) - 1, 1000, dtype=int)
                pts_r_sub = pts_robot[idx_r]
            if len(pts_ref) > 1000:
                idx_ref = np.linspace(0, len(pts_ref) - 1, 1000, dtype=int)
                pts_ref_sub = pts_ref[idx_ref]

            frechet_val = _discrete_frechet(pts_r_sub, pts_ref_sub)
            results["frechet_distance"] = float(frechet_val)

            try:
                from dtaidistance import dtw_ndim
                adtw = dtw_ndim.distance(pts_r_sub, pts_ref_sub)
                results["adtw"] = float(adtw)
            except Exception:
                try:
                    d_mat = cdist(pts_r_sub, pts_ref_sub)
                    nr, nc = d_mat.shape
                    dtw_cost = np.full((nr, nc), np.inf)
                    dtw_cost[0, 0] = d_mat[0, 0]
                    for i in range(1, nr):
                        dtw_cost[i, 0] = dtw_cost[i - 1, 0] + d_mat[i, 0]
                    for j in range(1, nc):
                        dtw_cost[0, j] = dtw_cost[0, j - 1] + d_mat[0, j]
                    for i in range(1, nr):
                        for j in range(1, nc):
                            dtw_cost[i, j] = d_mat[i, j] + min(
                                dtw_cost[i - 1, j],
                                dtw_cost[i, j - 1],
                                dtw_cost[i - 1, j - 1],
                            )
                    results["adtw"] = float(dtw_cost[-1, -1] / (nr + nc))
                except Exception:
                    pass
        elif goal_pt:
            # Fallback when map or Theta* solver is unavailable: use Euclidean straight line
            if len(pos_x) > 0:
                results["fde"] = float(np.sqrt((pos_x[-1] - goal_pt[0])**2 + (pos_y[-1] - goal_pt[1])**2))
            eucl_len = float(np.sqrt((goal_pt[0] - start_pt[0])**2 + (goal_pt[1] - start_pt[1])**2)) if start_pt else 0.0
            results["theta_star_length"] = eucl_len

        return results
