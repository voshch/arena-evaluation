from __future__ import annotations

import typing

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """Computes pedestrian deflection and slowdown metrics against baseline trajectories."""

    NAME = "pedestrian_disturbance"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["peds"]

    UNITS = {
        "ped_path_deflection_m": "m",
        "ped_velocity_delay_ratio": "",
        "ped_round_trips_completed": "",
    }

    PRIMARY_OUTPUTS = ["ped_path_deflection_m"]

    OUTPUT_DIRECTIONS = {"ped_path_deflection_m": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_path_deflection_m",
            "ped_velocity_delay_ratio",
            "ped_round_trips_completed",
        ]

    @staticmethod
    def _compute_trajectory_deflection(actual_coords: np.ndarray, reference_coords: np.ndarray) -> float:
        """
        Compute mean distance between actual trajectory points and nearest reference trajectory points.
        actual_coords: (N, 2) or (N, 3)
        reference_coords: (M, 2) or (M, 3)
        """
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return 0.0

        valid_act = actual_coords[~np.isnan(actual_coords).any(axis=1)]
        valid_ref = reference_coords[~np.isnan(reference_coords).any(axis=1)]
        if len(valid_act) == 0 or len(valid_ref) == 0:
            return 0.0

        valid_act = np.ascontiguousarray(valid_act[:, :2], dtype=np.float64)
        valid_ref = np.ascontiguousarray(valid_ref[:, :2], dtype=np.float64)

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(valid_ref)
            dists, _ = tree.query(valid_act, k=1)
            return float(np.mean(dists))
        except Exception:
            from scipy.spatial.distance import cdist

            dists = cdist(valid_act, valid_ref)
            return float(np.mean(np.min(dists, axis=1)))

    @staticmethod
    def _count_round_trips(dists_from_start: np.ndarray, max_disp: float) -> int:
        """Counts returns to within a quarter of the farthest reach, each after passing three quarters of it."""
        trips = 0
        away = False
        for dist in dists_from_start:
            if not away and dist >= 0.75 * max_disp:
                away = True
            elif away and dist <= 0.25 * max_disp:
                trips += 1
                away = False
        return trips

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}

        peds_pos = episode.data["peds_positions"].to_list()
        if not peds_pos or len(peds_pos) == 0:
            return {k: None for k in self.output_keys()}

        # Extract pedestrian positions per agent
        agent_paths: dict[int, list[tuple[float, float]]] = {}
        for row in peds_pos:
            arr = self._parse_peds(row)
            for agent_idx, item in enumerate(arr):
                if not (np.isfinite(item[0]) and np.isfinite(item[1]) and abs(item[0]) < 1e6 and abs(item[1]) < 1e6):
                    continue
                agent_paths.setdefault(agent_idx, []).append((float(item[0]), float(item[1])))

        if not agent_paths:
            return {
                "ped_path_deflection_m": 0.0,
                "ped_velocity_delay_ratio": 0.0,
                "ped_round_trips_completed": 0,
            }

        round_trips = 0
        total_deflection = 0.0
        num_agents = len(agent_paths)

        for coords in agent_paths.values():
            arr = np.array(coords)
            if len(arr) < 5:
                continue

            diffs = arr - arr[0]
            dists_from_start = np.hypot(diffs[:, 0], diffs[:, 1])
            max_disp = float(np.max(dists_from_start))

            if max_disp > 0.5:
                round_trips += self._count_round_trips(dists_from_start, max_disp)

            # Standalone geometric lateral deflection against straight-line start-to-end vector
            p_start, p_end = arr[0], arr[-1]
            seg_vec = p_end - p_start
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            if np.isfinite(seg_len) and seg_len > 0.5:
                u_vec = seg_vec / seg_len
                # Cross-product magnitude gives perpendicular distance in 2D
                rel_pos = arr - p_start
                lateral_dists = np.abs(rel_pos[:, 0] * u_vec[1] - rel_pos[:, 1] * u_vec[0])
                finite_lat = lateral_dists[np.isfinite(lateral_dists)]
                total_deflection += float(np.mean(finite_lat))

        # Velocity delay: compare actual mean speed to baseline desired speed (~1.2 m/s)
        time_ns = episode.data["time_ns"].to_numpy() if "time_ns" in episode.data.columns else None
        avg_speed = 0.0
        if time_ns is not None and len(time_ns) > 1:
            dt_s = (time_ns[-1] - time_ns[0]) / 1e9
            if dt_s > 0:
                all_dists = []
                for pts in agent_paths.values():
                    if len(pts) > 1:
                        p_arr = np.array(pts)
                        step_dists = np.hypot(np.diff(p_arr[:, 0]), np.diff(p_arr[:, 1]))
                        finite_d = step_dists[np.isfinite(step_dists)]
                        all_dists.append(float(np.sum(finite_d)))
                if all_dists:
                    avg_speed = float(np.mean(all_dists)) / dt_s

        desired_speed = 1.2  # Nominal pedsim desired speed
        velocity_delay_ratio = max(0.0, float(1.0 - (avg_speed / desired_speed))) if avg_speed > 0 and np.isfinite(avg_speed) else 0.0

        deflection_val = round(total_deflection / num_agents, 3) if num_agents > 0 and np.isfinite(total_deflection) else 0.0
        delay_val = round(velocity_delay_ratio, 3) if np.isfinite(velocity_delay_ratio) else 0.0
        trips_val = round_trips / num_agents if num_agents > 0 and np.isfinite(round_trips) else 0

        return {
            "ped_path_deflection_m": deflection_val,
            "ped_velocity_delay_ratio": delay_val,
            "ped_round_trips_completed": trips_val,
        }
