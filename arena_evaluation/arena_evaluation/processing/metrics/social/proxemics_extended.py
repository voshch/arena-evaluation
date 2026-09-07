from __future__ import annotations

import ast
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class ProxemicsExtendedCalculator(BaseMetricCalculator):
    """Computes edge-to-edge proxemic zone metrics on the native pedestrian time base."""

    NAME = "proxemics_extended"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["motion_metrics", "path_metrics"]
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    # Hall's proxemic zones (meters), edge-to-edge
    _INTIMATE_R = 0.45
    _PERSONAL_R = 1.2
    _SOCIAL_R = 3.6

    _PED_RADIUS = 0.3  # m
    _TTI_LOOKAHEAD = 5.0  # s
    _EVENT_GAP_S = 2.0  # s, min gap between distinct interactions

    UNITS = {
        "time_in_intimate_zone": "s",
        "time_in_personal_zone": "s",
        "time_in_social_zone": "s",
        "time_in_public_zone": "s",
        "psi_intimate_events": "",
        "psi_personal_events": "",
        "psi_social_events": "",
        "max_speed_intimate_zone": "m/s",
        "max_speed_personal_zone": "m/s",
        "max_speed_social_zone": "m/s",
        "max_speed_public_zone": "m/s",
        "movement_towards_peds_ratio": "",
        "tti_min": "s",
        "tti_mean": "s",
        "personal_space_intrusion_integral": "m·s",
        "timeseries_min_ped_clearance": "m",
    }

    PRIMARY_OUTPUTS = [
        "time_in_intimate_zone",
        "psi_intimate_events",
        "tti_min",
    ]
    OUTPUT_DIRECTIONS = {
        "time_in_intimate_zone": "lower",
        "time_in_personal_zone": "lower",
        "max_speed_intimate_zone": "lower",
        "max_speed_personal_zone": "lower",
        "tti_min": "higher",
        "personal_space_intrusion_integral": "lower",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_in_intimate_zone",
            "time_in_personal_zone",
            "time_in_social_zone",
            "time_in_public_zone",
            "psi_intimate_events",
            "psi_personal_events",
            "psi_social_events",
            "max_speed_intimate_zone",
            "max_speed_personal_zone",
            "max_speed_social_zone",
            "max_speed_public_zone",
            "movement_towards_peds_ratio",
            "tti_min",
            "tti_mean",
            "personal_space_intrusion_integral",
            "timeseries_min_ped_clearance",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        peds_df = self.native_ped_frame(episode)
        if peds_df is None or "peds_positions" not in peds_df.columns:
            return {k: None for k in self.output_keys()}
        pos_x, pos_y, yaw, t_odom = self.resolve_native_pose(episode)
        if len(pos_x) == 0:
            return {k: None for k in self.output_keys()}

        peds_time_ns = peds_df["time_ns"].to_numpy()
        peds_positions = peds_df["peds_positions"].to_list()
        num_peds_col = (
            peds_df["num_pedestrians"].to_numpy()
            if "num_pedestrians" in peds_df.columns
            else None
        )
        peds_twists_list = (
            peds_df["peds_twists"].to_list()
            if "peds_twists" in peds_df.columns
            else None
        )

        N = len(peds_time_ns)
        if N == 0:
            return {k: None for k in self.output_keys()}

        rpx, rpy, _ = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)
        vx_full, vy_full = self.velocity_from_pose(pos_x, pos_y, t_odom)
        speed_full = self.speed_from_pose(pos_x, pos_y, t_odom)
        rvx = self.values_at_times(vx_full, t_odom, peds_time_ns)
        rvy = self.values_at_times(vy_full, t_odom, peds_time_ns)
        rspeed = self.values_at_times(speed_full, t_odom, peds_time_ns)
        rvx = np.nan_to_num(rvx, nan=0.0)
        rvy = np.nan_to_num(rvy, nan=0.0)
        rspeed = np.nan_to_num(rspeed, nan=0.0)

        dt = np.diff(peds_time_ns) / 1e9
        dt = np.append(dt, 0.0)
        dt = np.where(dt == 0.0, 1e-6, dt)

        robot_radius = self.robot_params.robot_radius
        d_combined = robot_radius + self._PED_RADIUS

        time_zone = {"intimate": 0.0, "personal": 0.0, "social": 0.0, "public": 0.0}
        max_speed_zone = {"intimate": 0.0, "personal": 0.0, "social": 0.0, "public": 0.0}
        zone_mask = {"intimate": [], "personal": [], "social": []}
        approaching_count = 0
        peds_frames = 0
        psii_sum = 0.0
        all_tti = []
        min_clearances: list[float | None] = []

        def band(d_eff: float) -> str:
            if d_eff < self._INTIMATE_R:
                return "intimate"
            if d_eff < self._PERSONAL_R:
                return "personal"
            if d_eff < self._SOCIAL_R:
                return "social"
            return "public"

        for i in range(N):
            rx, ry = float(rpx[i]), float(rpy[i])
            peds_arr = self._parse_peds(
                peds_positions[i],
                num_peds_col[i] if num_peds_col is not None else None,
            )

            if peds_arr.shape[0] == 0:
                min_clearances.append(None)
                zone_mask["intimate"].append(False)
                zone_mask["personal"].append(False)
                zone_mask["social"].append(False)
                continue

            peds_frames += 1
            dx_all = peds_arr[:, 0] - rx
            dy_all = peds_arr[:, 1] - ry
            dists = np.sqrt(dx_all**2 + dy_all**2)
            min_dist = float(np.min(dists))
            d_eff = min_dist - d_combined
            min_clearances.append(float(d_eff))

            z = band(d_eff)
            time_zone[z] += dt[i]
            zone_mask["intimate"].append(d_eff < self._INTIMATE_R)
            zone_mask["personal"].append(self._INTIMATE_R <= d_eff < self._PERSONAL_R)
            zone_mask["social"].append(self._PERSONAL_R <= d_eff < self._SOCIAL_R)
            max_speed_zone[z] = max(max_speed_zone[z], float(rspeed[i]))

            min_idx = int(np.argmin(dists))
            r_to_ped = np.array(
                [peds_arr[min_idx, 0] - rx, peds_arr[min_idx, 1] - ry]
            )
            v_robot = np.array([rvx[i], rvy[i]])
            if np.dot(v_robot, r_to_ped) > 0:
                approaching_count += 1

            if d_eff < self._PERSONAL_R:
                psii_sum += (self._PERSONAL_R - max(d_eff, 0.0)) * dt[i]

            ped_vels = None
            if peds_twists_list is not None and i < len(peds_twists_list):
                tw_raw = peds_twists_list[i]
                if tw_raw and len(tw_raw) > 0:
                    if isinstance(tw_raw, str):
                        try:
                            tw_raw = ast.literal_eval(tw_raw)
                        except Exception:
                            tw_raw = []
                    tw_arr = np.array(tw_raw, dtype=np.float64)
                    if tw_arr.size > 0 and len(tw_arr) % 3 == 0:
                        ped_vels = tw_arr.reshape(-1, 3)

            for j in range(peds_arr.shape[0]):
                p_rel = np.array([peds_arr[j, 0] - rx, peds_arr[j, 1] - ry])
                d_ij = dists[j]
                v_ped = np.array([0.0, 0.0])
                if ped_vels is not None and j < ped_vels.shape[0]:
                    v_ped = ped_vels[j, :2]
                v_rel = v_ped - v_robot
                rel_speed = float(np.linalg.norm(v_rel))
                if rel_speed > 0.05 and np.dot(p_rel, v_rel) < 0:
                    d_eff_j = d_ij - d_combined
                    if d_eff_j > 0:
                        tti_val = d_eff_j / rel_speed
                        if 0 < tti_val < self._TTI_LOOKAHEAD:
                            all_tti.append(float(tti_val))

        def count_events(mask: list[bool]) -> int:
            if not any(mask):
                return 0
            m = np.array(mask, dtype=bool)
            padded = np.concatenate([[False], m, [False]])
            edges = np.diff(padded.astype(np.int8))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            gap_ns = int(self._EVENT_GAP_S * 1e9)
            events = 0
            prev_end_ns = 0
            for s, e in zip(starts, ends):
                start_ns = int(peds_time_ns[s]) if s < len(peds_time_ns) else 0
                # Blocks closer together than the gap belong to one interaction.
                if events == 0 or start_ns - prev_end_ns >= gap_ns:
                    events += 1
                if e > 0 and e <= len(peds_time_ns):
                    prev_end_ns = int(peds_time_ns[e - 1])
            return events

        if all_tti:
            tti_min = float(np.min(all_tti))
            tti_mean = float(np.mean(all_tti))
        else:
            tti_min = None
            tti_mean = None

        return {
            "time_in_intimate_zone": float(time_zone["intimate"]),
            "time_in_personal_zone": float(time_zone["personal"]),
            "time_in_social_zone": float(time_zone["social"]),
            "time_in_public_zone": float(time_zone["public"]),
            "psi_intimate_events": count_events(zone_mask["intimate"]),
            "psi_personal_events": count_events(zone_mask["personal"]),
            "psi_social_events": count_events(zone_mask["social"]),
            "max_speed_intimate_zone": float(max_speed_zone["intimate"]),
            "max_speed_personal_zone": float(max_speed_zone["personal"]),
            "max_speed_social_zone": float(max_speed_zone["social"]),
            "max_speed_public_zone": float(max_speed_zone["public"]),
            "movement_towards_peds_ratio": (
                float(approaching_count / peds_frames) if peds_frames > 0 else None
            ),
            "tti_min": tti_min,
            "tti_mean": tti_mean,
            "personal_space_intrusion_integral": float(psii_sum),
            "timeseries_min_ped_clearance": min_clearances,
        }
