from __future__ import annotations

import ast
import typing
import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class SocialForcesCalculator(BaseMetricCalculator):
    """Computes social force and personal space violation metrics."""

    NAME = "social_forces"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("tf_gt", "odom"), "peds"]

    UNITS = {
        "sfm_cumulative_force": "N·s",
        "sfm_peak_force": "N",
        "sfm_mean_force": "N",
        "esfm_cumulative_force": "N·s",
        "esfm_peak_force": "N",
        "esfm_mean_force": "N",
        "esfm_ped_cumulative_force": "N·s",
        "esfm_ped_peak_force": "N",
        "esfm_ped_mean_force": "N",
        "social_force_max": "N",
        "social_force_mean": "N",
        "social_force_ped_max": "N",
        "social_force_ped_mean": "N",
        "ci_max": "",
        "ci_mean": "",
        "timeseries_sfm_force": "N",
        "timeseries_ci": "",
    }

    PRIMARY_OUTPUTS = ["sfm_mean_force", "ci_mean", "social_force_ped_max"]
    OUTPUT_DIRECTIONS = {
        "sfm_mean_force": "lower",
        "ci_mean": "lower",
        "social_force_ped_max": "lower",
        "social_force_ped_mean": "lower",
    }

    # SFM constants (Helbing & Molnar)
    _A = 2.1       # interaction strength (N)
    _B = 0.3       # interaction range (m)
    _PED_RADIUS = 0.3  # m
    _CUTOFF = 5.0  # m, ignore pedestrians beyond this radius
    _MAX_FORCE = 100.0  # N, clamp so near-zero d_ij cannot overflow
    _LAMBDA_ESFM = 0.5  # anisotropy, 0.5 = moderate rear de-weighting

    # CI / SII personal-space sigmas
    _SIGMA_FRONT = 1.0  # m, along pedestrian heading
    _SIGMA_SIDE = 0.5   # m, perpendicular to heading
    _SIGMA_PX0 = 0.28  # m
    _SIGMA_PY0 = 0.28  # m

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "sfm_cumulative_force",
            "sfm_peak_force",
            "sfm_mean_force",
            "esfm_cumulative_force",
            "esfm_peak_force",
            "esfm_mean_force",
            "esfm_ped_cumulative_force",
            "esfm_ped_peak_force",
            "esfm_ped_mean_force",
            "social_force_max",
            "social_force_mean",
            "social_force_ped_max",
            "social_force_ped_mean",
            "ci_max",
            "ci_mean",
            "timeseries_sfm_force",
            "timeseries_ci",
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
        peds_headings_list = (
            peds_df["peds_headings"].to_list()
            if "peds_headings" in peds_df.columns
            else None
        )

        N = len(peds_time_ns)
        if N == 0:
            return {k: None for k in self.output_keys()}

        # Robot pose sampled onto the peds time axis (backward-asof, 100 ms)
        rpx, rpy, ryaw = self.pose_at_times(peds_time_ns, pos_x, pos_y, yaw, t_odom)

        dt = np.diff(peds_time_ns) / 1e9
        dt = np.append(dt, 0.0)
        dt = np.where(dt == 0.0, 1e-6, dt)

        d_combined = self.robot_params.robot_radius + self._PED_RADIUS
        sx2 = 2.0 * self._SIGMA_PX0 ** 2
        sy2 = 2.0 * self._SIGMA_PY0 ** 2

        sfm_forces = []
        esfm_forces = []
        esfm_ped_forces = []
        ci_values = []

        for i in range(N):
            rx, ry = float(rpx[i]), float(rpy[i])
            peds_raw = peds_positions[i]
            peds_arr = self._parse_peds(
                peds_raw,
                num_peds_col[i] if num_peds_col is not None else None,
            )

            if peds_arr.shape[0] == 0:
                sfm_forces.append(0.0)
                esfm_forces.append(0.0)
                esfm_ped_forces.append(0.0)
                ci_values.append(0.0)
                continue

            # Missing headings fall back to isotropic
            headings = None
            if peds_headings_list is not None and i < len(peds_headings_list):
                h_raw = peds_headings_list[i]
                if h_raw and len(h_raw) > 0:
                    if isinstance(h_raw, str):
                        try:
                            h_raw = ast.literal_eval(h_raw)
                        except Exception:
                            h_raw = []
                    headings = np.array(h_raw, dtype=np.float64) if len(h_raw) > 0 else None

            robot_yaw_i = float(ryaw[i])
            total_sfm = 0.0
            total_esfm = 0.0
            total_esfm_ped = 0.0
            max_ci = 0.0

            for j in range(peds_arr.shape[0]):
                px, py = peds_arr[j, 0], peds_arr[j, 1]
                dx = px - rx
                dy = py - ry
                d_ij = np.sqrt(dx**2 + dy**2)

                if headings is not None and j < len(headings):
                    ped_heading = float(headings[j])
                    cos_h = np.cos(ped_heading)
                    sin_h = np.sin(ped_heading)
                    d_front = dx * cos_h + dy * sin_h
                    d_side = -dx * sin_h + dy * cos_h
                    ci_val = np.exp(-(d_front**2 / (2 * self._SIGMA_FRONT**2) + d_side**2 / (2 * self._SIGMA_SIDE**2)))
                else:
                    ci_val = np.exp(-((dx**2) / sx2 + (dy**2) / sy2))

                max_ci = max(max_ci, float(ci_val))

                if d_ij > self._CUTOFF:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    force_mag = self._A * np.exp((d_combined - d_ij) / self._B)
                force_mag = min(float(force_mag), self._MAX_FORCE)
                if np.isnan(force_mag) or np.isinf(force_mag):
                    force_mag = 0.0
                total_sfm += force_mag

                # phi = angle between robot facing and direction to the ped
                phi = robot_yaw_i - np.arctan2(py - ry, px - rx)
                w_robot = self._LAMBDA_ESFM + (1.0 - self._LAMBDA_ESFM) * (1.0 + np.cos(phi)) / 2.0
                w_robot = max(0.0, min(1.0, w_robot))
                total_esfm += force_mag * w_robot

                # Ped-heading anisotropy: the force the robot exerts
                w_ped = 1.0
                if headings is not None and j < len(headings):
                    ped_heading = float(headings[j])
                    phi_ped = ped_heading - np.arctan2(ry - py, rx - px)
                    w_ped = self._LAMBDA_ESFM + (1.0 - self._LAMBDA_ESFM) * (1.0 + np.cos(phi_ped)) / 2.0
                    w_ped = max(0.0, min(1.0, w_ped))
                total_esfm_ped += force_mag * w_ped

            sfm_forces.append(float(total_sfm))
            esfm_forces.append(float(total_esfm))
            esfm_ped_forces.append(float(total_esfm_ped))
            ci_values.append(float(max_ci))

        sfm_arr = np.array(sfm_forces)
        esfm_arr = np.array(esfm_forces)
        esfm_ped_arr = np.array(esfm_ped_forces)
        ci_arr = np.array(ci_values)

        with np.errstate(divide="ignore", invalid="ignore"):
            sfm_cumulative = float(np.sum(sfm_arr * dt))
            esfm_cumulative = float(np.sum(esfm_arr * dt))
            esfm_ped_cumulative = float(np.sum(esfm_ped_arr * dt))

        return {
            "sfm_cumulative_force": sfm_cumulative,
            "sfm_peak_force": float(np.max(sfm_arr)),
            "sfm_mean_force": float(np.mean(sfm_arr)),
            "esfm_cumulative_force": esfm_cumulative,
            "esfm_peak_force": float(np.max(esfm_arr)),
            "esfm_mean_force": float(np.mean(esfm_arr)),
            "esfm_ped_cumulative_force": esfm_ped_cumulative,
            "esfm_ped_peak_force": float(np.max(esfm_ped_arr)),
            "esfm_ped_mean_force": float(np.mean(esfm_ped_arr)),
            "social_force_max": float(np.max(sfm_arr)),
            "social_force_mean": float(np.mean(sfm_arr)),
            "social_force_ped_max": float(np.max(esfm_ped_arr)),
            "social_force_ped_mean": float(np.mean(esfm_ped_arr)),
            "ci_max": float(np.max(ci_arr)),
            "ci_mean": float(np.mean(ci_arr)),
            "timeseries_sfm_force": sfm_forces,
            "timeseries_ci": ci_values,
        }
