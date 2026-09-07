from __future__ import annotations
import typing
import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class ClearanceMetricsCalculator(BaseMetricCalculator):
    """Computes minimum clearance distances to obstacles and pedestrians."""

    NAME = "clearance_metrics"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("scan", "peds")]  # at least one of the two

    UNITS = {
        "min_obstacle_clearance": "m",
        "mean_obstacle_clearance": "m",
        "min_pedestrian_clearance": "m",
        "mean_pedestrian_clearance": "m",
        "clearance_timeseries": "m",
    }

    PRIMARY_OUTPUTS = ["min_obstacle_clearance", "min_pedestrian_clearance"]
    OUTPUT_DIRECTIONS = {
        "min_obstacle_clearance": "higher",
        "mean_obstacle_clearance": "higher",
        "min_pedestrian_clearance": "higher",
        "mean_pedestrian_clearance": "higher",
    }

    _PED_RADIUS = 0.3  # m

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "min_obstacle_clearance",
            "mean_obstacle_clearance",
            "min_pedestrian_clearance",
            "mean_pedestrian_clearance",
            "clearance_timeseries",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None:
            return {k: None for k in self.output_keys()}

        topics = self.native_topics(episode)
        scan_df = topics.get("scan")
        peds_df = topics.get("peds")
        has_scan = scan_df is not None and "scan_min" in scan_df.columns
        has_peds = peds_df is not None and "peds_positions" in peds_df.columns

        if not has_scan and not has_peds:
            return {k: None for k in self.output_keys()}

        pos_x, pos_y, _yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(episode)
        robot_radius = self.robot_params.robot_radius
        N = len(pos_x)
        if N == 0:
            return {k: None for k in self.output_keys()}

        odom_times_ns = episode.data["time_ns"].to_numpy()
        tol = 100_000_000

        scan_min_arr = None
        scan_max_arr = None
        if has_scan:
            s_df = scan_df.sort("time_ns")
            joined = (
                pl.DataFrame({"time_ns": odom_times_ns})
                .join_asof(
                    s_df.select(["time_ns", "scan_min", "scan_range_max"]),
                    on="time_ns",
                    strategy="backward",
                    tolerance=tol,
                )
                .sort("time_ns")
            )
            scan_min_arr = joined["scan_min"].to_numpy()
            scan_max_arr = (
                joined["scan_range_max"].to_numpy()
                if "scan_range_max" in joined.columns
                else None
            )

        peds_positions_list = None
        num_peds_col = None
        if has_peds:
            p_df = peds_df.sort("time_ns")
            cols = ["time_ns", "peds_positions"]
            if "num_pedestrians" in p_df.columns:
                cols.append("num_pedestrians")
            joined = (
                pl.DataFrame({"time_ns": odom_times_ns})
                .join_asof(
                    p_df.select(cols),
                    on="time_ns",
                    strategy="backward",
                    tolerance=tol,
                )
                .sort("time_ns")
            )
            peds_positions_list = joined["peds_positions"].to_list()
            if "num_pedestrians" in joined.columns:
                num_peds_col = joined["num_pedestrians"].to_numpy()

        obstacle_clearances: list[float | None] = []
        ped_clearances: list[float | None] = []
        combined_clearances: list[float | None] = []

        for i in range(N):
            obs_clear = None
            if scan_min_arr is not None and i < len(scan_min_arr):
                sm = scan_min_arr[i]
                if (
                    sm is not None
                    and not np.isnan(sm)
                    and not np.isinf(sm)
                    and sm > 0
                ):
                    r_max = (
                        scan_max_arr[i]
                        if scan_max_arr is not None and i < len(scan_max_arr)
                        else None
                    )
                    # At or beyond max range means nothing was detected
                    if r_max is None or np.isnan(r_max) or sm < float(r_max) - 1e-6:
                        obs_clear = max(0.0, float(sm) - robot_radius)
            obstacle_clearances.append(obs_clear)

            ped_clear = None
            if peds_positions_list is not None and i < len(peds_positions_list):
                peds_raw = peds_positions_list[i]
                peds_arr = self._parse_peds(
                    peds_raw,
                    num_peds_col[i] if num_peds_col is not None else None,
                )
                if peds_arr.shape[0] > 0:
                    rx, ry = pos_x[i], pos_y[i]
                    dx = peds_arr[:, 0] - rx
                    dy = peds_arr[:, 1] - ry
                    dists = np.sqrt(dx**2 + dy**2)
                    min_dist = float(np.min(dists))
                    ped_clear = max(0.0, min_dist - robot_radius - self._PED_RADIUS)
            ped_clearances.append(ped_clear)

            if obs_clear is not None and ped_clear is not None:
                combined_clearances.append(float(min(obs_clear, ped_clear)))
            elif obs_clear is not None:
                combined_clearances.append(float(obs_clear))
            elif ped_clear is not None:
                combined_clearances.append(float(ped_clear))
            else:
                combined_clearances.append(None)

        valid_obs = [c for c in obstacle_clearances if c is not None]
        valid_ped = [c for c in ped_clearances if c is not None]

        return {
            "min_obstacle_clearance": float(np.min(valid_obs)) if valid_obs else None,
            "mean_obstacle_clearance": float(np.mean(valid_obs)) if valid_obs else None,
            "min_pedestrian_clearance": float(np.min(valid_ped)) if valid_ped else None,
            "mean_pedestrian_clearance": float(np.mean(valid_ped)) if valid_ped else None,
            "clearance_timeseries": combined_clearances,
        }
