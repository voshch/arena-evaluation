from __future__ import annotations

import ast
import typing
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

from ..pose_segments import teleport_jumps

if typing.TYPE_CHECKING:
    from ...storage.schemas import AlignedEpisodeBundle, RobotParams


class BaseMetricCalculator(ABC):
    """Abstract Base Class for all metric calculators."""

    NAME: str = ""
    CATEGORY: str = "general"
    REQUIRES_PEDSIM: bool = False
    DEPENDS_ON: list[str] = []
    REQUIRED_TOPICS: list[str | list[str] | tuple[str, ...] | set[str]] = []
    UNITS: dict[str, str] = {}
    PRIMARY_OUTPUTS: list[str] = []
    OUTPUT_DIRECTIONS: dict[str, str] = {}

    def __init__(self, robot_params: RobotParams):
        """Initializes the calculator with robot parameters."""
        self.robot_params = robot_params
        self.metrics_config: dict[str, int] = {}

    def resolve_robot_pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map-frame robot pose (pos_x, pos_y, yaw); the start-pose re-anchor is the odom-only fallback."""
        if episode.data is not None and len(episode.data) > 0:
            if "pos_x_gt" in episode.data.columns:
                episode.data = episode.data.filter(pl.col("pos_x_gt").is_not_null() & ~pl.col("pos_x_gt").is_nan() & pl.col("pos_y_gt").is_not_null() & ~pl.col("pos_y_gt").is_nan() & pl.col("yaw_gt").is_not_null() & ~pl.col("yaw_gt").is_nan())
            elif "pos_x" in episode.data.columns:
                episode.data = episode.data.filter(pl.col("pos_x").is_not_null() & ~pl.col("pos_x").is_nan() & pl.col("pos_y").is_not_null() & ~pl.col("pos_y").is_nan() & pl.col("yaw").is_not_null() & ~pl.col("yaw").is_nan())

        if episode.data is not None and len(episode.data) > 0:
            if "pos_x_gt" in episode.data.columns:
                odom_x = episode.data["pos_x_gt"].to_numpy().copy()
                odom_y = episode.data["pos_y_gt"].to_numpy().copy()
                odom_yaw = episode.data["yaw_gt"].to_numpy().copy()
            elif "pos_x" in episode.data.columns:
                odom_x = episode.data["pos_x"].to_numpy().copy()
                odom_y = episode.data["pos_y"].to_numpy().copy()
                odom_yaw = episode.data["yaw"].to_numpy().copy()
            else:
                odom_x = np.array([], dtype=np.float64)
                odom_y = np.array([], dtype=np.float64)
                odom_yaw = np.array([], dtype=np.float64)

            raw_odom_x0 = odom_x[0] if len(odom_x) > 0 else 0.0
            raw_odom_y0 = odom_y[0] if len(odom_y) > 0 else 0.0
            raw_odom_yaw0 = odom_yaw[0] if len(odom_yaw) > 0 else 0.0
            if len(odom_x) > 1:
                stamp_col = "stamp_ns_gt" if "pos_x_gt" in episode.data.columns else "stamp_ns"
                jumps = teleport_jumps(odom_x, odom_y, episode.data[stamp_col].to_numpy() if stamp_col in episode.data.columns else None)

                if len(jumps) > 0:
                    split_indices = jumps + 1
                    segments_x = np.split(odom_x, split_indices)
                    segments_y = np.split(odom_y, split_indices)

                    best_seg_idx = -1
                    best_len = -1.0

                    for i in range(len(segments_x)):
                        seg_x = segments_x[i]
                        seg_y = segments_y[i]
                        if len(seg_x) < 2:
                            seg_len = 0.0
                        else:
                            seg_len = np.sum(np.sqrt(np.diff(seg_x) ** 2 + np.diff(seg_y) ** 2))

                        if seg_len >= 0.2 and seg_len > best_len:
                            best_len = seg_len
                            best_seg_idx = i

                    if best_seg_idx != -1:
                        start_idx = 0 if best_seg_idx == 0 else int(split_indices[best_seg_idx - 1])
                        end_idx = int(split_indices[best_seg_idx]) if best_seg_idx < len(split_indices) - 1 else len(odom_x)

                        odom_x = odom_x[start_idx:end_idx]
                        odom_y = odom_y[start_idx:end_idx]
                        odom_yaw = odom_yaw[start_idx:end_idx]
                        episode.data = episode.data.slice(start_idx, end_idx - start_idx)

        else:
            odom_x = np.array([], dtype=np.float64)
            odom_y = np.array([], dtype=np.float64)
            odom_yaw = np.array([], dtype=np.float64)

        if episode.start_pos and len(episode.start_pos) >= 2 and len(odom_x) > 0:
            start_x, start_y = episode.start_pos[0], episode.start_pos[1]
            start_yaw = episode.start_pos[2] if len(episode.start_pos) >= 3 else 0.0

            odom_x0 = raw_odom_x0
            odom_y0 = raw_odom_y0
            odom_yaw0 = raw_odom_yaw0

            theta = start_yaw - odom_yaw0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            dx = odom_x - odom_x0
            dy = odom_y - odom_y0

            odom_x_trans = start_x + dx * cos_t - dy * sin_t
            odom_y_trans = start_y + dx * sin_t + dy * cos_t
            odom_yaw_trans = odom_yaw + theta
            odom_yaw_trans = (odom_yaw_trans + np.pi) % (2 * np.pi) - np.pi
        else:
            odom_x_trans, odom_y_trans, odom_yaw_trans = odom_x, odom_y, odom_yaw

        use_gt = episode.data is not None and "pos_x_gt" in episode.data.columns
        if use_gt:
            pos_x = episode.data["pos_x_gt"].to_numpy().copy()
            pos_y = episode.data["pos_y_gt"].to_numpy().copy()
            yaw = episode.data["yaw_gt"].to_numpy().copy()
        else:
            pos_x = odom_x_trans
            pos_y = odom_y_trans
            yaw = odom_yaw_trans

        return pos_x, pos_y, yaw, odom_x_trans, odom_y_trans, odom_yaw_trans

    def native_topics(self, episode: AlignedEpisodeBundle) -> dict:
        """Raw native-rate topic frames keyed by topic name; {} when absent."""
        return episode.topics if isinstance(episode.topics, dict) else {}

    def resolve_native_pose(self, episode: AlignedEpisodeBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth-first robot pose on the native odom time axis.

        Returns (pos_x, pos_y, yaw, time_ns) float64 arrays.
        tf_gt is the trusted source when recorded (world frame,
        offset-corrected at extraction). Otherwise raw odom is transformed
        from the robot-local origin via the episode start pose, the same
        transform ``resolve_robot_pose`` applies.
        """
        topics = self.native_topics(episode)
        odom = topics.get("odom")
        tf_gt = topics.get("tf_gt")

        if tf_gt is not None and "pos_x_gt" in tf_gt.columns:
            t_df = tf_gt.drop_nulls(["pos_x_gt", "pos_y_gt", "yaw_gt"]).sort("time_ns")
            if len(t_df) > 0:
                return (
                    t_df["pos_x_gt"].to_numpy().astype(np.float64),
                    t_df["pos_y_gt"].to_numpy().astype(np.float64),
                    t_df["yaw_gt"].to_numpy().astype(np.float64),
                    t_df["time_ns"].to_numpy().astype(np.int64),
                )

        if odom is None or "pos_x" not in odom.columns:
            return np.array([]), np.array([]), np.array([]), np.array([])

        o_df = odom.drop_nulls(["pos_x", "pos_y", "yaw"]).sort("time_ns")
        if len(o_df) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ox = o_df["pos_x"].to_numpy().astype(np.float64)
        oy = o_df["pos_y"].to_numpy().astype(np.float64)
        oyaw = o_df["yaw"].to_numpy().astype(np.float64)
        time_ns = o_df["time_ns"].to_numpy().astype(np.int64)

        if episode.start_pos and len(episode.start_pos) >= 2:
            start_x, start_y = float(episode.start_pos[0]), float(episode.start_pos[1])
            start_yaw = float(episode.start_pos[2]) if len(episode.start_pos) >= 3 else 0.0
            theta = start_yaw - float(oyaw[0])
            c, s = np.cos(theta), np.sin(theta)
            dx = ox - ox[0]
            dy = oy - oy[0]
            ox = start_x + dx * c - dy * s
            oy = start_y + dx * s + dy * c
            oyaw = (oyaw + theta + np.pi) % (2 * np.pi) - np.pi

        return ox, oy, oyaw, time_ns

    def native_ped_frame(self, episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """Raw pedestrian frame (native rate) sorted by time_ns, or None."""
        topics = self.native_topics(episode)
        peds = topics.get("peds")
        if peds is None:
            return None
        if "time_ns" not in peds.columns:
            return None
        return peds.sort("time_ns")

    def pose_at_times(
        self,
        times_ns: np.ndarray,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        yaw: np.ndarray,
        time_ns: np.ndarray,
        tolerance_ns: int | None = 100_000_000,
        continuous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate pose arrays onto query timestamps ``times_ns``."""
        if len(time_ns) == 0 or len(times_ns) == 0:
            return np.zeros(len(times_ns)), np.zeros(len(times_ns)), np.zeros(len(times_ns))

        order = np.argsort(time_ns)
        t_sorted = time_ns[order]
        x_sorted = np.asarray(pos_x, dtype=np.float64)[order]
        y_sorted = np.asarray(pos_y, dtype=np.float64)[order]
        yaw_sorted = np.asarray(yaw, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            px = np.interp(times_ns, t_sorted, x_sorted)
            py = np.interp(times_ns, t_sorted, y_sorted)
            ya = np.interp(times_ns, t_sorted, yaw_sorted)
            return px, py, ya

        import polars as pl

        df_ref = pl.DataFrame(
            {
                "time_ns": t_sorted,
                "pos_x": x_sorted,
                "pos_y": y_sorted,
                "yaw": yaw_sorted,
            }
        )
        df_query = pl.DataFrame({"time_ns": times_ns})
        merged = df_query.join_asof(df_ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
        px = np.nan_to_num(merged["pos_x"].to_numpy(), nan=0.0)
        py = np.nan_to_num(merged["pos_y"].to_numpy(), nan=0.0)
        ya = np.nan_to_num(merged["yaw"].to_numpy(), nan=0.0)
        return px, py, ya

    def values_at_times(
        self,
        values: np.ndarray,
        values_time_ns: np.ndarray,
        query_times_ns: np.ndarray,
        tolerance_ns: int | None = 100_000_000,
        continuous: bool = False,
    ) -> np.ndarray:
        """Interpolate values onto query timestamps."""
        if len(values_time_ns) == 0 or len(query_times_ns) == 0:
            return np.zeros(len(query_times_ns))

        order = np.argsort(values_time_ns)
        t_sorted = values_time_ns[order]
        v_sorted = np.asarray(values, dtype=np.float64)[order]

        if continuous or tolerance_ns is None:
            return np.interp(query_times_ns, t_sorted, v_sorted)

        import polars as pl

        df_ref = pl.DataFrame(
            {
                "time_ns": t_sorted,
                "val": v_sorted,
            }
        )
        df_query = pl.DataFrame({"time_ns": query_times_ns})
        merged = df_query.join_asof(df_ref, on="time_ns", strategy="backward", tolerance=tolerance_ns)
        return merged["val"].to_numpy()

    @staticmethod
    def speed_from_pose(pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
        """GT speed (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        speed = np.zeros(n)
        if n < 2:
            return speed
        dt = np.diff(time_ns) / 1e9
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        speed[1:] = np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2) / dt_safe
        return np.nan_to_num(speed, nan=0.0)

    @staticmethod
    def velocity_from_pose(pos_x: np.ndarray, pos_y: np.ndarray, time_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GT velocity vector (m/s) by finite-differencing GT positions."""
        n = len(pos_x)
        vx = np.zeros(n)
        vy = np.zeros(n)
        if n < 2:
            return vx, vy
        dt = np.diff(time_ns) / 1e9
        dt_safe = np.where(dt <= 0.0, 1e-6, dt)
        vx[1:] = np.diff(pos_x) / dt_safe
        vy[1:] = np.diff(pos_y) / dt_safe
        return np.nan_to_num(vx, nan=0.0), np.nan_to_num(vy, nan=0.0)

    @staticmethod
    def _parse_peds(peds_raw: object, num_peds_hint: float | None = None) -> np.ndarray:
        """Parse pedestrian positions into an (N, 2) or (N, 3) array of coordinates."""
        if peds_raw is None or len(peds_raw) == 0:
            return np.empty((0, 2))
        if isinstance(peds_raw, str):
            try:
                peds_raw = ast.literal_eval(peds_raw)
            except (ValueError, SyntaxError):
                return np.empty((0, 2))

        if isinstance(peds_raw, (list, tuple, np.ndarray)):
            if len(peds_raw) == 0:
                return np.empty((0, 2))
            first = peds_raw[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                pts = []
                for p in peds_raw:
                    if len(p) >= 2:
                        try:
                            pts.append([float(v) for v in p])
                        except (ValueError, TypeError):
                            pass
                if pts:
                    return np.array(pts, dtype=np.float64)
                return np.empty((0, 2))

            try:
                flat = [float(v) for v in peds_raw if v is not None]
            except (ValueError, TypeError):
                return np.empty((0, 2))

            if len(flat) == 0:
                return np.empty((0, 2))

            # The hint arrives as a float column after the asof join, NaN where no sample matched.
            hint = int(num_peds_hint) if num_peds_hint is not None and num_peds_hint == num_peds_hint else 0
            if hint > 0:
                if len(flat) == hint * 3:
                    return np.array(flat, dtype=np.float64).reshape(hint, 3)
                elif len(flat) == hint * 2:
                    return np.array(flat, dtype=np.float64).reshape(hint, 2)
                elif len(flat) >= hint * 2:
                    stride = len(flat) // hint
                    pts = [flat[k * stride : k * stride + 2] for k in range(hint)]
                    return np.array(pts, dtype=np.float64)
                else:
                    return np.empty((0, 2))

            if len(flat) % 3 == 0:
                return np.array(flat, dtype=np.float64).reshape(-1, 3)
            elif len(flat) % 2 == 0:
                return np.array(flat, dtype=np.float64).reshape(-1, 2)

        return np.empty((0, 2))

    @classmethod
    @abstractmethod
    def output_keys(cls) -> list[str]:
        """Returns a list of keys that this calculator will output."""
        pass

    @abstractmethod
    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Calculate metrics for a single episode."""
        pass
