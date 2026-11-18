from __future__ import annotations

import typing

import numpy as np
import polars as pl
from dtaidistance import dtw_ndim
from scipy.spatial.distance import cdist

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import (
    asof_join,
    interpolate_values,
)
from arena_evaluation.processing.metrics.common.routes import (
    plan_mission,
    route_waypoints,
)
from arena_evaluation.processing.path.static_environment import grid_for_map
from arena_evaluation.processing.path.theta_star import plan_route
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_TWO_PI: typing.Final[float] = 2.0 * np.pi

# The world asset is absent from disk: the route was never measurable, but the episode still ran.
MISSING_WORLD: typing.Final[str] = "not_recorded:world"
_GRID_OUTPUTS: typing.Final[tuple[str, ...]] = (
    "theta_star_length",
    "ade",
    "mhd",
    "frechet_distance",
    "adtw",
)

ADE = define_metric(
    "ade",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Single-leg tracking: average displacement error against the current-leg Theta* reference path, m.",
    null_reason=True,
)
FDE = define_metric(
    "fde",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Single-leg tracking: final displacement error against the current-leg Theta* reference path, m.",
    null_reason=True,
)
ADTW = define_metric(
    "adtw",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Dynamic-time-warping distance against the current-leg Theta* reference path, m: plain DTW with no penalty weighting.",
    null_reason=True,
)
MHD = define_metric(
    "mhd",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Single-leg tracking: modified Hausdorff distance against the current-leg Theta* reference path, m.",
    null_reason=True,
)
FRECHET_DISTANCE = define_metric(
    "frechet_distance",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Single-leg tracking: discrete Frechet distance against the current-leg Theta* reference path, m.",
    null_reason=True,
)
PATH_IRREGULARITY = define_metric(
    "path_irregularity",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="rad/m",
    description="Heading change beyond the minimum turn to the goal, per metre travelled.",
    null_reason=True,
)
TOPOLOGICAL_COMPLEXITY = define_metric(
    "topological_complexity",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="1",
    description="Total absolute net winding of every pedestrian around the robot's path, in turns.",
    null_reason=True,
)
THETA_STAR_LENGTH = define_metric(
    "theta_star_length",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="m",
    description="Whole-mission route geodesic: sum of per-leg Theta* between the start and every commanded task goal, reached or not; the shared route reference of path_efficiency and spl_geodesic.",
    null_reason=True,
)


def _discrete_frechet(path: np.ndarray, reference: np.ndarray) -> float | None:
    """Discrete Frechet distance via dynamic programming with <=500 point decimation."""
    n, m = len(path), len(reference)
    if n == 0 or m == 0:
        return None
    if n > 500:
        path = path[np.linspace(0, n - 1, 500, dtype=int)]
        n = len(path)
    if m > 500:
        reference = reference[np.linspace(0, m - 1, 500, dtype=int)]
        m = len(reference)

    ca = np.full((n, m), -1.0)
    ca[0, 0] = np.linalg.norm(path[0] - reference[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(path[i] - reference[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(path[0] - reference[j]))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]),
                np.linalg.norm(path[i] - reference[j]),
            )
    return float(ca[n - 1, m - 1])


def _path_irregularity(
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    yaw: np.ndarray,
    goal_pt: tuple[float, float] | None,
) -> float | None:
    """Heading change beyond the minimum turn to the goal, per metre travelled; None without a usable track."""
    if len(pos_x) < 2 or goal_pt is None:
        return None
    length_m = float(np.sum(np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)))
    if length_m <= 0.1:
        return None
    delta_yaw = np.diff(yaw)
    delta_yaw = (delta_yaw + np.pi) % _TWO_PI - np.pi
    target_angle = np.arctan2(goal_pt[1] - pos_y[0], goal_pt[0] - pos_x[0])
    rotation_to_target = (target_angle - yaw[0] + np.pi) % _TWO_PI - np.pi
    return float((np.sum(np.abs(delta_yaw)) - np.abs(rotation_to_target)) / length_m)


def _topological_complexity(
    peds_df: pl.DataFrame | None, pos_x: np.ndarray, pos_y: np.ndarray, t_ns: np.ndarray
) -> float | None:
    """Total absolute net winding of every pedestrian around the robot's path."""
    if peds_df is None or len(peds_df) == 0 or len(pos_x) < 2 or len(t_ns) < 2:
        return None
    required = {"t_ns", "ped_id", "x", "y"}
    if not required <= set(peds_df.columns):
        return None
    robot = pl.LazyFrame({"t_ns": t_ns, "rx": pos_x, "ry": pos_y}).sort("t_ns")
    pedestrians = (
        peds_df.lazy()
        .select("t_ns", "ped_id", "x", "y")
        .drop_nulls(["t_ns", "ped_id", "x", "y"])
        .sort("t_ns")
    )
    rel_x = pl.col("rx") - pl.col("x")
    rel_y = pl.col("ry") - pl.col("y")
    angle = pl.arctan2(rel_y, rel_x)
    delta_angle = angle.diff().over("ped_id")
    wrapped = ((delta_angle + pl.lit(np.pi)) % pl.lit(_TWO_PI)) - pl.lit(np.pi)
    winding = (
        asof_join(pedestrians, robot, tolerance_ns=None)
        .with_columns(wrapped_d_angle=wrapped)
        .group_by("ped_id")
        .agg(
            net_winding=pl.col("wrapped_d_angle").drop_nulls().sum().abs()
            / pl.lit(_TWO_PI)
        )
        .select(total_winding=pl.col("net_winding").sum())
        .collect()
    )
    if len(winding) == 0 or winding["total_winding"][0] is None:
        return None
    return float(winding["total_winding"][0])


class TrajectoryMetricsCalculator(BaseMetricCalculator):
    """Whole-mission route reference plus current-leg tracking errors.

    ``theta_star_length`` is the whole-mission route geodesic: the sum of per-leg Theta* lengths
    between the start and every commanded task goal, reached or not; ``path_efficiency`` and
    ``spl_geodesic`` share it. ``ade``/``fde``/``mhd``/``frechet_distance``/``adtw`` are tracking
    errors against the current-leg start -> goal Theta* reference and deliberately stay single-leg;
    ``adtw`` is plain dtaidistance DTW, not a penalty-weighted asymmetric variant.
    """

    NAME = "trajectory_naturalness"
    CATEGORY = "naturalness"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]

    OUTPUTS = (
        ADE,
        FDE,
        ADTW,
        MHD,
        FRECHET_DISTANCE,
        PATH_IRREGULARITY,
        TOPOLOGICAL_COMPLEXITY,
        THETA_STAR_LENGTH,
    )
    DIRECTIONS = {
        "ade": Direction.LOWER,
        "fde": Direction.LOWER,
        "adtw": Direction.LOWER,
        "mhd": Direction.LOWER,
        "frechet_distance": Direction.LOWER,
        "path_irregularity": Direction.LOWER,
        "topological_complexity": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = ["ade", "fde", "mhd", "frechet_distance"]

    def _episode_pose(
        self, episode: AlignedEpisodeBundle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ground-truth pose of the episode on the native /tf clock, falling back to the aligned ground-truth segment."""
        pos_x, pos_y, yaw, t_ns = self.resolve_native_pose(episode)
        if len(pos_x) > 0:
            return pos_x, pos_y, yaw, t_ns
        if episode.data is None or len(episode.data) == 0:
            empty = np.array([])
            return empty, empty, empty, np.array([], dtype=np.int64)
        pos_x, pos_y, yaw, _odom_x, _odom_y, _odom_yaw = self.resolve_robot_pose(
            episode
        )
        t_ns = episode.data["t_ns"].to_numpy().astype(np.int64)
        return pos_x, pos_y, yaw, t_ns

    @staticmethod
    def _goal_point(episode: AlignedEpisodeBundle) -> tuple[float, float] | None:
        return (
            (float(episode.goal_pos[0]), float(episode.goal_pos[1]))
            if episode.goal_pos and len(episode.goal_pos) >= 2
            else None
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        pos_x, pos_y, yaw, t_ns = self._episode_pose(episode)
        if len(pos_x) == 0:
            return {key: None for key in self.output_keys()}

        goal_pt = self._goal_point(episode)
        results: dict[str, typing.Any] = {key: None for key in self.output_keys()}
        results["path_irregularity"] = _path_irregularity(pos_x, pos_y, yaw, goal_pt)
        results["topological_complexity"] = _topological_complexity(
            self.native_ped_frame(episode), pos_x, pos_y, t_ns
        )

        if goal_pt is None:
            return results
        start_pt = (float(pos_x[0]), float(pos_y[0]))
        results["fde"] = float(
            np.sqrt((pos_x[-1] - goal_pt[0]) ** 2 + (pos_y[-1] - goal_pt[1]) ** 2)
        )
        try:
            grid = grid_for_map(
                episode.map, episode.run_dir, self.robot_params.robot_radius
            )
        except FileNotFoundError:
            # The world asset is not on disk: the grid outputs were never measurable, the pose ones still are.
            for key in _GRID_OUTPUTS:
                results[f"{key}_null_reason"] = MISSING_WORLD
            return results
        waypoints = route_waypoints(
            start_pt=start_pt,
            goal_pt=goal_pt,
            phases=episode.phases,
            topics=episode.topics,
        )
        mission = plan_mission(grid=grid, waypoints=waypoints)
        results["theta_star_length"] = mission.length_m
        if mission.length_m is None:
            results["theta_star_length_null_reason"] = mission.null_reason
        if grid is not None:
            reference = plan_route(grid, start_pt, goal_pt)
            if reference.points is not None:
                results.update(self._reference_metrics(pos_x, pos_y, reference.points))
        return results

    @staticmethod
    def _reference_metrics(
        pos_x: np.ndarray, pos_y: np.ndarray, reference_path: np.ndarray
    ) -> dict[str, float | None]:
        """ADE/MHD/Frechet/DTW against the geometric Theta* reference path."""
        ref_x, ref_y = reference_path[:, 0], reference_path[:, 1]
        metrics: dict[str, float | None] = {
            "ade": None,
            "mhd": None,
            "frechet_distance": None,
            "adtw": None,
        }

        current_distance = np.insert(
            np.cumsum(np.sqrt(np.diff(pos_x) ** 2 + np.diff(pos_y) ** 2)), 0, 0
        )
        reference_distance = np.insert(
            np.cumsum(np.sqrt(np.diff(ref_x) ** 2 + np.diff(ref_y) ** 2)), 0, 0
        )
        if current_distance[-1] > 0 and reference_distance[-1] > 0:
            current_fraction = current_distance / current_distance[-1]
            reference_fraction = reference_distance / reference_distance[-1]
            interpolated_x = interpolate_values(
                reference_fraction, ref_x, current_fraction
            )
            interpolated_y = interpolate_values(
                reference_fraction, ref_y, current_fraction
            )
            metrics["ade"] = float(
                np.mean(
                    np.sqrt(
                        (pos_x - interpolated_x) ** 2 + (pos_y - interpolated_y) ** 2
                    )
                )
            )

        robot_points = np.column_stack((pos_x, pos_y))
        reference_points = np.column_stack((ref_x, ref_y))
        metrics["mhd"] = TrajectoryMetricsCalculator._modified_hausdorff(
            robot_points, reference_points
        )

        robot_sample = robot_points
        reference_sample = reference_points
        if len(robot_points) > 1000:
            robot_sample = robot_points[
                np.linspace(0, len(robot_points) - 1, 1000, dtype=int)
            ]
        if len(reference_points) > 1000:
            reference_sample = reference_points[
                np.linspace(0, len(reference_points) - 1, 1000, dtype=int)
            ]
        metrics["frechet_distance"] = _discrete_frechet(robot_sample, reference_sample)
        metrics["adtw"] = TrajectoryMetricsCalculator._dtw(
            robot_sample, reference_sample
        )
        return metrics

    @staticmethod
    def _modified_hausdorff(
        robot_points: np.ndarray, reference_points: np.ndarray
    ) -> float:
        """Modified Hausdorff distance between two paths."""
        try:
            from scipy.spatial import cKDTree

            robot_tree = cKDTree(robot_points)
            reference_tree = cKDTree(reference_points)
            robot_to_reference, _ = reference_tree.query(robot_points, k=1)
            reference_to_robot, _ = robot_tree.query(reference_points, k=1)
        except (ImportError, ValueError):
            distances = cdist(robot_points, reference_points)
            return float(
                max(
                    np.mean(np.min(distances, axis=1)),
                    np.mean(np.min(distances, axis=0)),
                )
            )
        return float(max(np.mean(robot_to_reference), np.mean(reference_to_robot)))

    @staticmethod
    def _dtw(robot_points: np.ndarray, reference_points: np.ndarray) -> float:
        """Plain dynamic-time-warping distance of the robot path against the Theta* reference, no penalty."""
        return float(dtw_ndim.distance(robot_points, reference_points))

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        raise ValueError(
            f"trajectory_naturalness does not serve the {scope_slice.scope.value} scope"
        )
