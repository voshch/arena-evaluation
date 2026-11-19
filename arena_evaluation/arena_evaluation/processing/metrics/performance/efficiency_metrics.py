from __future__ import annotations

import math
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX, BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.common.routes import REFERENCE_EXCEEDS_PATH
from arena_evaluation.processing.metrics.social.interaction_common import (
    NS_PER_S,
    detector_params_from_config,
    pedestrian_encounter_spans,
    resolve_robot_trajectory,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PATH_EFFICIENCY = define_metric("path_efficiency", pl.Float64, scopes=(MetricScope.ROBOT,), unit="1", description="Mission route geodesic through the commanded task goals, each leg ending on its 1 m arrival radius, over the travelled path length; NULL when either side is unavailable or the planned route is longer than the sampled path (a ratio above one, never clamped).", null_reason=True)
RELATIVE_THROUGHPUT = define_metric("relative_throughput", pl.Float64, scopes=(MetricScope.ROBOT,), unit="1", description="Planned mission time over the episode's time to goal, higher is better. Planned mission time is the mission route length at the robot's free speed, the median speed outside encounter windows.", null_reason=True)


def _finite_positive(value: object) -> float | None:
    """The value as a finite float when it is positive, else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(typing.cast(float, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


class PathEfficiencyCalculator(BaseMetricCalculator):
    """Whole-mission route geodesic over travelled path length, and the relative throughput of the same mission.

    The reference is the mission route through every commanded task goal, not just the current
    start -> goal leg, and ``spl_geodesic`` uses the same reference. A null result carries its
    reason: ``no_samples`` when the reference or the path is unavailable,
    ``invalid_episode:reference_exceeds_path`` when the planned route is longer than the travelled
    path. The travelled length is sampled ground truth, so it can undercount by a few percent; a
    route longer than the path is that sampling anomaly, not an efficiency above one.
    """

    NAME = "path_efficiency"
    CATEGORY = "performance"
    DEPENDS_ON = ["path_metrics", "trajectory_naturalness", "time_metrics"]
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]

    OUTPUTS = (PATH_EFFICIENCY, RELATIVE_THROUGHPUT)
    DIRECTIONS = {"path_efficiency": Direction.HIGHER, "relative_throughput": Direction.HIGHER}
    PRIMARY_OUTPUTS = ["path_efficiency", "relative_throughput"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Geodesic reference over travelled length and planned-over-actual mission time; null when either side is unavailable."""
        results = self._path_efficiency(prior_results)
        results["relative_throughput"] = self._relative_throughput(episode, prior_results)
        return results

    @staticmethod
    def _path_efficiency(prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        path_length = _finite_positive(prior_results.get("path_length_m"))
        reference_length = _finite_positive(prior_results.get("theta_star_length"))
        if path_length is None or reference_length is None:
            return {"path_efficiency": None}
        if reference_length > path_length:
            return {"path_efficiency": None, f"path_efficiency{NULL_REASON_SUFFIX}": REFERENCE_EXCEEDS_PATH}
        efficiency = reference_length / path_length
        if not math.isfinite(efficiency):
            return {"path_efficiency": None}
        return {"path_efficiency": efficiency}

    def _relative_throughput(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> float | None:
        planned_m = _finite_positive(prior_results.get("theta_star_length"))
        time_to_goal_s = _finite_positive(prior_results.get("time_to_goal"))
        if planned_m is None or time_to_goal_s is None:
            return None
        free_speed_mps = self._robot_free_speed_mps(episode)
        if free_speed_mps is None:
            return None
        return (planned_m / free_speed_mps) / time_to_goal_s

    def _robot_free_speed_mps(self, episode: AlignedEpisodeBundle) -> float | None:
        """Median robot speed over the sample pairs outside every pedestrian encounter window."""
        robot, _ = resolve_robot_trajectory(self, episode)
        spans = pedestrian_encounter_spans(episode, float(self.robot_params.robot_radius), detector_params_from_config(self.metrics_config))
        free = np.ones(len(robot.t_ns), dtype=bool)
        for _, start_ns, end_ns in spans:
            free &= ~((robot.t_ns >= start_ns) & (robot.t_ns <= end_ns))
        dt_s = np.diff(robot.t_ns).astype(np.float64) / NS_PER_S
        distance_m = np.hypot(np.diff(robot.x), np.diff(robot.y))
        both_free = free[:-1] & free[1:] & (dt_s > 0.0)
        speeds = distance_m[both_free] / dt_s[both_free]
        if speeds.size == 0:
            return None
        speed = float(np.median(speeds))
        return speed if speed > 0.0 else None
