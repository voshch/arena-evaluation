"""Episode duration, the robot's time to its goal, and idling dwell."""

from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    BaseMetricCalculator,
    Direction,
)
from arena_evaluation.processing.metrics.common.routes import GOAL_TOLERANCE_M
from arena_evaluation.processing.metrics.performance.motion_metrics import (
    episode_speed_mps,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

NANOSECONDS_PER_SECOND: typing.Final[float] = 1e9
IDLE_SPEED_MPS: typing.Final[float] = 0.01

TIME_DIFF = define_metric(
    "time_diff",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    unit="ns",
    description="Recorded episode duration in nanoseconds, the span of the aligned clock.",
    null_reason=True,
)
TIME_TO_GOAL = define_metric(
    "time_to_goal",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds from the episode start until this robot's ground-truth track first comes within 1 m of its mission goal; NULL when it never does.",
    null_reason=True,
)
IDLING_TIME = define_metric(
    "idling_time",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Time spent with speed below 0.01 m/s; NULL when the filtered speed track has no sample per episode clock step.",
    null_reason=True,
)


class TimeMetricsCalculator(BaseMetricCalculator):
    """Episode span, the robot's time to goal, and idling time.

    Metrics:
    - time_diff: Recorded episode duration in nanoseconds (EPISODE)
    - time_to_goal: Seconds to this robot's first arrival at its goal (ROBOT)
    - idling_time: Time with speed below 0.01 m/s (ROBOT)
    """

    NAME = "time_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["motion_metrics"]
    REQUIRED_TOPICS = ["robot_tracks"]

    OUTPUTS = (TIME_DIFF, TIME_TO_GOAL, IDLING_TIME)
    DIRECTIONS = {"time_to_goal": Direction.LOWER, "idling_time": Direction.LOWER}
    PRIMARY_OUTPUTS = ["time_to_goal", "idling_time"]

    def _time_to_goal(self, episode: AlignedEpisodeBundle, start_ns: int) -> tuple[float | None, str | None]:
        """Seconds from the episode start to the first goal arrival; with the reason when there is none."""
        if len(episode.goal_pos) < 2:
            return None, "reference_missing"
        track = self.resolve_robot_frame(episode)
        if track is None:
            return None, "topic_missing:robot_tracks"
        distance_sq = (track["x"] - float(episode.goal_pos[0])) ** 2 + (track["y"] - float(episode.goal_pos[1])) ** 2
        arrivals = (distance_sq <= GOAL_TOLERANCE_M**2).arg_true()
        if len(arrivals) == 0:
            return None, "no_samples"
        return float(track["t_ns"][int(arrivals[0])] - start_ns) / NANOSECONDS_PER_SECOND, None

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        if episode.data is None or len(episode.data) == 0:
            return dict.fromkeys(("time_diff", "time_to_goal", "idling_time"))

        t_ns = episode.data["t_ns"].to_numpy()
        if len(t_ns) < 2:
            return dict.fromkeys(("time_diff", "time_to_goal", "idling_time"))

        duration_ns = int(t_ns[-1] - t_ns[0])
        goal_s, goal_reason = self._time_to_goal(episode, int(t_ns[0]))
        speed_mps, _speed_time_ns = episode_speed_mps(self, episode)
        idling_time_s: float | None = None
        if len(speed_mps) == len(t_ns):
            dt_s = np.diff(t_ns) / NANOSECONDS_PER_SECOND
            idling_time_s = float(np.sum(dt_s[speed_mps[:-1] < IDLE_SPEED_MPS]))
        return {
            "time_diff": duration_ns,
            "time_to_goal": goal_s,
            f"time_to_goal{NULL_REASON_SUFFIX}": goal_reason,
            "idling_time": idling_time_s,
        }
