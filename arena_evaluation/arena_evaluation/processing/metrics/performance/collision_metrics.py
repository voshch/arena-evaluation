from __future__ import annotations

import typing
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.safety.contact_attribution import (
    ROBOT_COLLISION_CONTACTS_TOPIC,
    ROBOT_COLLISION_EVENTS_TOPIC,
)
from arena_evaluation.processing.outcome import UNRESOLVED, outcome_verdict
from arena_evaluation.storage.registry import MetricScope, NullReason, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

# Any collision fails unless the suite sets max_collisions.
DEFAULT_MAX_COLLISIONS: Final[int] = 1

COLLISION_AMOUNT = define_metric(
    "collision_amount",
    pl.Int64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER, MetricScope.CONTACT),
    unit="",
    description="Collision onsets of the scope window in the native robot_collision_events stream; NULL when that stream or count was not recorded.",
    null_reason=True,
)
COLLISION_AMOUNT_PEDESTRIAN = define_metric(
    "collision_amount_pedestrian",
    pl.Int64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="",
    description="Collision onsets with a pedestrian in the scope window; NULL when the pedestrian count was not recorded.",
    null_reason=True,
)
COLLISION_AMOUNT_STATIC = define_metric(
    "collision_amount_static",
    pl.Int64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER),
    unit="",
    description="Collision onsets with a static obstacle in the scope window; NULL when the static count was not recorded.",
    null_reason=True,
)
COLLISION_AMOUNT_WALL = define_metric(
    "collision_amount_wall",
    pl.Int64,
    scopes=(MetricScope.ROBOT, MetricScope.ENCOUNTER, MetricScope.CONTACT),
    unit="",
    description="Collision onsets with a wall in the scope window; NULL when the wall count was not recorded.",
    null_reason=True,
)
SUCCESS = define_metric(
    "success",
    pl.Boolean,
    scopes=(MetricScope.ROBOT,),
    unit="",
    description="Robot success: runtime verdict SUCCESS and fewer collisions than max_collisions (default 1).",
    null_reason=True,
)


class CollisionMetricsCalculator(BaseMetricCalculator):
    """Collision onsets of the robot, its windows and contacts, and the runtime's success verdict."""

    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["robot_collision_events", ("robot_tracks", "robot_odom")]

    OUTPUTS = (
        COLLISION_AMOUNT,
        COLLISION_AMOUNT_WALL,
        COLLISION_AMOUNT_STATIC,
        COLLISION_AMOUNT_PEDESTRIAN,
        SUCCESS,
    )
    CHILD_OUTPUTS = ("collision_obstacles", "collisions")
    DIRECTIONS = {
        "collision_amount": Direction.LOWER,
        "success": Direction.HIGHER,
    }
    PRIMARY_OUTPUTS = ["success", "collision_amount"]

    @classmethod
    def required_topics_for_scope(
        cls, scope: MetricScope
    ) -> tuple[str | tuple[str, ...], ...]:
        # A window may legitimately lack an optional sensor stream; a missing stream is then per-metric
        # `no_samples`, not a row-level `topic_missing` that would block the whole encounter or contact.
        return (
            ()
            if scope in (MetricScope.ENCOUNTER, MetricScope.CONTACT)
            else super().required_topics_for_scope(scope)
        )

    @staticmethod
    def _count_rising_edges(is_active: np.ndarray) -> int:
        count = 0
        if len(is_active) > 0 and is_active[0]:
            count += 1
        for index in range(1, len(is_active)):
            if is_active[index] and not is_active[index - 1]:
                count += 1
        return count

    @staticmethod
    def _collision_frame(episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The native robot_collision_events stream; None when the recording carries none."""
        if (
            episode.topics is not None
            and ROBOT_COLLISION_EVENTS_TOPIC in episode.topics
        ):
            return episode.topics[ROBOT_COLLISION_EVENTS_TOPIC].sort("t_ns")
        return None

    @classmethod
    def _column_rising_edges(
        cls, episode: AlignedEpisodeBundle, column: str
    ) -> int | None:
        frame = cls._collision_frame(episode)
        if frame is None:
            return None
        if column not in frame.columns:
            return None
        counts = frame[column].to_numpy().astype(float)
        if np.isnan(counts).all():
            return None
        return cls._count_rising_edges(counts > 0.0)

    @staticmethod
    def _hit_obstacles(episode: AlignedEpisodeBundle) -> list[str]:
        contacts = (
            episode.topics.get(ROBOT_COLLISION_CONTACTS_TOPIC)
            if episode.topics
            else None
        )
        if contacts is None or len(contacts) == 0:
            return []
        return sorted(
            {
                str(obstacle)
                for obstacle in contacts["obstacle_id"].drop_nulls()
                if obstacle
            }
        )

    @classmethod
    def _collision_counts(cls, episode: AlignedEpisodeBundle) -> dict[str, typing.Any]:
        """Rising-edge collision counts of each kind plus the episode's hit obstacles."""
        return {
            "collision_amount": cls._column_rising_edges(episode, "event_count"),
            "collision_amount_wall": cls._column_rising_edges(episode, "wall_count"),
            "collision_amount_static": cls._column_rising_edges(episode, "static_count"),
            "collision_amount_pedestrian": cls._column_rising_edges(
                episode, "ped_count"
            ),
            "collision_obstacles": cls._hit_obstacles(episode),
        }

    def _window_onset_count(
        self,
        episode: AlignedEpisodeBundle,
        scope_slice: ScopeSlice,
        column: str,
    ) -> int | None:
        """Collision onsets of the window's own contact interval.

        The rising edge is detected on the padded window the slice carries, then counted only when its
        timestamp falls inside ``[t_start_ns, t_end_ns]``. The padding supplies the preceding state, and
        the interval filter keeps a continued bout from being counted again by the next contact window of
        the same collision (the detector splits one bout into several contact windows on the raw stream).
        """
        events = self.native_topics(episode).get(ROBOT_COLLISION_EVENTS_TOPIC)
        if events is None:
            return None
        active = (pl.col(column).cast(pl.Float64) > 0.0).alias("active")
        onsets = (
            events.select(pl.col("t_ns").cast(pl.Int64), active)
            .sort("t_ns")
            .filter(pl.col("active") & ~pl.col("active").shift(1, fill_value=False))
            .get_column("t_ns")
        )
        inside = (onsets >= scope_slice.t_start_ns) & (onsets <= scope_slice.t_end_ns)
        return int(inside.sum())

    def _contact_result(
        self, episode: AlignedEpisodeBundle, scope_slice: ScopeSlice
    ) -> dict[str, typing.Any]:
        """One contact window's own collision counts and its subject actor."""
        return {
            "collision_amount": self._window_onset_count(
                episode, scope_slice, "event_count"
            ),
            "collision_amount_wall": self._window_onset_count(
                episode, scope_slice, "wall_count"
            ),
            "collision_obstacles": [str(scope_slice.identity["obstacle_id"])],
            "collisions": [],
        }

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        del prior_results
        if scope_slice is None or scope_slice.scope == MetricScope.ROBOT:
            return self.calculate(episode, {})
        if scope_slice.scope == MetricScope.ENCOUNTER:
            return self._collision_counts(episode)
        if scope_slice.scope == MetricScope.CONTACT:
            return self._contact_result(episode, scope_slice)
        raise ValueError(
            f"collision_metrics does not serve the {scope_slice.scope.value} scope"
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        del prior_results
        result, success = outcome_verdict(episode.outcome_state, episode.outcome_info)
        counts = self._collision_counts(episode)
        collision_amount = counts["collision_amount"]
        max_collisions = int(
            self.metrics_config.get("max_collisions", DEFAULT_MAX_COLLISIONS)
        )
        if result == UNRESOLVED:
            success_value: bool | None = None
            success_reason: str | None = NullReason.NOT_RECORDED.value
        elif collision_amount is None:
            success_value = None
            success_reason = NullReason.NO_SAMPLES.value
        else:
            success_value = success and collision_amount < max_collisions
            success_reason = None
        return {
            "collision_amount": counts["collision_amount"],
            "collision_amount_wall": counts["collision_amount_wall"],
            "collision_amount_static": counts["collision_amount_static"],
            "collision_amount_pedestrian": counts["collision_amount_pedestrian"],
            "success": success_value,
            f"success{NULL_REASON_SUFFIX}": success_reason,
            "collision_obstacles": counts["collision_obstacles"],
            "collisions": [],
        }
