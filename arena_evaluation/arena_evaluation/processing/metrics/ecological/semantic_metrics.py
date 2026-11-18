from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import (
    BaseMetricCalculator,
    Direction,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.common.alignment import asof_position
from arena_evaluation.processing.metrics.common.events import (
    reconstruct_events as _reconstruct_events,
)
from arena_evaluation.processing.metrics.common.parsing import parse_bool, parse_float
from arena_evaluation.processing.metrics.ecological.mechanism_common import (
    RobotIdentity,
    episode_end_t_ns,
    parse_members,
    robot_env_snapshot,
    robot_identity,
)
from arena_evaluation.storage.registry import MetricScope, define_metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

ELEVATOR_RIDES_ANYONE = define_metric(
    "elevator_rides_anyone",
    pl.Int64,
    scopes=(MetricScope.EPISODE,),
    description="Elevator rides: arrivals of a cabin with occupants aboard, counted for anyone of the environment.",
    null_reason=True,
)
ELEVATOR_RIDES_ROBOT = define_metric(
    "elevator_rides_robot",
    pl.Int64,
    scopes=(MetricScope.ROBOT,),
    description="Elevator rides with the robot itself in the cabin members within 3 s of the arrival.",
    null_reason=True,
)
TIME_WAITING_AT_DOORS_ANYONE = define_metric(
    "time_waiting_at_doors_anyone",
    pl.Float64,
    scopes=(MetricScope.EPISODE,),
    unit="s",
    description="Seconds a door is triggered and not open while anyone waits.",
    null_reason=True,
)
TIME_WAITING_AT_DOORS_ROBOT = define_metric(
    "time_waiting_at_doors_robot",
    pl.Float64,
    scopes=(MetricScope.ROBOT,),
    unit="s",
    description="Seconds a door is triggered and not open while the robot itself is a door member.",
    null_reason=True,
)

ARRIVAL_MEMBER_WINDOW_S = 3.0


def _members_recorded(events: pl.DataFrame, kind: str) -> bool:
    """True when any `kind` entity reports a non-empty members list in the episode."""
    rows = events.filter((pl.col("kind") == kind) & (pl.col("field") == "members"))
    return any(bool(v) for v in rows["current"])


class SemanticInteractionMetricsCalculator(BaseMetricCalculator):
    """Door waiting and elevator rides, with one name for anyone and one for the robot itself."""

    NAME = "semantic_interaction_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["world_semantics"]
    PARAMS = ()

    OUTPUTS = (
        TIME_WAITING_AT_DOORS_ANYONE,
        TIME_WAITING_AT_DOORS_ROBOT,
        ELEVATOR_RIDES_ANYONE,
        ELEVATOR_RIDES_ROBOT,
    )
    DIRECTIONS = {
        "time_waiting_at_doors_anyone": Direction.LOWER,
        "time_waiting_at_doors_robot": Direction.LOWER,
    }
    PRIMARY_OUTPUTS = [
        "time_waiting_at_doors_anyone",
        "time_waiting_at_doors_robot",
        "elevator_rides_anyone",
        "elevator_rides_robot",
    ]

    @staticmethod
    def _events(episode: AlignedEpisodeBundle) -> pl.DataFrame:
        """Semantic events reconstructed from the robot's env snapshot."""
        return _reconstruct_events(
            robot_env_snapshot(episode.semantic_snapshot, episode.robot_name)
        )

    def calculate(
        self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """Episode scope: door waiting and rides by anyone aboard; NULL when no semantic snapshot was recorded."""
        del prior_results
        if episode.semantic_snapshot is None:
            return {
                "time_waiting_at_doors_anyone": None,
                "elevator_rides_anyone": None,
            }
        events = self._events(episode)
        return {
            "time_waiting_at_doors_anyone": self._time_waiting_at_doors(
                events, episode_end_t_ns(episode), None
            ),
            "elevator_rides_anyone": self._elevator_rides(events),
        }

    def calculate_for_scope(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
        scope_slice: ScopeSlice | None = None,
    ) -> dict[str, typing.Any]:
        if scope_slice is None or scope_slice.scope != MetricScope.ROBOT:
            return self.calculate(episode, prior_results)
        identity = robot_identity(episode)
        if episode.semantic_snapshot is None or identity is None:
            return {
                "time_waiting_at_doors_robot": None,
                "elevator_rides_robot": None,
            }
        events = self._events(episode)
        end_t_ns = episode_end_t_ns(episode)
        anyone_waiting = self._time_waiting_at_doors(events, end_t_ns, None)
        anyone_rides = self._elevator_rides(events)
        robot_waiting_defined = self._has_kind(events, "door") and (
            anyone_waiting == 0.0 or _members_recorded(events, "door")
        )
        robot_rides_defined = self._has_kind(events, "elevator") and (
            anyone_rides == 0 or _members_recorded(events, "elevator")
        )
        return {
            "time_waiting_at_doors_robot": self._time_waiting_at_doors(
                events, end_t_ns, identity
            )
            if robot_waiting_defined
            else None,
            "elevator_rides_robot": self._elevator_rides_robot(events, identity)
            if robot_rides_defined
            else None,
        }

    @staticmethod
    def _has_kind(events: pl.DataFrame, kind: str) -> bool:
        return len(events.filter(pl.col("kind") == kind)) > 0

    def _time_waiting_at_doors(
        self, events: pl.DataFrame, end_t_ns: int | None, identity: RobotIdentity | None
    ) -> float:
        """Seconds each door is triggered and not open while the waiting party is present: anyone when `identity` is None."""
        door_events = events.filter(
            (pl.col("kind") == "door")
            & pl.col("field").is_in(["state", "triggered", "members"])
        ).sort("t_ns")

        total_s = 0.0
        for _entity, group in door_events.group_by("entity"):
            state = "closed"
            triggered = False
            in_party = identity is None
            prev_time_ns = None

            g = group.sort("t_ns")
            for t_raw, field, current in zip(
                g["t_ns"].to_numpy(),
                g["field"].to_numpy(),
                g["current"].to_numpy(),
                strict=True,
            ):
                if current is None:
                    continue
                t = int(t_raw)
                if (
                    prev_time_ns is not None
                    and triggered
                    and in_party
                    and state != "open"
                ):
                    total_s += (t - prev_time_ns) / 1e9

                if field == "state":
                    state = str(current)
                elif field == "triggered":
                    triggered = parse_bool(current)
                elif identity is not None:
                    in_party = identity.any_of(parse_members(current))
                prev_time_ns = t

            if (
                end_t_ns is not None
                and prev_time_ns is not None
                and end_t_ns > prev_time_ns
                and triggered
                and in_party
                and state != "open"
            ):
                total_s += (end_t_ns - prev_time_ns) / 1e9

        return total_s

    def _elevator_rides_robot(
        self, events: pl.DataFrame, identity: RobotIdentity
    ) -> int:
        window_ns = int(ARRIVAL_MEMBER_WINDOW_S * 1e9)
        rides = 0
        for _entity, group in events.filter(pl.col("kind") == "elevator").group_by(
            "entity"
        ):
            g = group.sort("t_ns")
            arrivals = g.filter(
                (pl.col("field") == "just_arrived")
                & (pl.col("current") == "true")
                & (pl.col("previous") != "true")
            )
            members = g.filter(pl.col("field") == "members")
            if len(arrivals) == 0 or len(members) == 0:
                continue
            m_times = members["t_ns"].to_numpy()
            m_robot = [identity.any_of(parse_members(v)) for v in members["current"]]
            for t in arrivals["t_ns"].to_numpy():
                idx = max(asof_position(m_times, t), 0)
                hi = int(np.searchsorted(m_times, t + window_ns, side="right"))
                if any(m_robot[idx : max(hi, idx + 1)]):
                    rides += 1
        return rides

    def _elevator_rides(self, events: pl.DataFrame) -> int:
        elevator_events = events.filter(
            (pl.col("kind") == "elevator")
            & pl.col("field").is_in(["occupants", "just_arrived"])
        ).sort("t_ns")

        rides = 0
        for _entity, group in elevator_events.group_by("entity"):
            occupants = 0.0
            g = group.sort("t_ns")
            for field, current in zip(
                g["field"].to_numpy(), g["current"].to_numpy(), strict=True
            ):
                if current is None:
                    continue
                if field == "occupants":
                    occupants = parse_float(current)
                elif parse_bool(current) and occupants > 0:
                    rides += 1

        return rides
