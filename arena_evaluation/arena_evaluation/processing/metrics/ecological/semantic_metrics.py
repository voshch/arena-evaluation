from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _reconstruct_events
from arena_evaluation.processing.metrics.ecological.mechanism_common import RobotIdentity, parse_members, robot_env_snapshot
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


def _parse_bool(token: str) -> bool:
    return token.strip().lower() in ("true", "1")


def _parse_float(token: str) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return 0.0


ARRIVAL_MEMBER_WINDOW_S = 3.0


def _members_recorded(events: pl.DataFrame, kind: str) -> bool:
    """True when any `kind` entity reports a non-empty members list in the episode."""
    rows = events.filter((pl.col("kind") == kind) & (pl.col("field") == "members"))
    return any(bool(v) for v in rows["current"].to_list())


class SemanticInteractionMetricsCalculator(BaseMetricCalculator):
    """Computes door waiting time and elevator ride metrics from semantic snapshots."""

    NAME = "semantic_interaction_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["semantic_snapshot"]

    UNITS = {
        "time_waiting_at_doors": "s",
        "elevator_rides": "",
        "time_waiting_at_doors_robot": "s",
        "elevator_rides_robot": "",
    }

    OUTPUT_DIRECTIONS = {"time_waiting_at_doors_robot": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_waiting_at_doors",
            "elevator_rides",
            "time_waiting_at_doors_robot",
            "elevator_rides_robot",
        ]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:

        events = _reconstruct_events(robot_env_snapshot(episode.semantic_snapshot, episode.robot_name))
        if len(events) == 0:
            return {"time_waiting_at_doors": 0.0, "elevator_rides": 0, "time_waiting_at_doors_robot": None, "elevator_rides_robot": None}

        end_time_ns = None
        if episode.data is not None and "time_ns" in episode.data.columns and len(episode.data) > 0:
            end_time_ns = int(typing.cast("float", episode.data["time_ns"].max()))

        if isinstance(episode.data, pl.LazyFrame):
            episode.data = episode.data.collect()
        if isinstance(episode.semantic_snapshot, pl.LazyFrame):
            episode.semantic_snapshot = episode.semantic_snapshot.collect()

        identity = RobotIdentity(episode.robot_name) if episode.robot_name else None
        waiting = self._time_waiting_at_doors(events, end_time_ns)
        rides = self._elevator_rides(events)
        waiting_robot = None
        rides_robot = None
        if identity is not None:
            if self._has_kind(events, "door") and (waiting == 0.0 or _members_recorded(events, "door")):
                waiting_robot = self._time_waiting_at_doors_robot(events, end_time_ns, identity)
            if self._has_kind(events, "elevator") and (rides == 0 or _members_recorded(events, "elevator")):
                rides_robot = self._elevator_rides_robot(events, identity)

        return {
            "time_waiting_at_doors": waiting,
            "elevator_rides": rides,
            "time_waiting_at_doors_robot": waiting_robot,
            "elevator_rides_robot": rides_robot,
        }

    @staticmethod
    def _has_kind(events: pl.DataFrame, kind: str) -> bool:
        return len(events.filter(pl.col("kind") == kind)) > 0

    def _time_waiting_at_doors_robot(self, events: pl.DataFrame, end_time_ns: int | None, identity: RobotIdentity) -> float:
        door_events = events.filter((pl.col("kind") == "door") & pl.col("field").is_in(["state", "triggered", "members"])).sort("time_ns")

        total_s = 0.0
        for _entity, group in door_events.group_by("entity"):
            state = "closed"
            triggered = False
            robot_in = False
            prev_time_ns = None

            for row in group.sort("time_ns").iter_rows(named=True):
                t = row["time_ns"]
                if prev_time_ns is not None and triggered and robot_in and state != "open":
                    total_s += (t - prev_time_ns) / 1e9

                if row["field"] == "state":
                    state = row["current"]
                elif row["field"] == "triggered":
                    triggered = _parse_bool(row["current"])
                else:
                    robot_in = identity.any_of(parse_members(row["current"]))
                prev_time_ns = t

            if end_time_ns is not None and prev_time_ns is not None and end_time_ns > prev_time_ns and triggered and robot_in and state != "open":
                total_s += (end_time_ns - prev_time_ns) / 1e9

        return total_s

    def _elevator_rides_robot(self, events: pl.DataFrame, identity: RobotIdentity) -> int:
        window_ns = int(ARRIVAL_MEMBER_WINDOW_S * 1e9)
        rides = 0
        for _entity, group in events.filter(pl.col("kind") == "elevator").group_by("entity"):
            g = group.sort("time_ns")
            arrivals = g.filter((pl.col("field") == "just_arrived") & (pl.col("current") == "true") & (pl.col("previous") != "true"))
            members = g.filter(pl.col("field") == "members")
            if len(arrivals) == 0 or len(members) == 0:
                continue
            m_times = members["time_ns"].to_numpy()
            m_robot = [identity.any_of(parse_members(v)) for v in members["current"].to_list()]
            for t in arrivals["time_ns"].to_list():
                idx = max(int(np.searchsorted(m_times, t, side="right")) - 1, 0)
                hi = int(np.searchsorted(m_times, t + window_ns, side="right"))
                if any(m_robot[idx : max(hi, idx + 1)]):
                    rides += 1
        return rides

    def _time_waiting_at_doors(self, events: pl.DataFrame, end_time_ns: int | None) -> float:
        door_events = events.filter((pl.col("kind") == "door") & pl.col("field").is_in(["state", "triggered"])).sort("time_ns")

        if len(door_events) == 0:
            return 0.0

        total_s = 0.0
        for _entity, group in door_events.group_by("entity"):
            state = "closed"
            triggered = False
            prev_time_ns = None

            for row in group.sort("time_ns").iter_rows(named=True):
                t = row["time_ns"]
                if prev_time_ns is not None and triggered and state != "open":
                    total_s += (t - prev_time_ns) / 1e9

                if row["field"] == "state":
                    state = row["current"]
                else:
                    triggered = _parse_bool(row["current"])
                prev_time_ns = t

            if end_time_ns is not None and prev_time_ns is not None and end_time_ns > prev_time_ns and triggered and state != "open":
                total_s += (end_time_ns - prev_time_ns) / 1e9

        return total_s

    def _elevator_rides(self, events: pl.DataFrame) -> int:
        elevator_events = events.filter((pl.col("kind") == "elevator") & pl.col("field").is_in(["occupants", "just_arrived"])).sort("time_ns")

        if len(elevator_events) == 0:
            return 0

        rides = 0
        for _entity, group in elevator_events.group_by("entity"):
            occupants = 0.0
            for row in group.sort("time_ns").iter_rows(named=True):
                if row["field"] == "occupants":
                    occupants = _parse_float(row["current"])
                elif _parse_bool(row["current"]) and occupants > 0:
                    rides += 1

        return rides
