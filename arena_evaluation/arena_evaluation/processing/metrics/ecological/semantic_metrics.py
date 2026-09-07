from __future__ import annotations
import typing

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _reconstruct_events

from arena_evaluation.storage.schemas import AlignedEpisodeBundle


def _parse_bool(token: str) -> bool:
    return token.strip().lower() in ("true", "1")


def _parse_float(token: str) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        return 0.0


class SemanticInteractionMetricsCalculator(BaseMetricCalculator):
    """Computes door waiting time and elevator ride metrics from semantic snapshots."""

    NAME = "semantic_interaction_metrics"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = ["semantic_snapshot"]

    UNITS = {
        "time_waiting_at_doors": "s",
        "elevator_rides": "",
    }

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "time_waiting_at_doors",
            "elevator_rides",
        ]

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:

        events = _reconstruct_events(episode.semantic_snapshot)
        if len(events) == 0:
            return {"time_waiting_at_doors": 0.0, "elevator_rides": 0}

        end_time_ns = None
        if episode.data is not None and "time_ns" in episode.data.columns and len(episode.data) > 0:
            end_time_ns = int(episode.data["time_ns"].max())

        if isinstance(episode.data, pl.LazyFrame):
            episode.data = episode.data.collect()
        if isinstance(episode.semantic_snapshot, pl.LazyFrame):
            episode.semantic_snapshot = episode.semantic_snapshot.collect()

        return {
            "time_waiting_at_doors": self._time_waiting_at_doors(events, end_time_ns),
            "elevator_rides": self._elevator_rides(events),
        }

    def _time_waiting_at_doors(self, events: pl.DataFrame, end_time_ns: int | None) -> float:
        door_events = events.filter(
            (pl.col("kind") == "door") & pl.col("field").is_in(["state", "triggered"])
        ).sort("time_ns")

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

            if end_time_ns is not None and prev_time_ns is not None and end_time_ns > prev_time_ns \
                    and triggered and state != "open":
                total_s += (end_time_ns - prev_time_ns) / 1e9

        return total_s

    def _elevator_rides(self, events: pl.DataFrame) -> int:
        elevator_events = events.filter(
            (pl.col("kind") == "elevator") & pl.col("field").is_in(["occupants", "just_arrived"])
        ).sort("time_ns")

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
