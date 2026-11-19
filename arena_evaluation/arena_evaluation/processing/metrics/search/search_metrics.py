"""Search driver counters of one episode, read from the arena_hearing `search_state` topic."""

from __future__ import annotations

import typing

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

COUNTERS = {
    "stops": "search_stops",
    "listened_s": "search_listened_s",
    "goals_sent": "search_goals_sent",
    "goals_failed": "search_goals_failed",
    "calls": "search_calls",
}
PHASE_DONE = "done"


def since_reset(states: pl.DataFrame) -> pl.DataFrame:
    """The states from the driver's reset on, without the latched state the episode before left."""
    fresh = ((states["stops"] == 0) & (states["goals_sent"] == 0)).arg_true()
    return states[int(fresh[0]) :] if len(fresh) else states


def done_seconds(states: pl.DataFrame) -> float:
    """Seconds the driver spent in phase done, each state lasting until the next one."""
    t_s = states["time_ns"].cast(pl.Float64) * 1e-9
    held = t_s.shift(-1, fill_value=t_s[-1]) - t_s
    return float(held.filter(states["phase"] == PHASE_DONE).sum())


class SearchMetricsCalculator(BaseMetricCalculator):
    """Stops, listening time, goals, heard calls and time without a frontier of a search driver episode."""

    NAME = "search"
    CATEGORY = "search"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = []
    REQUIRED_TOPICS = ["search_state"]

    UNITS = {
        "search_stops": "",
        "search_listened_s": "s",
        "search_goals_sent": "",
        "search_goals_failed": "",
        "search_calls": "",
        "search_done_s": "s",
    }
    PRIMARY_OUTPUTS = ["search_stops", "search_listened_s"]
    OUTPUT_DIRECTIONS = {"search_goals_failed": "lower", "search_done_s": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [*COUNTERS.values(), "search_done_s"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        states = (episode.topics or {}).get("search_state")
        if states is None or states.is_empty():
            return dict.fromkeys(self.output_keys())
        states = since_reset(states.sort("time_ns"))
        last = states.row(-1, named=True)
        return {**{key: last[column] for column, key in COUNTERS.items()}, "search_done_s": done_seconds(states)}
