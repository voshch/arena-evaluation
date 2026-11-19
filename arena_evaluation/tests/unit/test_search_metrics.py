from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.search import SearchMetricsCalculator
from arena_evaluation.processing.metrics.search.search_metrics import done_seconds, since_reset
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _states(rows: list[tuple[float, str, int, float, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [int(t * SEC) for t, *_ in rows],
            "phase": [row[1] for row in rows],
            "stops": [row[2] for row in rows],
            "listened_s": [row[3] for row in rows],
            "goals_sent": [row[4] for row in rows],
            "goals_failed": [row[5] for row in rows],
            "calls": [row[6] for row in rows],
        }
    )


EPISODE = _states(
    [
        (0.0, "drive", 9, 30.0, 12, 1, 5),
        (0.5, "select", 0, 0.0, 0, 0, 0),
        (1.0, "drive", 0, 0.0, 1, 0, 0),
        (3.0, "listen", 1, 0.0, 1, 0, 1),
        (5.0, "drive", 1, 2.0, 2, 0, 2),
        (8.0, "done", 1, 2.0, 2, 1, 2),
        (12.0, "drive", 1, 2.0, 3, 1, 3),
        (13.0, "done", 1, 2.0, 3, 1, 3),
        (15.0, "done", 1, 2.0, 3, 1, 3),
    ]
)


def _episode(topics: dict[str, pl.DataFrame]) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(episode_id=0, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[0.0, 0.0], topics=topics)


def test_the_latched_state_of_the_episode_before_is_dropped() -> None:
    states = since_reset(EPISODE)
    assert states["time_ns"][0] == int(0.5 * SEC)
    assert states.height == EPISODE.height - 1


def test_a_driver_that_never_reset_keeps_every_state() -> None:
    states = EPISODE.filter(pl.col("goals_sent") > 0)
    assert since_reset(states).height == states.height


def test_time_in_done_lasts_until_the_next_state_and_the_last_state_adds_nothing() -> None:
    assert done_seconds(since_reset(EPISODE)) == pytest.approx(4.0 + 2.0)


def test_the_calculator_reports_the_last_counters_and_the_time_in_done() -> None:
    out = SearchMetricsCalculator(RobotParams()).calculate(_episode({"search_state": EPISODE}), {})
    assert out == {
        "search_stops": 1,
        "search_listened_s": 2.0,
        "search_goals_sent": 3,
        "search_goals_failed": 1,
        "search_calls": 3,
        "search_done_s": pytest.approx(6.0),
    }
    assert set(out) == set(SearchMetricsCalculator.output_keys())


@pytest.mark.parametrize("topics", [{}, {"search_state": EPISODE.clear()}])
def test_an_episode_without_search_states_reports_none(topics: dict[str, pl.DataFrame]) -> None:
    out = SearchMetricsCalculator(RobotParams()).calculate(_episode(topics), {})
    assert out == dict.fromkeys(SearchMetricsCalculator.output_keys())
