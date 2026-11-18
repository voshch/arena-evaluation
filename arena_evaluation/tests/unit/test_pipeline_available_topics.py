from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.pipeline import _calculator_topic_names
from arena_evaluation.storage.schemas import TopicBundle


@pytest.mark.parametrize(
    ("recorded_field", "calculator_topic"),
    [
        ("pedestrian_arena_peds", "pedestrian_arena_peds"),
        ("pedestrian_agent_states", "pedestrian_agent_states"),
        ("robot_tracks", "robot_tracks"),
    ],
)
def test_recorded_table_is_listed_under_the_topic_name_its_calculators_read(
    recorded_field: str, calculator_topic: str
) -> None:
    bundle = TopicBundle(**{recorded_field: pl.DataFrame({"t_ns": [0]})})

    names = _calculator_topic_names(bundle)

    assert calculator_topic in names
    assert recorded_field not in names or recorded_field == calculator_topic


def test_bundle_without_tables_lists_no_topics() -> None:
    assert _calculator_topic_names(TopicBundle()) == set()
