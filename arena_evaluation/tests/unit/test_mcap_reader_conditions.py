import json
import pathlib

import polars as pl
import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import EPISODE_RECORD, _write_bag


def _record(outcome_state: int, conditions: str) -> dict:
    return {
        "episode_id": 1,
        "world": "test_world",
        "seed": 42,
        "task_layer": "standard",
        "modes": [],
        "robots": ["jackal"],
        "outcome_state": outcome_state,
        "outcome_info": "",
        "goal_uuid": "",
        "goal_dist_start": 5.0,
        "goal_dist_min": 0.1,
        "path_length": 6.2,
        "start_time": {"sec": 1, "nanosec": 0},
        "integrity": [],
        "params": [],
        "human_params": [],
        "conditions": conditions,
        "phases": "",
    }


def test_reader_keeps_episode_conditions(tmp_path: pathlib.Path):
    """Smoke run regression: conditions were dropped, so condition_success was always None."""
    clauses = json.dumps([{"op": "never", "p": "robot in restricted", "q": None, "text": "never robot in restricted"}])
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/state/episode", EPISODE_RECORD, _record(0, ""), 1),
            ("/env_0/state/episode", EPISODE_RECORD, _record(2, clauses), 2),
        ],
    )

    MCAPReader(bag).read(tmp_path / "raw")

    episodes = pl.read_parquet(tmp_path / "raw" / "episode_lifecycle.parquet")
    assert episodes["conditions_yaml"].to_list() == [None, clauses]
