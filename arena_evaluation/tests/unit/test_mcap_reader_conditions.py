import json
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import PARAM, PARAM_VALUE, _msgdef, _write_bag

EPISODE_RECORD = ("task_generator_msgs/msg/EpisodeRecord", _msgdef("uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nrcl_interfaces/Parameter[] robots_params\nstring conditions", PARAM, PARAM_VALUE))


def _record(outcome_state: int, conditions: str) -> dict:
    return {"episode_id": 1, "outcome_state": outcome_state, "outcome_info": "", "goal_uuid": "", "robots_params": [], "conditions": conditions}


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

    er = pl.read_parquet(tmp_path / "raw" / "env_0" / "episode_record.parquet")
    assert er["conditions"].to_list() == ["", clauses]
