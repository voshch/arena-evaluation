"""A repeated telemetry row key is one contract violation; the sink keeps the last recorded row."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import EPISODE_RECORD, TF_MESSAGE, _transform, _write_bag

SEC = 1_000_000_000


def _episode_record(outcome_state: int) -> dict:
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
        "conditions": "",
        "phases": "",
    }


def test_a_repeated_row_key_is_written_once_with_the_later_row(
    tmp_path: pathlib.Path,
) -> None:
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/state/episode", EPISODE_RECORD, _episode_record(0), 1 * SEC),
            ("/env_0/state/episode", EPISODE_RECORD, _episode_record(2), 1 * SEC),
            (
                "/tf",
                TF_MESSAGE,
                {
                    "transforms": [
                        _transform("map", "env_0/jackal/base_link", 1.0, 1.0),
                        _transform("map", "env_0/jackal/base_link", 2.0, 2.0),
                    ]
                },
                2 * SEC,
            ),
        ],
    )

    MCAPReader(bag).read(tmp_path / "topics")

    episodes = pl.read_parquet(tmp_path / "topics" / "episode_lifecycle.parquet")
    assert episodes.height == 1
    assert episodes["outcome_state"].to_list() == [2]
    transforms = pl.read_parquet(tmp_path / "topics" / "tf.parquet")
    assert transforms.height == 1
    assert transforms["tx"].to_list() == [2.0]
