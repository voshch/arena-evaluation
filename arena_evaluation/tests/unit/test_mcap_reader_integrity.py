from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import polars as pl
from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import KEY_VALUE, PARAM, PARAM_VALUE, TIME, _msgdef

SEC = 1_000_000_000
NAME = "arena_runtime_msgs/msg/EpisodeRecord"

_BASE_FIELDS = """uint32 episode_id
string world
int64 seed
string task_layer
diagnostic_msgs/KeyValue[] modes
string[] robots
uint8 outcome_state
string outcome_info
string goal_uuid
float32 goal_dist_start
float32 goal_dist_min
float32 path_length
builtin_interfaces/Time start_time
string conditions
string phases
rcl_interfaces/Parameter[] params
rcl_interfaces/Parameter[] human_params"""

CURRENT = _msgdef(_BASE_FIELDS + "\nstring[] integrity", KEY_VALUE, PARAM, PARAM_VALUE, TIME)


def _bag(path: pathlib.Path, records: list[tuple[str, str, dict, int]]) -> pathlib.Path:
    with path.open("wb") as fh:
        writer = Writer(fh)
        schemas = {}
        for topic, text, msg, log_time in records:
            if text not in schemas:
                schemas[text] = writer.register_msgdef(NAME, text)
            writer.write_message(topic, schemas[text], msg, log_time=log_time)
        writer.finish()
    return path


def _record(state: int, **fields: object) -> dict:
    return {
        "episode_id": 1,
        "world": "test_world",
        "seed": 42,
        "task_layer": "standard",
        "modes": [],
        "robots": ["jackal"],
        "outcome_state": state,
        "outcome_info": "",
        "goal_uuid": "",
        "goal_dist_start": 5.0,
        "goal_dist_min": 0.1,
        "path_length": 6.2,
        "start_time": {"sec": 1, "nanosec": 0},
        "conditions": "",
        "phases": "",
        "params": [],
        "human_params": [],
        **fields,
    }


@pytest.fixture
def extracted(tmp_path: pathlib.Path) -> pathlib.Path:
    bag = _bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/state/episode", CURRENT, _record(1, integrity=[]), 1 * SEC),
            ("/env_0/state/episode", CURRENT, _record(1, integrity=["robot jackal spawned by hand"]), 2 * SEC),
            ("/env_0/state/episode", CURRENT, _record(2, integrity=["robot jackal spawned by hand", "task parameters changed mid-episode"]), 3 * SEC),
        ],
    )
    out = tmp_path / "topics"
    MCAPReader(bag).read(out)
    return out


def test_each_flag_of_each_record_is_kept_as_it_accrues(extracted: pathlib.Path) -> None:
    flags = pl.read_parquet(extracted / "episode_integrity.parquet")
    assert flags.select("t_ns", "env_id", "flag_idx", "flag").rows() == [
        (2 * SEC, "env_0", 0, "robot jackal spawned by hand"),
        (3 * SEC, "env_0", 0, "robot jackal spawned by hand"),
        (3 * SEC, "env_0", 1, "task parameters changed mid-episode"),
    ]
