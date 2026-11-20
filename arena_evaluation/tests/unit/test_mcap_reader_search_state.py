"""Round trip: the search driver state recorded through rosbag2 comes back as a table of the robot that published it."""

from __future__ import annotations

import json
import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")

import polars as pl
from rclpy.serialization import serialize_message
from std_msgs.msg import String

from arena_evaluation.processing.mcap_reader import MCAPReader

TOPIC = "/arena/env_0/arena_env_node/turtlebot/hearing/search_state"
SEC = 1_000_000_000


def _state(phase: str, stops: int, listened_s: float) -> String:
    return String(data=json.dumps({"phase": phase, "goal": None, "stops": stops, "listened_s": listened_s, "goals_sent": 2, "goals_failed": 0, "calls": 1, "nav": "idle"}))


def _mcap(tmp_path: pathlib.Path, messages: list[tuple[String, int]]) -> MCAPReader:
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(tmp_path / "bag"), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=TOPIC, type="std_msgs/msg/String", serialization_format="cdr"))
    for msg, t in messages:
        writer.write(TOPIC, serialize_message(msg), t)
    del writer
    return MCAPReader(next((tmp_path / "bag").glob("*.mcap")))


def _read(tmp_path: pathlib.Path, messages: list[tuple[String, int]]) -> pathlib.Path:
    out = tmp_path / "topics"
    _mcap(tmp_path, messages).read(out)
    return out


def test_search_states_land_under_the_robot_not_under_hearing(tmp_path: pathlib.Path) -> None:
    out = _read(tmp_path, [(_state("drive", 0, 0.0), 1 * SEC), (_state("listen", 1, 1.5), 2 * SEC)])
    assert (out / "env_0_turtlebot" / "search_state.parquet").is_file()
    assert not (out / "env_0_hearing").exists()
    table = pl.read_parquet(out / "env_0_turtlebot" / "search_state.parquet").sort("time_ns")
    assert table.columns == ["time_ns", "phase", "stops", "listened_s", "goals_sent", "goals_failed", "calls"]
    assert table["phase"].to_list() == ["drive", "listen"]
    assert table["stops"].to_list() == [0, 1]
    assert table["listened_s"].to_list() == [0.0, 1.5]


def test_a_payload_that_is_no_search_state_is_skipped(tmp_path: pathlib.Path) -> None:
    out = _read(tmp_path, [(String(data="not json"), 1 * SEC), (String(data=json.dumps({"phase": "drive"})), 2 * SEC), (_state("done", 3, 6.0), 3 * SEC)])
    table = pl.read_parquet(out / "env_0_turtlebot" / "search_state.parquet")
    assert table["phase"].to_list() == ["done"]


def test_the_robot_bundle_exposes_the_search_states(tmp_path: pathlib.Path) -> None:
    bundles = _mcap(tmp_path, [(_state("drive", 0, 0.0), 1 * SEC)]).read(tmp_path / "topics")
    assert "search_state" in bundles["env_0_turtlebot"].available()
