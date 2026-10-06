from __future__ import annotations

import pathlib

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")

import pyarrow.parquet as pq
from arena_humansim_msgs.msg import AgentFrame, AgentState, AgentStates
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader

_TOPIC = "/arena/env_0/task_generator_node/agent_states"

# (agent_id, kind, x, y, theta, vx, vy)
_AGENTS = [
    (3, AgentState.KIND_HUMAN, 1.5, -2.0, 0.75, 0.3, 0.4),
    (99, AgentState.KIND_ROBOT, 4.0, 4.0, 0.0, 1.0, 0.0),
    (7, AgentState.KIND_HUMAN, -0.5, 3.25, -1.5, -0.2, 0.1),
]


def _nested() -> AgentStates:
    msg = AgentStates()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 2
    for agent_id, kind, x, y, theta, vx, vy in _AGENTS:
        agent = AgentState(agent_id=agent_id, kind=kind)
        agent.pose.x, agent.pose.y, agent.pose.theta = x, y, theta
        agent.velocity.x, agent.velocity.y = vx, vy
        msg.agents.append(agent)
    return msg


def _flat() -> AgentFrame:
    msg = AgentFrame()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 2
    msg.agent_id, msg.kind, msg.x, msg.y, msg.theta, msg.vx, msg.vy = (list(column) for column in zip(*_AGENTS, strict=True))
    return msg


def _peds_rows(tmp_path: pathlib.Path, name: str, type_name: str, msg: object) -> list[dict]:
    bag = tmp_path / name
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, type_name, "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 2_000_000_000)
    writer.close()
    out_dir = tmp_path / f"{name}_out"
    MCAPReader(bag).read(out_dir)
    return pq.read_table(out_dir / "env_0" / "peds.parquet").to_pylist()


def test_agent_states_nested_and_flat_layouts_read_into_identical_peds(tmp_path: pathlib.Path) -> None:
    nested = _peds_rows(tmp_path, "nested", "arena_humansim_msgs/msg/AgentStates", _nested())
    flat = _peds_rows(tmp_path, "flat", "arena_humansim_msgs/msg/AgentFrame", _flat())

    assert nested == flat
    assert flat == [
        {
            "time_ns": 2_000_000_000,
            "peds_frame_id": "map",
            "num_pedestrians": 2,
            "peds_names": ["3", "7"],
            "peds_positions": [1.5, -2.0, 0.0, -0.5, 3.25, 0.0],
            "peds_headings": [0.75, -1.5],
            "peds_twists": [0.3, 0.4, 0.0, -0.2, 0.1, 0.0],
        }
    ]
