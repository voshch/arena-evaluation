from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")

import polars as pl
import rosbag2_py
from arena_humansim_msgs.msg import AgentFrame, AgentState, AgentStates
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader

_TOPIC = "/arena/env_0/arena_env_node/agent_states"
_PHYSICS = "pedestrian_agent_states.parquet"

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
    msg.desired_velocity = [0.0, 0.0, 0.0]
    msg.radius = [0.0, 0.0, 0.0]
    msg.policy_idx = [-1, -1, -1]
    msg.animation_state = [0, 0, 0]
    msg.interaction_id = [-1, -1, -1]
    msg.interaction_type = [0, 0, 0]
    msg.gait_phase = [0.0, 0.0, 0.0]
    msg.gait_cadence = [0.0, 0.0, 0.0]
    return msg


def _physics_frame(tmp_path: pathlib.Path, name: str, type_name: str, msg: object) -> pl.DataFrame:
    bag = tmp_path / name
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, type_name, "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 2_000_000_000)
    writer.close()
    out_dir = tmp_path / f"{name}_out"
    MCAPReader(bag).read(out_dir)
    return pl.read_parquet(out_dir / _PHYSICS)


def test_agent_states_nested_and_flat_layouts_read_into_identical_humans(tmp_path: pathlib.Path) -> None:
    nested = _physics_frame(tmp_path, "nested", "arena_humansim_msgs/msg/AgentStates", _nested())
    flat = _physics_frame(tmp_path, "flat", "arena_humansim_msgs/msg/AgentFrame", _flat())

    assert nested.equals(flat)
    assert flat.select("ped_id", "ped_name", "x", "y", "yaw", "vx", "vy").rows() == [
        ("Pedestrian_3", "3", 1.5, -2.0, 0.75, 0.3, 0.4),
        ("Pedestrian_7", "7", -0.5, 3.25, -1.5, -0.2, 0.1),
    ]


def _robot_only() -> AgentFrame:
    msg = AgentFrame()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 1
    msg.agent_id, msg.kind, msg.x, msg.y, msg.theta, msg.vx, msg.vy = [99], [AgentState.KIND_ROBOT], [4.0], [4.0], [0.0], [1.0], [0.0]
    msg.desired_velocity = [0.0]
    msg.radius = [0.0]
    msg.policy_idx = [-1]
    msg.animation_state = [0]
    msg.interaction_id = [-1]
    msg.interaction_type = [0]
    msg.gait_phase = [0.0]
    msg.gait_cadence = [0.0]
    return msg


def _flat_with_interactions() -> AgentFrame:
    msg = _flat()
    msg.animation_state = [1, 0, 2]
    msg.interaction_id = [4, -1, -1]
    msg.interaction_type = [10, 0, 0]
    return msg


