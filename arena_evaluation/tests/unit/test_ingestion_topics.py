"""Unit tests for arena_evaluation.ingestion.topics (topic registry)."""

from __future__ import annotations

import dataclasses

import pytest

for _pkg in (
    "geometry_msgs.msg",
    "sensor_msgs.msg",
    "nav_msgs.msg",
    "std_msgs.msg",
    "tf2_msgs.msg",
    "arena_people_msgs.msg",
    "arena_humansim_msgs.msg",
    "arena_runtime_msgs.msg",
    "arena_robots_msgs.msg",
    "nav2_msgs.msg",
    "rosidl_runtime_py.utilities",
):
    pytest.importorskip(_pkg)

from geometry_msgs.msg import Twist  # noqa: E402
from arena_runtime_msgs.msg import RecordedTopic  # noqa: E402

from arena_evaluation.ingestion.topics import (  # noqa: E402
    TopicDefinition,
)


# (key, default name with empty namespaces, msg type name, throttled, transient_local, rate_hz)
_TOPIC_EXPECTATIONS = [
    ("cmd_vel", "/cmd_vel", "Twist", True, False, 20.0),
    ("scan", "/scan", "LaserScan", True, False, 10.0),
    ("lidar", "/lidar", "LaserScan", True, False, 10.0),
    ("joint_states", "/joint_states", "JointState", True, False, 20.0),
    ("plan", "/plan", "Path", False, False, 20.0),
    ("goal_pose", "/goal_pose", "PoseStamped", False, False, 20.0),
    ("task_pose", "/task_pose", "PoseStamped", False, False, 20.0),
    ("initialpose", "/initialpose", "PoseWithCovarianceStamped", False, False, 20.0),
    ("tf", "/tf", "TFMessage", True, False, 20.0),
    ("tf_static", "/tf_static", "TFMessage", False, True, 20.0),
    ("tf_humans", "/humans/tf", "TFMessage", True, False, 20.0),
    ("peds", "/arena_peds", "Pedestrians", True, False, 20.0),
    ("agent_states", "/agent_states", "AgentFrame", True, False, 20.0),
    ("agent_meta", "/agent_meta", "AgentMeta", False, True, 20.0),
    ("interactions", "/interactions", "InteractionFrame", False, True, 20.0),
    ("episode_record", "/state/episode", "EpisodeRecord", False, True, 20.0),
    ("robots_fleet", "/state/robots", "RobotFleet", False, True, 20.0),
    ("semantic_snapshot", "/state/semantics", "SemanticSnapshot", False, True, 20.0),
    ("collision_events", "/collision_events", "CollisionEvents", False, False, 20.0),
    ("power", "/power_publisher/power", "Power", True, False, 10.0),
    ("energy", "/power_publisher/energy", "Energy", True, False, 10.0),
    ("acoustics", "/acoustics", "Acoustics", True, False, 10.0),
    ("characterization_phase", "/characterization_phase", "String", False, False, 20.0),
    (
        "characterization_schedule",
        "/characterization_schedule",
        "String",
        False,
        True,
        20.0,
    ),
    (
        "collision_monitor_state",
        "/collision_monitor_state",
        "CollisionMonitorState",
        False,
        False,
        20.0,
    ),
    ("map", "/map", "OccupancyGrid", False, True, 20.0),
    ("door_mask", "/door_mask", "OccupancyGrid", False, True, 20.0),
]


def test_topic_definition_dataclass_defaults_and_overrides():
    assert dataclasses.is_dataclass(TopicDefinition)
    td = TopicDefinition(name_template="/x", msg_type=Twist)
    assert td.throttled is True
    assert td.throttle_rate_hz == 20.0
    assert td.qos_transient_local is False
    assert td.reliable is False
    assert td.depth == 0

    over = TopicDefinition(
        "/y", Twist, throttled=False, throttle_rate_hz=30.0, qos_transient_local=True
    )
    assert over.throttled is False
    assert over.throttle_rate_hz == 30.0
    assert over.qos_transient_local is True


def _row(topic: str, msg_type: str = "std_msgs/msg/String", **flags) -> RecordedTopic:
    return RecordedTopic(key="row", topic=topic, msg_type=msg_type, **flags)
