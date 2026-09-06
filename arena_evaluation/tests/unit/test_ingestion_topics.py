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
    "task_generator_msgs.msg",
    "arena_robots_msgs.msg",
    "nav2_msgs.msg",
):
    pytest.importorskip(_pkg)

from geometry_msgs.msg import Twist
from nav2_msgs.msg import CollisionMonitorState
from sensor_msgs.msg import JointState

from arena_evaluation.ingestion.topics import TopicDefinition, get_topics


# (key, default name with empty namespaces, msg type name, throttled, transient_local)
_TOPIC_EXPECTATIONS = [
    ("cmd_vel", "/cmd_vel", "Twist", True, False),
    ("scan", "/scan", "LaserScan", True, False),
    ("lidar", "/lidar", "LaserScan", True, False),
    ("joint_states", "/joint_states", "JointState", True, False),
    ("plan", "/plan", "Path", False, False),
    ("goal_pose", "/goal_pose", "PoseStamped", False, False),
    ("initialpose", "/initialpose", "PoseWithCovarianceStamped", False, False),
    ("tf", "/tf", "TFMessage", True, False),
    ("tf_static", "/tf_static", "TFMessage", False, True),
    ("peds", "/arena_peds", "Pedestrians", True, False),
    ("agent_states", "/agent_states", "AgentStates", True, False),
    ("episode_record", "/state/episode", "EpisodeRecord", False, True),
    ("robots_fleet", "/state/robots", "RobotFleet", False, True),
    ("semantic_snapshot", "/state/semantics", "SemanticSnapshot", False, True),
    ("collision_events", "/collision_events", "CollisionEvents", False, False),
    ("power", "/power_publisher/power", "Power", True, False),
    ("energy", "/power_publisher/energy", "Energy", True, False),
    ("acoustics", "/acoustics", "Acoustics", True, False),
    ("characterization_phase", "/characterization_phase", "String", False, False),
    ("characterization_schedule", "/characterization_schedule", "String", False, True),
    ("collision_monitor_state", "/collision_monitor_state", "CollisionMonitorState", False, False),
    ("audio_raw", "/audio/raw_array", "AudioFrame", False, False),
    ("audio_rendered", "/audio/headphones/stereo", "AudioFrame", False, False),
    ("map", "/map", "OccupancyGrid", False, True),
    ("door_mask", "/door_mask", "OccupancyGrid", False, True),
]


def test_get_topics_returns_expected_key_set():
    topics = get_topics("")
    assert set(topics) == {row[0] for row in _TOPIC_EXPECTATIONS}


def test_get_topics_default_namespaces():
    topics = get_topics("")
    for key, name, *_ in _TOPIC_EXPECTATIONS:
        assert topics[key].name_template == name


@pytest.mark.parametrize("key, name, msg_name, throttled, qos_transient_local", _TOPIC_EXPECTATIONS)
def test_get_topics_topic_definition_properties(key, name, msg_name, throttled, qos_transient_local):
    topics = get_topics("")
    td = topics[key]
    assert isinstance(td, TopicDefinition)
    assert td.name_template == name
    assert td.msg_type.__name__ == msg_name
    assert td.throttled is throttled
    assert td.qos_transient_local is qos_transient_local
    assert td.throttle_rate_hz == 10.0


def test_get_topics_namespace_and_parent_namespace():
    # parent_namespace is the recorder's own namespace, the task generator node's
    # fully qualified name. Pedestrians are published one level up, by the env.
    topics = get_topics("env_0", "arena_0/task_generator_node")
    assert topics["cmd_vel"].name_template == "/env_0/cmd_vel"
    assert topics["joint_states"].name_template == "/env_0/joint_states"
    assert topics["plan"].name_template == "/env_0/plan"
    assert topics["goal_pose"].name_template == "/env_0/goal_pose"
    assert topics["initialpose"].name_template == "/arena_0/task_generator_node/initialpose"
    assert topics["peds"].name_template == "/arena_0/arena_peds"
    assert topics["agent_states"].name_template == "/arena_0/task_generator_node/agent_states"
    assert topics["episode_record"].name_template == "/arena_0/task_generator_node/state/episode"
    assert topics["robots_fleet"].name_template == "/arena_0/task_generator_node/state/robots"
    assert topics["semantic_snapshot"].name_template == "/arena_0/task_generator_node/state/semantics"
    assert topics["power"].name_template == "/env_0/power_publisher/power"
    assert topics["energy"].name_template == "/env_0/power_publisher/energy"
    assert topics["collision_events"].name_template == "/env_0/collision_events"


def test_get_topics_tf_topics_are_global():
    topics = get_topics("env_0", "arena_0")
    assert topics["tf"].name_template == "/tf"
    assert topics["tf_static"].name_template == "/tf_static"


def test_get_topics_slash_namespaces_treated_as_empty():
    # Suspected bug in topics.py: namespace="/" is intended to normalize to ""
    # (see the `if ns == "/": ns = ""` guards), but `f"/{namespace}"` yields
    # "//" for namespace="/", so those guards never fire and names gain a
    # doubled slash. We assert the actual behavior so a future fix is noticed.
    assert get_topics("/")["cmd_vel"].name_template == "///cmd_vel"
    assert get_topics("", "/")["initialpose"].name_template == "///initialpose"


def test_get_topics_parent_only_namespace():
    topics = get_topics("", "arena_0")
    assert topics["cmd_vel"].name_template == "/cmd_vel"
    assert topics["goal_pose"].name_template == "/goal_pose"
    assert topics["initialpose"].name_template == "/arena_0/initialpose"


def test_get_topics_msg_type_identity():
    topics = get_topics("")
    assert topics["cmd_vel"].msg_type is Twist
    assert topics["joint_states"].msg_type is JointState
    assert topics["collision_monitor_state"].msg_type is CollisionMonitorState


def test_get_topics_returns_fresh_instances():
    a = get_topics("env_0")
    b = get_topics("env_0")
    assert a == b
    assert a is not b
    assert a["cmd_vel"] is not b["cmd_vel"]


def test_topic_definition_dataclass_defaults_and_overrides():
    assert dataclasses.is_dataclass(TopicDefinition)
    td = TopicDefinition(name_template="/x", msg_type=Twist)
    assert td.throttled is True
    assert td.throttle_rate_hz == 10.0
    assert td.qos_transient_local is False

    over = TopicDefinition("/y", Twist, throttled=False, throttle_rate_hz=30.0, qos_transient_local=True)
    assert over.throttled is False
    assert over.throttle_rate_hz == 30.0
    assert over.qos_transient_local is True
