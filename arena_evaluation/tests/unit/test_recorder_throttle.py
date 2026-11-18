"""Unit tests for the recorder's rate gate, its config contract, its tf stamping and its initial-pose attribution.

The gate and the config loader are exercised as plain objects, the callbacks, the
/tf writes and the initial-pose subscription through a bare node and a real bag.
"""

from __future__ import annotations

import io
import pathlib
import re
import threading
from collections.abc import Callable

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import rosbag2_py
from arena_runtime_msgs.msg import RobotFleet, RobotState
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from mcap.reader import make_reader
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from arena_evaluation.ingestion import recorder
from arena_evaluation.ingestion.recorder import DataRecorderNode
from arena_evaluation.ingestion.topics import TopicDefinition
from arena_evaluation.processing.telemetry.messages import robot_id_of, topic_parts

_CONVERTER = rosbag2_py.ConverterOptions(
    input_serialization_format="cdr", output_serialization_format="cdr"
)


def _bare_node(
    *,
    record_rates_hz: dict[str, float] | None = None,
    current_time: int | None = None,
) -> DataRecorderNode:
    """A DataRecorderNode carrying only the attributes the logic under test needs."""
    node = DataRecorderNode.__new__(DataRecorderNode)
    node.log_file = io.StringIO()
    node.log_file_path = None
    node.current_time = current_time
    node.is_shutting_down = False
    node.writer = None
    node.writer_lock = threading.Lock()
    node.current_episode_dir = None
    node.current_metadata = None
    node.recorded_topics = set()
    node.latched_topic_names = set()
    node._pre_episode_buffer = []
    node._pre_clock_buffer = []
    node._topic_registry = {}
    node.topics_metadata = {}
    node._write_success_count = 0
    node._write_drop_count = 0
    node._throttle_gates = {}
    node._tf_lock = threading.Lock()
    node._tf_last_written = {}
    node.record_rates_hz = record_rates_hz if record_rates_hz is not None else {}
    node.tf_frames = []
    node.env_ns_root = "/env_0"
    node.qos = object()
    node.latched_qos = object()
    return node


def _mcap_rows(path: pathlib.Path) -> list[tuple[str, bytes, int]]:
    with path.open("rb") as f:
        return [
            (channel.topic, message.data, message.log_time)
            for _, channel, message in make_reader(f).iter_messages()
        ]


def _bag(
    node: DataRecorderNode, uri: pathlib.Path
) -> Callable[[], list[tuple[str, bytes, int]]]:
    """Point the node at a real writer on uri, returning a closure that closes it and reads the bag back."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"), _CONVERTER)
    node.writer = writer

    def read() -> list[tuple[str, bytes, int]]:
        if node.writer is not None:
            node.writer.close()
            node.writer = None
        return [row for path in sorted(uri.glob("*.mcap")) for row in _mcap_rows(path)]

    return read


def _strings(rows: list[tuple[str, bytes, int]]) -> list[tuple[int, str]]:
    return [(ts, deserialize_message(data, String).data) for _, data, ts in rows]


def _transform(
    frame_id: str, child_frame_id: str, x: float, stamp_ns: int
) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    tf.header.stamp.sec = stamp_ns // 1_000_000_000
    tf.header.stamp.nanosec = stamp_ns % 1_000_000_000
    tf.transform.translation.x = x
    return tf


def _tf_message(*pairs: tuple[str, str, float, int]) -> TFMessage:
    return TFMessage(transforms=[_transform(*pair) for pair in pairs])


def _tf_writes(
    rows: list[tuple[str, bytes, int]],
) -> list[tuple[int, dict[str, float]]]:
    """(timestamp, {child_frame: x}) for every /tf message in the bag."""
    return [
        (
            ts,
            {
                transform.child_frame_id: transform.transform.translation.x
                for transform in deserialize_message(data, TFMessage).transforms
            },
        )
        for topic, data, ts in rows
        if topic == "tf"
    ]


def _initial_pose(frame_id: str) -> PoseWithCovarianceStamped:
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    return msg


def _fleet(*robots: tuple[str, str]) -> RobotFleet:
    fleet = RobotFleet()
    fleet.robots = [_robot_state(ns, model) for ns, model in robots]
    return fleet


def _robot_state(namespace: str, model: str) -> RobotState:
    state = RobotState()
    state.descriptor.ns = namespace
    state.descriptor.model = model
    return state


def _config_node(tmp_path: pathlib.Path, text: str) -> DataRecorderNode:
    config_dir = tmp_path / "share" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "data_recorder_config.yaml").write_text(text)
    node = _bare_node()
    node.base_dir = str(tmp_path / "share")
    return node


@pytest.fixture
def env_node(tmp_path: pathlib.Path):
    node = DataRecorderNode(
        namespace="env_0",
        parameter_overrides=[
            Parameter("record_data_dir", value=str(tmp_path / "episodes"))
        ],
    )
    yield node
    node.destroy_node()


def test_initialpose_is_subscribed_once_at_the_environment(env_node: DataRecorderNode):
    subscriptions = [sub for sub in env_node.subs if sub.topic.endswith("/initialpose")]
    assert [sub.topic for sub in subscriptions] == ["/env_0/initialpose"]
    assert subscriptions[0].raw is False

    env_node.robots_fleet_callback(_fleet(("jackal", "jackal")))

    assert [
        sub.topic for sub in env_node.subs if sub.topic.endswith("/initialpose")
    ] == ["/env_0/initialpose"]


def test_initialpose_is_written_under_the_robot_its_frame_names(
    env_node: DataRecorderNode,
):
    env_node.current_time = 5_000
    env_node._begin_episode(0)
    env_node.initialpose_callback(_initial_pose("env_0/jackal/base_link"))
    env_node._close_current_writer()

    rows = _mcap_rows(env_node.episodes_root / "episode_000" / "episode_000.mcap")
    assert [(topic, ts) for topic, _, ts in rows] == [
        ("env_0/jackal/initialpose", 5_000)
    ]
    assert (
        robot_id_of(topic_parts("env_0/jackal/initialpose"), "env_0") == "env_0_jackal"
    )


def test_initialpose_without_a_robot_frame_stays_at_the_environment(
    env_node: DataRecorderNode,
):
    env_node.current_time = 7_000
    env_node._begin_episode(0)
    env_node.initialpose_callback(_initial_pose("map"))
    env_node._close_current_writer()

    rows = _mcap_rows(env_node.episodes_root / "episode_000" / "episode_000.mcap")
    assert [(topic, ts) for topic, _, ts in rows] == [("env_0/initialpose", 7_000)]


def test_tf_pair_keeps_its_frame_family_interval(tmp_path: pathlib.Path):
    node = _bare_node(current_time=1_000_000_000)
    node.tf_frames = [(re.compile(r"_agent_\d+$"), 200.0)]
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/tf", TFMessage)

    for stamp_ns in (0, 100_000_000, 199_999_999, 200_000_000):
        node._tf_callback(_tf_message(("map", "l_arm_env_0_agent_2", 1.0, stamp_ns)))

    assert [ts for ts, _ in _tf_writes(read())] == [0, 200_000_000]


def test_a_refused_transform_does_not_shift_the_family_interval(
    tmp_path: pathlib.Path,
):
    node = _bare_node(current_time=1_000_000_000)
    node.tf_frames = [(re.compile(r"_agent_\d+$"), 200.0)]
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/tf", TFMessage)

    node._tf_callback(_tf_message(("map", "bone_env_0_agent_1", 1.0, 0)))
    node._tf_callback(_tf_message(("map", "bone_env_0_agent_1", 2.0, 199_999_999)))
    node._tf_callback(_tf_message(("map", "bone_env_0_agent_1", 3.0, 200_000_000)))

    assert _tf_writes(read()) == [
        (0, {"bone_env_0_agent_1": 1.0}),
        (200_000_000, {"bone_env_0_agent_1": 3.0}),
    ]


def test_min_interval_ns_rounds_a_rate_up_to_whole_nanoseconds():
    assert recorder.min_interval_ns(20.0) == 50_000_000
    assert recorder.min_interval_ns(10.0) == 100_000_000
    assert recorder.min_interval_ns(50.0) == 20_000_000
    assert recorder.min_interval_ns(3.0) == 333_333_334
    assert recorder.min_interval_ns(0.5) == 2_000_000_000
    assert recorder.min_interval_ns(0.0) == 0


@pytest.mark.parametrize(
    "delta_ns, accepted",
    [
        (0, False),
        (1, False),
        (49_999_999, False),
        (50_000_000, True),
        (50_000_001, True),
        (99_999_999, True),
        (100_000_000, True),
    ],
)
def test_a_20_hz_gate_opens_at_50_ms(delta_ns: int, accepted: bool):
    gate = recorder.RateGate(recorder.min_interval_ns(20.0), 1_000_000_000)
    assert gate.accept(1_000_000_000 + delta_ns) is accepted


def test_a_refused_sample_does_not_shift_the_gate_interval():
    gate = recorder.RateGate(recorder.min_interval_ns(20.0))
    assert gate.accept(0) is True
    assert gate.accept(49_999_999) is False
    assert gate.accept(50_000_000) is True
    assert gate.accept(99_999_999) is False
    assert gate.accept(100_000_000) is True


def test_a_zero_interval_passes_every_sample():
    gate = recorder.RateGate(recorder.min_interval_ns(0.0))
    assert [gate.accept(t_ns) for t_ns in (0, 0, 1)] == [True, True, True]


@pytest.mark.parametrize(
    "period_ns, messages, kept",
    [
        (1_000_000, 1000, 20),
        (10_000_000, 1000, 200),
    ],
)
def test_sample_count_of_1000_messages_through_a_20_hz_gate(
    period_ns: int, messages: int, kept: int
):
    """1000 messages 1 ms apart span 1 s and keep 20; 10 ms apart they span 10 s and keep 200."""
    gate = recorder.RateGate(recorder.min_interval_ns(20.0))
    assert sum(gate.accept(index * period_ns) for index in range(messages)) == kept


def test_an_override_of_the_exact_topic_name_wins_over_the_default():
    overrides = {"/env_0/front/scan": 5.0}
    assert recorder.resolve_rate_hz(overrides, "/env_0/front/scan", 10.0) == 5.0


def test_an_override_matches_no_substring_of_a_topic_name():
    overrides = {"/odom": 30.0}
    assert recorder.resolve_rate_hz(overrides, "/robot_1/odom_filtered", 50.0) == 50.0
    assert recorder.resolve_rate_hz(overrides, "/odom", 50.0) == 30.0


def test_an_absent_key_keeps_the_definition_rate():
    assert recorder.resolve_rate_hz({}, "/env_0/scan", 10.0) == 10.0


def test_a_config_still_carrying_record_frequencies_is_refused_with_the_hint(
    tmp_path: pathlib.Path,
):
    node = _config_node(tmp_path, "record_frequencies:\n  default: 20.0\n")
    with pytest.raises(ValueError, match="record_rates_hz"):
        node.read_config()


def test_a_record_rates_hz_mapping_is_read_as_floats(tmp_path: pathlib.Path):
    node = _config_node(tmp_path, 'record_rates_hz:\n  "/env_0/jackal/scan": 5\n')
    assert node.read_config() == {"record_rates_hz": {"/env_0/jackal/scan": 5.0}}


def test_a_negative_record_rate_is_refused(tmp_path: pathlib.Path):
    node = _config_node(tmp_path, 'record_rates_hz:\n  "/env_0/jackal/scan": -1.0\n')
    with pytest.raises(ValueError, match="record_rates_hz"):
        node.read_config()


def test_the_callback_honours_a_record_rates_hz_override_at_the_boundary(
    tmp_path: pathlib.Path,
):
    node = _bare_node(record_rates_hz={"/cmd_vel": 10.0})
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    callback = node._create_throttled_callback("/cmd_vel")

    for t_ns, payload in ((0, "a"), (99_999_999, "b"), (100_000_000, "c")):
        node.current_time = t_ns
        callback(String(data=payload))

    assert _strings(read()) == [(0, "a"), (100_000_000, "c")]


def test_a_rate_of_zero_passes_every_message(tmp_path: pathlib.Path):
    node = _bare_node(record_rates_hz={"/cmd_vel": 0.0})
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    callback = node._create_throttled_callback("/cmd_vel")

    for t_ns in (0, 0, 1, 2):
        node.current_time = t_ns
        callback(String(data=str(t_ns)))

    assert [ts for ts, _ in _strings(read())] == [0, 0, 1, 2]


def test_an_absent_key_keeps_the_20_hz_default(tmp_path: pathlib.Path):
    node = _bare_node()
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/joint_states", String)
    callback = node._create_throttled_callback("/joint_states")

    node.current_time = 0
    callback(String(data="a"))
    node.current_time = 50_000_000
    callback(String(data="b"))

    assert _strings(read()) == [(0, "a"), (50_000_000, "b")]


def test_a_definition_with_throttled_false_is_not_throttled(tmp_path: pathlib.Path):
    node = _bare_node(record_rates_hz={"/plan": 1.0})
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/plan", String)
    qos, callback = node._env_topic_binding(
        "plan", TopicDefinition("/plan", String, throttled=False)
    )
    assert qos is node.qos

    for t_ns in (0, 0):
        node.current_time = t_ns
        callback(String(data="x"))

    assert len(_strings(read())) == 2
