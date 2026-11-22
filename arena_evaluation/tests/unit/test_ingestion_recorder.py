"""Unit tests for arena_evaluation.ingestion.recorder (DataRecorderNode)."""

from __future__ import annotations

import io
import os
import pathlib
import re
import subprocess
import sys
import threading
from collections.abc import Callable

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import rosbag2_py
import yaml
from arena_runtime.env.task_layer import encode_modes
from arena_evaluation_msgs.srv import RecordEpisode
from mcap.reader import make_reader
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterType, ParameterValue
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from arena_runtime_msgs.msg import (
    EpisodeRecord,
    RecordedTopic,
    RecordedTopics,
    RobotFleet,
    RobotState,
)
from geometry_msgs.msg import TransformStamped, Twist
from tf2_msgs.msg import TFMessage

from arena_evaluation.ingestion.recorder import DataRecorderNode
from arena_evaluation.storage.schemas import RunMetadata

_CONVERTER = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")


_TF_FRAMES = [
    (re.compile(r"^body_env_\d+_agent_\d+$"), 0.0),
    (re.compile(r"_agent_\d+$"), 200.0),
]


def _transform(frame_id: str, child_frame_id: str, x: float = 0.0, stamp_ns: int = 0) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
    tf.header.stamp.sec = stamp_ns // 1_000_000_000
    tf.header.stamp.nanosec = stamp_ns % 1_000_000_000
    tf.transform.translation.x = x
    return tf


def _tf_message(*pairs) -> TFMessage:
    msg = TFMessage()
    msg.transforms = [_transform(*pair) for pair in pairs]
    return msg


def _make_run_metadata(**overrides) -> RunMetadata:
    kwargs = dict(
        benchmark_id="bench",
        planner="teb",
        robot_model=["jackal"],
        map="hospital_1",
        stage="stage_1",
        episode_id=0,
        recording_started_at="2026-01-01T00:00:00+00:00",
        python_version="3.12",
        ros_distro="humble",
        modes={},
        params={},
    )
    kwargs.update(overrides)
    return RunMetadata(**kwargs)


def _bare_node(**overrides) -> DataRecorderNode:
    """A DataRecorderNode carrying only the attributes the logic under test needs, logging to a buffer."""
    node = DataRecorderNode.__new__(DataRecorderNode)
    node.log_file = io.StringIO()
    node.log_file_path = None
    node.current_time = None
    node.current_metadata = None
    node.current_metadata_path = None
    node.current_episode_id = None
    node.current_episode_dir = None
    node.current_sim_episode_id = None
    node.is_shutting_down = False
    node.recorded_topics = set()
    node.latched_topic_names = set()
    node._throttle_gates = {}
    node.record_rates_hz = {}
    node._clock_received_count = 0
    node._write_success_count = 0
    node._write_drop_count = 0
    node.writer = None
    node.writer_lock = threading.Lock()
    node._tf_lock = threading.Lock()
    node._tf_last_written = {}
    node.tf_frames = list(_TF_FRAMES)
    node._topic_registry = {}
    node.topics_metadata = {}
    node._pre_episode_buffer = []
    node._pre_clock_buffer = []
    node.qos = object()
    node.tf_qos = object()
    node.latched_qos = object()
    node._seen_episodes = set()
    node._record_history = {}
    node._written_records = set()
    node.episodes_recorded = 0
    node._episode_id_offset = 0
    node.episodes_root = None
    node.robot_model = "unknown"
    node.known_robots = set()
    node._recorded_rows = {}
    node._unresolved_types = set()
    node.subs = []
    node._state_types = {}
    # metadata-writing attributes used by _write_episode_metadata
    node.benchmark_id = ""
    node.planner = ""
    node.stage = ""
    node.map_name = ""
    node.env_ns_root = ""
    node.is_reference = False
    node.reference_type = None
    node.suite_name = ""
    node.contest_name = ""
    node.episodes_requested = 0
    node.local_planner = ""
    node.inter_planner = ""
    node.workspace_dir = ""
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


def _log(node: DataRecorderNode, level: str) -> list[str]:
    return [line for line in node.log_file.getvalue().splitlines() if f"[{level}]" in line]


def _string(data: str) -> String:
    return String(data=data)


def _unwritable(path: pathlib.Path) -> pathlib.Path:
    path.mkdir()
    return path


def _mcap_rows(path: pathlib.Path) -> list[tuple[str, bytes, int]]:
    with path.open("rb") as f:
        return [(channel.topic, message.data, message.log_time) for _, channel, message in make_reader(f).iter_messages()]


def _bag(node: DataRecorderNode, uri: pathlib.Path) -> Callable[[], list[tuple[str, bytes, int]]]:
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


def _bag_topics(uri: pathlib.Path) -> dict[str, str]:
    with next(uri.glob("*.mcap")).open("rb") as f:
        summary = make_reader(f).get_summary()
    return {channel.topic: summary.schemas[channel.schema_id].name for channel in summary.channels.values()}


def _strings(rows: list[tuple[str, bytes, int]]) -> list[tuple[int, str]]:
    return [(ts, deserialize_message(data, String).data) for _, data, ts in rows]


def _tf_bag(node: DataRecorderNode, tmp_path: pathlib.Path) -> Callable[[], list[tuple[str, bytes, int]]]:
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/tf", TFMessage)
    return read


def _tf_writes(
    rows: list[tuple[str, bytes, int]],
) -> list[tuple[int, dict[str, float]]]:
    """(timestamp, {child_frame: x}) for every /tf message in the bag."""
    out = []
    for topic, data, ts in rows:
        assert topic == "tf"
        out.append(
            (
                ts,
                {t.child_frame_id: t.transform.translation.x for t in deserialize_message(data, TFMessage).transforms},
            )
        )
    return out


@pytest.fixture
def fake_share(tmp_path):
    """An arena_evaluation share dir on the ament prefix path, so workspace_root (= share.parents[3]) is tmp_path."""
    prefix = tmp_path / "w" / "install"
    share = prefix / "share" / "arena_evaluation"
    share.mkdir(parents=True)
    index = prefix / "share" / "ament_index" / "resource_index" / "packages"
    index.mkdir(parents=True)
    (index / "arena_evaluation").touch()
    old_prefix = os.environ.get("AMENT_PREFIX_PATH")
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join([str(prefix), old_prefix or ""])
    try:
        yield share
    finally:
        if old_prefix is None:
            os.environ.pop("AMENT_PREFIX_PATH", None)
        else:
            os.environ["AMENT_PREFIX_PATH"] = old_prefix


def _build_full_node(argv, **node_kwargs):
    old_argv = sys.argv
    sys.argv = argv
    try:
        return DataRecorderNode(**node_kwargs)
    finally:
        sys.argv = old_argv


@pytest.fixture
def full_node(tmp_path, fake_share):
    node = _build_full_node(["pytest", "--dir", str(tmp_path / "episodes")])
    yield node
    try:
        node.destroy_node()
    except Exception:
        pass


# Constructor


def test_constructor_argv_equals_form(tmp_path, fake_share):
    target = tmp_path / "eq"
    node = _build_full_node(["pytest", f"--dir={target}"])
    try:
        assert node.episodes_root == (target / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_relative_dir_resolves_under_workspace_root(tmp_path, fake_share):
    node = _build_full_node(["pytest", "--dir", "episodes/rel"])
    try:
        assert node.episodes_root == (tmp_path / "episodes" / "rel" / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_auto_mode_reuses_existing_autoprefix(tmp_path, fake_share):
    node = _build_full_node(
        ["pytest"],
        parameter_overrides=[Parameter("data_recorder_autoprefix", Parameter.Type.STRING, "20240101-000000")],
    )
    try:
        assert node.episodes_root == (tmp_path / "data" / "runs" / "20240101-000000" / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_data_root_creates_runs_uuid(tmp_path, fake_share):
    node = _build_full_node(["pytest", "--dir", "data"])
    try:
        assert node.episodes_root.parent.parent == (tmp_path / "data" / "runs").resolve()
        assert node.episodes_root.name == "episodes"
        assert (node.run_dir / "recorder.log").is_file()
    finally:
        node.destroy_node()


# read_config


# Throttling / callbacks


def test_throttled_callback_respects_the_rate_gate(tmp_path):
    node = _bare_node(record_rates_hz={"/cmd_vel": 20.0})
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    node.current_time = 0
    cb = node._create_throttled_callback("/cmd_vel")
    cb(_string("m1"))
    node.current_time = 49_999_999
    cb(_string("m2"))
    node.current_time = 50_000_000
    cb(_string("m3"))
    cb(_string("m4"))
    node.current_time = 100_000_000
    cb(_string("m5"))
    assert _strings(read()) == [(0, "m1"), (50_000_000, "m3"), (100_000_000, "m5")]


def test_throttled_callback_buffers_before_first_clock():
    node = _bare_node()
    cb = node._create_throttled_callback("/cmd_vel")
    cb("m1")
    assert node._pre_clock_buffer == [("/cmd_vel", "m1")]
    assert node._write_drop_count == 0


def test_throttled_callback_drops_message_when_clock_buffer_already_flushed(tmp_path):
    # after the first clock tick flushes a non-empty buffer, the buffer is set
    # to None; a pre-clock message arriving afterwards is simply discarded
    node = _bare_node()
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/a", String)
    node._pre_clock_buffer = [("/a", _string("m0"))]
    node.clock_callback(Clock())  # flush path drops _pre_clock_buffer
    assert node._pre_clock_buffer is None
    node.current_time = None
    cb = node._create_throttled_callback("/cmd_vel")
    cb(_string("m1"))
    assert _strings(read()) == [(0, "m0")]


def test_unthrottled_callback_buffers_before_first_clock():
    node = _bare_node()
    cb = node._create_unthrottled_callback("/scan")
    cb("a")
    assert node._pre_clock_buffer == [("/scan", "a")]
    assert node._write_drop_count == 0


def test_unthrottled_callback_discards_message_when_clock_buffer_already_flushed(
    tmp_path,
):
    node = _bare_node()
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/a", String)
    node._pre_clock_buffer = [("/a", _string("m0"))]
    node.clock_callback(Clock())
    node.current_time = None
    cb = node._create_unthrottled_callback("/scan")
    cb(_string("a"))
    assert _strings(read()) == [(0, "m0")]


# Clock handling


def test_tf_bone_frames_are_capped_while_root_frames_pass_every_stamp(tmp_path):
    node = _bare_node(current_time=1_000_000_000)
    read = _tf_bag(node, tmp_path)
    root = ("map", "body_env_0_agent_2", 0.0)
    bone = ("body_env_0_agent_2", "l_p_collar_env_0_agent_2", 0.0)

    for step in range(0, 400_000_001, 20_000_000):
        node._tf_callback(_tf_message((*root, step), (*bone, step)))

    writes = _tf_writes(read())
    assert [ts for ts, _ in writes] == list(range(0, 400_000_001, 20_000_000))
    assert all("body_env_0_agent_2" in frames for _, frames in writes)
    bone_stamps = [ts for ts, frames in writes if "l_p_collar_env_0_agent_2" in frames]
    assert bone_stamps == [0, 200_000_000, 400_000_000]


def test_tf_callback_buffers_before_first_clock_and_writes_at_header_stamps(tmp_path):
    node = _bare_node()
    read = _tf_bag(node, tmp_path)
    node._tf_callback(_tf_message(("map", "odom", 1.0, 5_000_000)))
    node._tf_callback(_tf_message(("odom", "base_link", 3.0, 7_000_000)))
    assert [topic for topic, _ in node._pre_clock_buffer] == ["/tf", "/tf"]
    assert node._write_success_count == 0

    msg = Clock()
    msg.clock.sec = 1
    node.clock_callback(msg)

    assert _tf_writes(read()) == [
        (5_000_000, {"odom": 1.0}),
        (7_000_000, {"base_link": 3.0}),
    ]
    assert node._pre_clock_buffer is None


def test_stop_episode_keeps_the_transforms_already_written(tmp_path):
    node = _bare_node(current_sim_episode_id=3, current_time=50_000_000)
    read = _tf_bag(node, tmp_path)
    node._tf_callback(_tf_message(("map", "odom", 1.0, 50_000_000)))

    node._stop_episode(3, outcome_state=2)

    assert node.writer is None
    assert _tf_writes(read()) == [(50_000_000, {"odom": 1.0})]


def test_clock_backward_jump_clears_the_tf_frame_state():
    node = _bare_node(current_time=10_000_000_000)
    node._tf_callback(_tf_message(("map", "odom", 1.0, 9_000_000_000)))
    assert node._tf_last_written == {("map", "odom"): 9_000_000_000}
    msg = Clock()
    msg.clock.sec = 5

    node.clock_callback(msg)

    assert node._tf_last_written == {}


# Writer plumbing: _register_topic / _ensure_topic_in_bag / _write_to_bag_at


def test_register_topic_fallback_to_str_when_no_module():
    node = _bare_node()
    node._register_topic("/mystery", object())
    assert node._topic_registry["/mystery"].type == str(object())


def test_write_to_bag_at_noop_when_shutting_down(tmp_path):
    node = _bare_node(is_shutting_down=True)
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    node._write_to_bag_at("/cmd_vel", String(), 1)
    assert read() == []


def test_write_to_bag_at_serialization_failure_logs_error(tmp_path):
    node = _bare_node()
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    node._write_to_bag_at("/cmd_vel", object(), 1)
    assert read() == []
    assert any("Serialization failed" in e for e in _log(node, "ERROR"))


def test_write_to_bag_at_writer_error_is_logged(tmp_path):
    node = _bare_node()
    _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", String)
    node.writer.close()
    node._write_to_bag_at("/cmd_vel", String(), 1)
    assert node._write_success_count == 0
    assert any("WRITE ERROR" in e for e in _log(node, "ERROR"))


# Raw (already-serialized) subscriptions: bytes reach the bag unchanged


def test_write_to_bag_at_raw_bytes_reach_the_bag_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    node = _bare_node()
    read = _bag(node, tmp_path / "bag")
    node._register_topic("/cmd_vel", Twist)

    twist = Twist()
    twist.linear.x = 1.5
    raw_bytes = serialize_message(twist)

    node._write_to_bag_at("/cmd_vel", raw_bytes, 42)

    assert read() == [("cmd_vel", raw_bytes, 42)]


def _node_fixture(tmp_path, namespace: str | None = None, **parameters):
    overrides = [
        Parameter("record_data_dir", value=str(tmp_path / "episodes")),
        *(Parameter(name, value=value) for name, value in parameters.items()),
    ]
    node = DataRecorderNode(namespace=namespace, parameter_overrides=overrides)
    yield node
    node.destroy_node()


@pytest.fixture
def recorder_node(tmp_path):
    yield from _node_fixture(tmp_path)


@pytest.fixture
def env_node(tmp_path):
    yield from _node_fixture(tmp_path, "env_0")


@pytest.fixture
def state_node(tmp_path):
    yield from _node_fixture(tmp_path, "arena/env_1/arena_env_node", robot="jackal")


def _recorded_rows() -> RecordedTopics:
    string = "std_msgs/msg/String"
    return RecordedTopics(
        topics=[
            RecordedTopic(
                key="audio_raw",
                topic="{ns}/audio/raw_array",
                msg_type=string,
                robot_scoped=True,
                depth=1000,
                recorded=True,
            ),
            RecordedTopic(
                key="heard_sound_events",
                topic="{tg}/heard_sound_events",
                msg_type=string,
                reliable=True,
                recorded=True,
            ),
            RecordedTopic(
                key="pose_log",
                topic="{tg}/pose_log",
                msg_type=string,
                throttled=True,
                recorded=True,
            ),
            RecordedTopic(
                key="room_impulses",
                topic="{tg}/acoustic/impulses",
                msg_type=string,
                qos_transient_local=True,
                reliable=True,
            ),
            RecordedTopic(
                key="unknown_type",
                topic="{tg}/unknown",
                msg_type="no_such_package_msgs/msg/Nothing",
                recorded=True,
            ),
        ]
    )


def test_recorded_topics_message_reaches_robots_known_before_it(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.known_robots.add("robot_0")
    before = len(recorder_node.subs)
    recorder_node.recorded_topics_callback(_recorded_rows())

    audio = {sub.topic: sub for sub in recorder_node.subs[before:]}["/robot_0/audio/raw_array"]
    assert audio.qos_profile.reliability == QoSReliabilityPolicy.BEST_EFFORT
    assert audio.qos_profile.depth == 1000


def test_robot_seen_after_the_recorded_topics_message_gets_its_rows(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.recorded_topics_callback(_recorded_rows())
    before = len(recorder_node.subs)
    recorder_node.robots_fleet_callback(_fleet_message([("robot_1", "jackal")]))
    assert "/robot_1/audio/raw_array" in {sub.topic for sub in recorder_node.subs[before:]}


def test_optional_recorded_row_is_subscribed_latched_only_when_listed(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.recorded_topics_callback(_recorded_rows())
    assert "/acoustic/impulses" not in {sub.topic for sub in recorder_node.subs}

    recorder_node.config["optional_topics"] = ["room_impulses"]
    recorder_node.recorded_topics_callback(_recorded_rows())
    impulses = {sub.topic: sub for sub in recorder_node.subs}["/acoustic/impulses"]
    assert impulses.qos_profile.reliability == QoSReliabilityPolicy.RELIABLE
    assert impulses.qos_profile.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
    assert impulses.qos_profile.depth == recorder_node.latched_qos.depth
    assert "acoustic/impulses" in recorder_node.latched_topic_names


# Episode lifecycle: _begin_episode / _stop_episode / start_episode service


def test_begin_episode_opens_new_episode_once(tmp_path):
    node = _recording_node(tmp_path)
    assert node._begin_episode(7, source="episode_record") is True
    assert node._begin_episode(7) is False
    assert node.episodes_recorded == 1
    assert node.current_sim_episode_id == 7
    assert [p.name for p in node.episodes_root.iterdir()] == ["episode_000"]
    node._close_current_writer()


def test_stop_episode_ignores_stale_episode(tmp_path):
    node = _bare_node(current_sim_episode_id=3)
    read = _bag(node, tmp_path / "bag")
    writer = node.writer
    node._stop_episode(5, outcome_state=2)
    assert node.writer is writer
    read()


def test_stop_episode_metadata_write_failure_still_closes(tmp_path):
    node = _bare_node(
        current_sim_episode_id=3,
        current_metadata=_make_run_metadata(),
        current_metadata_path=_unwritable(tmp_path / "episode_003.yaml"),
    )
    _bag(node, tmp_path / "bag")
    node._stop_episode(3, outcome_state=2)
    assert node.writer is None
    assert any("Failed to write outcome metadata" in e for e in _log(node, "ERROR"))


def _service_request(**fields) -> RecordEpisode.Request:
    return RecordEpisode.Request(**{"episode_id": 9, "command": RecordEpisode.Request.COMMAND_START, **fields})


def test_service_callback_unknown_command():
    node = _bare_node()
    resp = RecordEpisode.Response()
    node._start_episode_service_callback(_service_request(command=7), resp)
    assert resp.success is False
    assert resp.message == "unknown command 7"


# Topic callbacks: episode_record / semantic_snapshot / robots_fleet


def _episode_message(episode_id=3, outcome=1):
    msg = EpisodeRecord()
    msg.episode_id = episode_id
    msg.outcome_state = outcome
    msg.outcome_info = ""
    return msg


def test_episode_record_callback_begins_new_episode(env_node):
    env_node.current_time = 5000
    env_node.episode_record_callback(_episode_message(outcome=1))
    assert env_node.current_sim_episode_id == 3
    assert env_node.current_metadata.task_generator_episode_id == 3
    env_node._close_current_writer()
    rows = _mcap_rows(env_node.episodes_root / "episode_000" / "episode_000.mcap")
    assert [(topic, ts) for topic, _, ts in rows] == [("env_0/state/episode", 5000)]


def test_semantic_snapshot_callback_writes_state_semantics(env_node):
    env_node.current_time = 777
    env_node.semantic_snapshot_callback(b"snap")
    assert env_node._pre_episode_buffer == [("/env_0/state/semantics", b"snap", 777)]


def _fleet_message(robots_ns_model):
    fleet = RobotFleet()
    states = []
    for ns, model in robots_ns_model:
        st = RobotState()
        st.descriptor.ns = ns
        st.descriptor.model = model
        states.append(st)
    fleet.robots = states
    return fleet


def test_robots_fleet_callback_writes_and_discovers_robots(env_node):
    node = env_node
    node.current_time = 2000
    node._begin_episode(0)
    node.current_metadata = _make_run_metadata(robot_model=["unknown"])
    before = len(node.subs)

    node.robots_fleet_callback(_fleet_message([("robot_0", "jackal")]))

    first_wave = {sub.topic for sub in node.subs[before:]}
    assert {"/robot_0/cmd_vel", "/robot_0/scan", "/robot_0/plan"} <= first_wave
    assert "robot_0" in node.known_robots
    assert node.robot_model == "jackal"
    assert yaml.safe_load(node.current_metadata_path.read_text())["robot_model"] == ["jackal"]

    # second sighting of the same robot: no re-subscription
    count = len(node.subs)
    node.robots_fleet_callback(_fleet_message([("robot_0", "jackal")]))
    assert len(node.subs) == count

    # a new robot triggers a new subscription wave, no controller topics for a model without model_params
    node.robots_fleet_callback(_fleet_message([("robot_1", "turtlebot3")]))
    second_wave = {sub.topic for sub in node.subs[count:]}
    assert "/robot_1/cmd_vel" in second_wave
    assert len(second_wave) == len(first_wave) - 2
    assert "robot_1" in node.known_robots
    assert yaml.safe_load(node.current_metadata_path.read_text())["robot_model"] == [
        "jackal",
        "turtlebot3",
    ]

    node._close_current_writer()
    rows = _mcap_rows(node.episodes_root / "episode_000" / "episode_000.mcap")
    assert ("env_0/state/robots", 2000) in {(topic, ts) for topic, _, ts in rows}


# discover_topics


def _advertise(node: DataRecorderNode, *topics: tuple[str, type]) -> None:
    for name, msg_type in topics:
        node.create_publisher(msg_type, name, 1)


def test_discover_topics_subscribes_odom_and_infers_robot_model(env_node):
    env_node._begin_episode(0)
    _advertise(env_node, ("/env_0/robot/jackal/odom", Odometry))

    env_node.discover_topics()

    assert "/env_0/robot/jackal/odom" in {sub.topic for sub in env_node.subs}
    assert env_node.robot_model == "jackal"
    assert env_node.current_metadata.robot_model == ["jackal"]
    assert yaml.safe_load(env_node.current_metadata_path.read_text())["robot_model"] == ["jackal"]
    env_node._close_current_writer()


def test_discover_topics_records_interaction_and_animation_states_of_its_env(
    state_node,
):
    pytest.importorskip("arena_humansim_msgs.msg")
    from arena_humansim_msgs.msg import InteractionFrame
    from arena_people_msgs.msg import AnimationStates, Pedestrians

    before = len(state_node.subs)
    _advertise(
        state_node,
        ("/arena/env_1/interactions", InteractionFrame),
        ("/arena/env_1/animation_states", AnimationStates),
        ("/arena/env_10/animation_states", AnimationStates),
        ("/arena/env_1/human/stream", Pedestrians),
    )
    state_node.discover_topics()
    subscribed = {sub.topic: sub for sub in state_node.subs[before:]}
    assert set(subscribed) == {
        "/arena/env_1/interactions",
        "/arena/env_1/animation_states",
    }
    assert subscribed["/arena/env_1/interactions"].qos_profile.durability == QoSDurabilityPolicy.TRANSIENT_LOCAL
    assert subscribed["/arena/env_1/animation_states"].qos_profile.durability == QoSDurabilityPolicy.VOLATILE
    assert "arena/env_1/interactions" in state_node.latched_topic_names
    assert "arena/env_1/animation_states" not in state_node.latched_topic_names
    assert state_node._topic_registry["/arena/env_1/interactions"].type == "arena_humansim_msgs/msg/InteractionFrame"
    assert state_node._topic_registry["/arena/env_1/animation_states"].type == "arena_people_msgs/msg/AnimationStates"


def test_discovered_interactions_are_written_unthrottled(state_node):
    pytest.importorskip("arena_humansim_msgs.msg")
    from arena_humansim_msgs.msg import InteractionFrame

    before = len(state_node.subs)
    _advertise(state_node, ("/arena/env_1/interactions", InteractionFrame))
    state_node.discover_topics()
    (sub,) = state_node.subs[before:]
    state_node._begin_episode(0)
    state_node.current_time = 1_000
    sub.callback(InteractionFrame())
    sub.callback(InteractionFrame())
    state_node._close_current_writer()
    rows = _mcap_rows(state_node.episodes_root / "episode_000" / "episode_000.mcap")
    assert [topic for topic, _, _ in rows] == ["arena/env_1/interactions"] * 2


# Episode metadata: write / finalize / update-from-EpisodeRecord


def test_update_metadata_from_episode_populates_fields(tmp_path):
    node = _bare_node(
        current_metadata=_make_run_metadata(robot_model=["unknown"], map="unknown"),
        current_sim_episode_id=7,
    )
    node.current_metadata_path = tmp_path / "episode.yaml"
    pv_double = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.4)
    pv_string = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value="turtlebot3")
    msg = EpisodeRecord()
    msg.episode_id = 7
    msg.world = "hospital_1"
    msg.robots = ["jackal"]
    msg.task_layer = "generated_runtime"
    msg.modes = encode_modes({"robots": ["explore"], "fleet": ["convoy", "wedge"], "modules": []})
    msg.params = [
        RclParameter(name="task.random.robot.radius", value=pv_double),
        RclParameter(name="task.explore.model", value=pv_string),
    ]

    node._update_metadata_from_episode(msg)

    md = node.current_metadata
    assert md.robot_model == ["jackal"]
    assert md.map == "hospital_1"
    assert md.task_layer == "generated_runtime"
    assert md.modes == {
        "robots": ["explore"],
        "fleet": ["convoy", "wedge"],
        "modules": [],
    }
    assert md.params == {
        "task.random.robot.radius": 0.4,
        "task.explore.model": "turtlebot3",
    }
    assert yaml.safe_load((tmp_path / "episode.yaml").read_text())["map"] == "hospital_1"


def test_update_metadata_from_episode_write_failure_logged(tmp_path):
    node = _bare_node(current_metadata=_make_run_metadata(), current_sim_episode_id=3)
    node.current_metadata_path = _unwritable(tmp_path / "episode.yaml")
    node._update_metadata_from_episode(_episode_message())
    assert any("Failed to update episode metadata" in e for e in _log(node, "ERROR"))


# _param_value_to_py


@pytest.mark.parametrize(
    "ptype, field, value",
    [
        (ParameterType.PARAMETER_BOOL, "bool_value", True),
        (ParameterType.PARAMETER_INTEGER, "integer_value", 7),
        (ParameterType.PARAMETER_DOUBLE, "double_value", 1.5),
        (ParameterType.PARAMETER_STRING, "string_value", "x"),
        (ParameterType.PARAMETER_BYTE_ARRAY, "byte_array_value", [1, 2]),
        (ParameterType.PARAMETER_BOOL_ARRAY, "bool_array_value", [True, False]),
        (ParameterType.PARAMETER_INTEGER_ARRAY, "integer_array_value", [1, 2]),
        (ParameterType.PARAMETER_DOUBLE_ARRAY, "double_array_value", [0.1, 0.2]),
        (ParameterType.PARAMETER_STRING_ARRAY, "string_array_value", ["a", "b"]),
    ],
)
def test_param_value_to_py_maps_known_types(ptype, field, value):
    node = _bare_node()
    pv = ParameterValue(type=ptype, **{field: value})
    assert node._param_value_to_py(pv) == value


# _start_episode_recording


def _recording_node(tmp_path, **overrides):
    episodes_root = tmp_path / "episodes"
    episodes_root.mkdir()
    node = _bare_node(
        episodes_root=episodes_root,
        base_dir=str(tmp_path / "share"),  # no config dir -> "no compression" path
        benchmark_id="b1",
        planner="cont-teb",
        stage="s1",
        map_name="m1",
        robot_model="jackal",
        env_ns_root="/env_0",
        is_reference=False,
        reference_type=None,
        suite_name="suite",
        contest_name="contest",
        episodes_requested=5,
        local_planner="",
        inter_planner="",
        **overrides,
    )
    return node


def test_start_episode_recording_opens_writer_and_flushes_buffer(tmp_path):
    node = _recording_node(tmp_path, current_sim_episode_id=None)
    node._register_topic("/env_0/state/episode", EpisodeRecord)
    node._pre_episode_buffer = [("/env_0/state/episode", b"\x01\x02", 12345)]

    node._start_episode_recording(7)

    assert node.current_episode_id == 0
    assert node.current_sim_episode_id == 7
    assert node._episode_id_offset == 1
    assert node.current_episode_dir == node.episodes_root / "episode_000"
    assert node.current_metadata_path == node.episodes_root / "episode_000" / "episode_000.yaml"

    ep_dir = node.episodes_root / "episode_000"
    assert (ep_dir / "episode_000.yaml").exists()
    data = yaml.safe_load((ep_dir / "episode_000.yaml").read_text())
    assert data["benchmark_id"] == "b1"
    assert data["episode_id"] == 0
    assert data["task_generator_episode_id"] == 7

    assert "env_0/state/episode" in node.recorded_topics
    assert node._pre_episode_buffer == []
    assert node._write_success_count == 0

    node._close_current_writer()
    assert _mcap_rows(ep_dir / "episode_000.mcap") == [("env_0/state/episode", b"\x01\x02", 12345)]


def test_start_episode_recording_writer_failure_sets_writer_none(tmp_path):
    node = _recording_node(tmp_path)
    config_dir = tmp_path / "share" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcap_writer_options.yaml").write_text("{{{{ not yaml")

    node._start_episode_recording(1)

    assert node.writer is None
    assert node._episode_id_offset == 1
    assert any("Failed to open MCAP writer" in e for e in _log(node, "ERROR"))
    assert (node.episodes_root / "episode_000" / "episode_000.yaml").exists()


# _close_current_writer (flush + flatten + rosbag metadata merge)


def _episode_dir_with_bag(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"mcap-bytes")
    (inner / "episode_000_1.mcap").write_bytes(b"mcap-bytes")
    (inner / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  message_count: 42\n  topics_with_message_count:\n    - topic_metadata:\n        name: /cmd_vel\n      message_count: 42\n")
    return ep_dir


def _recorded_episode(tmp_path):
    """A node mid-episode with a real bag of three messages under episode_000."""
    ep_dir = tmp_path / "episode_000"
    ep_dir.mkdir()
    node = _bare_node(
        current_episode_dir=ep_dir,
        current_metadata=_make_run_metadata(),
        current_metadata_path=ep_dir / "episode_000.yaml",
    )
    _bag(node, ep_dir / "episode_000")
    node._register_topic("/cmd_vel", Twist)
    for stamp in range(3):
        node._write_to_bag_at("/cmd_vel", Twist(), stamp)
    return node, ep_dir


def test_close_current_writer_flattens_mcap_files_and_merges_rosbag_metadata(tmp_path):
    node, ep_dir = _recorded_episode(tmp_path)
    (ep_dir / "episode_000" / "episode_000_1.mcap").write_bytes(b"mcap-bytes")

    node._close_current_writer()

    assert node.writer is None
    # first mcap becomes episode_000.mcap; the second is suffixed to avoid collision
    assert len(_mcap_rows(ep_dir / "episode_000.mcap")) == 3
    assert (ep_dir / "episode_000_episode_000_1.mcap").exists()
    assert not (ep_dir / "episode_000").exists()
    assert node.current_metadata.rosbag2_message_count == 3
    assert node.current_metadata.rosbag2_topics[0]["message_count"] == 3
    data = yaml.safe_load((ep_dir / "episode_000.yaml").read_text())
    assert data["recording_ended_at"] is not None
    assert data["pedsim_available"] is False
    assert data["recorded_topics"] == ["cmd_vel"]


def test_close_current_writer_close_error_is_logged(tmp_path):
    ep_dir = _episode_dir_with_bag(tmp_path)
    node = _bare_node()
    node.writer = object()
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert node.writer is None
    assert any("Error closing MCAP writer" in e for e in _log(node, "ERROR"))
    assert (ep_dir / "episode_000.mcap").exists()


def test_close_current_writer_move_failure_is_logged(tmp_path):
    ep_dir = _episode_dir_with_bag(tmp_path)
    (ep_dir / "episode_000.mcap").mkdir()
    blocked = ep_dir / "episode_000_episode_000_0.mcap"
    blocked.mkdir()
    (blocked / "episode_000_0.mcap").write_bytes(b"taken")
    node = _bare_node()
    _bag(node, tmp_path / "elsewhere")
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"

    node._close_current_writer()

    assert any("Failed to flatten" in e for e in _log(node, "ERROR"))
    assert (ep_dir / "episode_000" / "episode_000_0.mcap").exists()  # untouched


def test_close_current_writer_corrupt_rosbag_metadata_is_logged(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"x")
    (inner / "metadata.yaml").write_text("{{{{ not yaml")
    node = _bare_node()
    _bag(node, tmp_path / "elsewhere")
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert any("Failed to merge rosbag metadata" in e for e in _log(node, "ERROR"))
    assert node.current_metadata.rosbag2_message_count is None


# Shutdown / lifecycle


def test_finalize_closes_writer_once(tmp_path, capsys):
    node = _bare_node()
    _bag(node, tmp_path / "bag")
    node.finalize()
    assert node.is_shutting_down is True
    assert node.writer is None
    node.finalize()  # second call is a no-op
    assert capsys.readouterr().out.count("finalize() called") == 1


def test_destroy_node_calls_finalize(tmp_path):
    node = DataRecorderNode(parameter_overrides=[Parameter("record_data_dir", value=str(tmp_path / "episodes"))])
    node.destroy_node()
    assert node.is_shutting_down is True


def test_log_methods_print_when_rclpy_not_initialized():
    script = "from arena_evaluation.ingestion.recorder import DataRecorderNode\nnode = DataRecorderNode.__new__(DataRecorderNode)\nnode.log_file = None\nnode._log_info('info-msg')\nnode._log_warn('warn-msg')\nnode._log_error('error-msg')\n"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    assert "[DataRecorder] [INFO] info-msg" in out
    assert "[DataRecorder] [WARN] warn-msg" in out
    assert "[DataRecorder] [ERROR] error-msg" in out


def test_finalize_tolerates_signal_registration_failure():
    node = _bare_node()
    off_the_main_thread = threading.Thread(target=node.finalize)
    off_the_main_thread.start()
    off_the_main_thread.join()
    assert node.is_shutting_down is True
    assert any("Recording finished" in i for i in _log(node, "INFO"))


def test_main_exception_path_exits_nonzero(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("")
    script = "import sys\nfrom arena_evaluation.ingestion.recorder import main\nmain(args=sys.argv)\n"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "--ros-args",
            "-p",
            f"record_data_dir:={blocked}/episodes",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "Exception in main" in proc.stdout


# End-to-end service flow on a fully constructed node


def test_start_stop_episode_service_end_to_end(tmp_path, full_node):
    start_resp = RecordEpisode.Response()
    full_node._start_episode_service_callback(_service_request(episode_id=11), start_resp)

    assert start_resp.success is True
    assert start_resp.message == "recording episode 0"
    assert full_node.current_episode_id == 0
    assert full_node.current_sim_episode_id == 11
    ep_dir = full_node.episodes_root / "episode_000"
    assert (ep_dir / "episode_000.yaml").exists()
    assert full_node.writer is not None

    stop_resp = RecordEpisode.Response()
    full_node._start_episode_service_callback(
        _service_request(
            episode_id=11,
            command=RecordEpisode.Request.COMMAND_STOP,
            outcome_state=2,
            outcome_info="ok",
        ),
        stop_resp,
    )

    assert stop_resp.success is True
    assert stop_resp.message == "stopped"
    assert full_node.writer is None
    assert (ep_dir / "episode_000.mcap").exists()
    data = yaml.safe_load((ep_dir / "episode_000.yaml").read_text())
    assert data["outcome_state"] == 2
    assert data["outcome_info"] == "ok"
    assert data["recording_ended_at"] is not None


def test_constructor_auto_mode_generates_run_id_from_map_robot_planner(fake_share):
    old_argv = sys.argv
    sys.argv = ["pytest"]
    node = None
    try:
        node = DataRecorderNode(
            parameter_overrides=[
                Parameter("map", Parameter.Type.STRING, "/tmp/hospital_1.world"),
                Parameter("robot", Parameter.Type.STRING, "jackal"),
                Parameter("local_planner", Parameter.Type.STRING, "dwb"),
            ]
        )
        assert node.run_dir.name.endswith("-hospital_1-jackal-dwb")
    finally:
        sys.argv = old_argv
        if node is not None:
            node.destroy_node()
