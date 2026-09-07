"""Unit tests for arena_evaluation.ingestion.recorder (DataRecorderNode).

Pure-logic units are exercised on a bare (uninitialized) node instance so no
live ROS graph is required; the full constructor is exercised against a
monkeypatched package share directory so all filesystem state stays under
tmp_path. The rosbag2 writer is mocked at every call site.
"""

from __future__ import annotations

import pathlib
import re
import sys
import threading
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import yaml

from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterType, ParameterValue
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from task_generator_msgs.msg import EpisodeRecord, RobotFleet, RobotState
from geometry_msgs.msg import TransformStamped, Twist
from tf2_msgs.msg import TFMessage

from arena_evaluation.ingestion import recorder
from arena_evaluation.ingestion.recorder import DataRecorderNode
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata


class _FakeLogger:
    def __init__(self):
        self.infos = []
        self.warns = []
        self.errors = []

    def info(self, msg):
        self.infos.append(msg)

    def warn(self, msg):
        self.warns.append(msg)

    def error(self, msg):
        self.errors.append(msg)


_TF_FRAMES = [
    (re.compile(r"^body_env_\d+_agent_\d+$"), 0.0),
    (re.compile(r"_agent_\d+$"), 200.0),
]


def _transform(frame_id: str, child_frame_id: str, x: float = 0.0) -> TransformStamped:
    tf = TransformStamped()
    tf.header.frame_id = frame_id
    tf.child_frame_id = child_frame_id
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
    )
    kwargs.update(overrides)
    return RunMetadata(**kwargs)


def _bare_node(**overrides) -> DataRecorderNode:
    """A DataRecorderNode carrying only the attributes the logic under test needs."""
    node = DataRecorderNode.__new__(DataRecorderNode)
    node.log_file = None
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
    node.last_recorded_times = {}
    node._clock_received_count = 0
    node._write_success_count = 0
    node._write_drop_count = 0
    node.writer = None
    node.writer_lock = threading.Lock()
    node._tf_lock = threading.Lock()
    node._tf_pending = {}
    node._tf_last_written = {}
    node._tf_last_flush = 0
    node.tf_frames = list(_TF_FRAMES)
    node._topic_registry = {}
    node.topics_metadata = {}
    node._pre_episode_buffer = []
    node._pre_clock_buffer = []
    node.freqs = {"default": 20.0}
    node.qos = object()
    node.tf_qos = object()
    node.latched_qos = object()
    node.reliable_volatile_qos = object()
    node._seen_episodes = set()
    node.episodes_recorded = 0
    node._episode_id_offset = 0
    node.episodes_root = None
    node.robot_model = "unknown"
    node.known_robots = set()
    node.subs = []
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
    node.logger = _FakeLogger()
    node.get_logger = lambda: node.logger
    for key, value in overrides.items():
        setattr(node, key, value)
    return node


@pytest.fixture
def fake_share(tmp_path, monkeypatch):
    """Point the recorder at a fake package share dir rooted under tmp_path.

    The depth is chosen so that workspace_root (= share.parents[3]) lands
    inside tmp_path for the auto:/ and relative-dir resolution branches.
    """
    share = tmp_path / "w1" / "w2" / "w3" / "arena_evaluation"
    share.mkdir(parents=True)
    monkeypatch.setattr(recorder, "get_package_share_directory", lambda _name: str(share))
    return share


def _build_full_node(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return DataRecorderNode()


@pytest.fixture
def full_node(tmp_path, fake_share, monkeypatch):
    node = _build_full_node(monkeypatch, ["pytest", "--dir", str(tmp_path / "episodes")])
    yield node
    try:
        node.destroy_node()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_uses_dir_flag(tmp_path, full_node):
    assert full_node.episodes_root == (tmp_path / "episodes").resolve()
    assert full_node.episodes_root.is_dir()
    assert (full_node.run_dir / "recorder.log").is_file() or (full_node.episodes_root / "recorder.log").is_file()


def test_constructor_argv_equals_form(tmp_path, fake_share, monkeypatch):
    target = tmp_path / "eq"
    node = _build_full_node(monkeypatch, ["pytest", f"--dir={target}"])
    try:
        assert node.episodes_root == (target / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_relative_dir_resolves_under_workspace_root(tmp_path, fake_share, monkeypatch):
    # workspace_root = Path(base_dir).parents[3] == tmp_path for the fake share
    # layout (w1/w2/w3/arena_evaluation), so a relative --dir lands under tmp_path.
    node = _build_full_node(monkeypatch, ["pytest", "--dir", "episodes/rel"])
    try:
        assert node.episodes_root == (tmp_path / "episodes" / "rel" / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_auto_mode_uses_timestamped_dir(tmp_path, fake_share, monkeypatch):
    node = _build_full_node(monkeypatch, ["pytest"])
    try:
        stamp = node.get_parameter("data_recorder_autoprefix").value
        assert stamp is not None
        assert node.episodes_root == (tmp_path / "data" / "runs" / stamp / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_auto_mode_reuses_existing_autoprefix(tmp_path, fake_share, monkeypatch):
    real_get_parameter = recorder.Node.get_parameter

    def _fake_get_parameter(self, name, *args, **kwargs):
        if name == "data_recorder_autoprefix":
            return Parameter("data_recorder_autoprefix", Parameter.Type.STRING, "20240101-000000")
        return real_get_parameter(self, name, *args, **kwargs)

    monkeypatch.setattr(recorder.Node, "get_parameter", _fake_get_parameter)
    node = _build_full_node(monkeypatch, ["pytest"])
    try:
        assert node.episodes_root == (tmp_path / "data" / "runs" / "20240101-000000" / "episodes").resolve()
    finally:
        node.destroy_node()


def test_constructor_data_root_creates_runs_uuid(tmp_path, fake_share, monkeypatch):
    node = _build_full_node(monkeypatch, ["pytest", "--dir", "data"])
    try:
        assert node.episodes_root.parent.parent == (tmp_path / "data" / "runs").resolve()
        assert node.episodes_root.name == "episodes"
        assert (node.run_dir / "recorder.log").is_file()
    finally:
        node.destroy_node()


def test_constructor_registers_subscriptions_and_service(tmp_path, full_node):
    assert full_node._start_service is not None
    assert len(full_node.subs) == 7
    # /tf and /tf_static are subscribed once, at construction
    assert full_node.latched_topic_names == {"state/episode", "state/robots", "state/semantics", "tf_static"}
    assert full_node.freqs == {"default": 20.0}


# ---------------------------------------------------------------------------
# read_config
# ---------------------------------------------------------------------------


def test_read_config_falls_back_when_file_missing(tmp_path):
    node = _bare_node(base_dir=str(tmp_path / "missing"))
    assert node.read_config() == {"record_frequencies": {"default": 20.0}, "tf_frames": []}


def test_read_config_parses_file(tmp_path):
    cfg_dir = tmp_path / "share" / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "data_recorder_config.yaml").write_text("record_frequencies:\n  default: 50.0\n  odom: 5.0\n")
    node = _bare_node(base_dir=str(tmp_path / "share"))
    assert node.read_config() == {"record_frequencies": {"default": 50.0, "odom": 5.0}}


def test_read_config_falls_back_on_corrupt_yaml(tmp_path):
    cfg_dir = tmp_path / "share" / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "data_recorder_config.yaml").write_text("{{{{ not yaml")
    node = _bare_node(base_dir=str(tmp_path / "share"))
    assert node.read_config() == {"record_frequencies": {"default": 20.0}, "tf_frames": []}


# ---------------------------------------------------------------------------
# Throttling / callbacks
# ---------------------------------------------------------------------------


def test_resolve_throttle_ms_matches_topic_substring():
    node = _bare_node(freqs={"default": 20.0, "odom": 10.0, "power": 2.0})
    assert node._resolve_throttle_ms("/env_0/odom") == 10.0
    assert node._resolve_throttle_ms("/env_0/power_publisher/power") == 2.0
    assert node._resolve_throttle_ms("/env_0/scan") == 20.0


def test_resolve_throttle_ms_is_case_insensitive():
    node = _bare_node(freqs={"default": 20.0, "odom": 10.0})
    assert node._resolve_throttle_ms("/ENV_0/ODOM") == 10.0


def test_resolve_throttle_ms_default_fallback_when_missing():
    node = _bare_node(freqs={"odom": 5.0})
    assert node._resolve_throttle_ms("/scan") == 20.0
    node = _bare_node(freqs={})
    assert node._resolve_throttle_ms("/scan") == 20.0


def test_throttled_callback_respects_throttle_window():
    node = _bare_node(freqs={"default": 10.0})
    node.current_time = 0
    node._write_to_bag_at = MagicMock()
    cb = node._create_throttled_callback("/cmd_vel")
    cb("m1")
    node._write_to_bag_at.assert_not_called()  # (now - last) / 1e6 == 0 < 10
    node.current_time = 5_000_000
    cb("m2")
    node._write_to_bag_at.assert_not_called()
    node.current_time = 10_000_000
    cb("m3")
    node._write_to_bag_at.assert_called_once_with("/cmd_vel", "m3", 10_000_000)
    node.current_time = 10_000_000
    cb("m4")
    assert node._write_to_bag_at.call_count == 1
    node.current_time = 20_000_000
    cb("m5")
    assert node._write_to_bag_at.call_count == 2


def test_throttled_callback_buffers_before_first_clock():
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    cb = node._create_throttled_callback("/cmd_vel")
    cb("m1")
    assert node._pre_clock_buffer == [("/cmd_vel", "m1")]
    node._write_to_bag_at.assert_not_called()


def test_throttled_callback_drops_message_when_clock_buffer_already_flushed():
    # after the first clock tick flushes a non-empty buffer, the buffer is set
    # to None; a pre-clock message arriving afterwards is simply discarded
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    node._pre_clock_buffer = [("/a", "m0")]
    node.clock_callback(Clock())  # flush path drops _pre_clock_buffer
    node._write_to_bag_at.assert_called_once_with("/a", "m0", 0)
    node._write_to_bag_at.reset_mock()
    assert node._pre_clock_buffer is None
    node.current_time = None
    cb = node._create_throttled_callback("/cmd_vel")
    cb("m1")
    node._write_to_bag_at.assert_not_called()


def test_unthrottled_callback_writes_every_message():
    node = _bare_node(current_time=1000)
    node._write_to_bag_at = MagicMock()
    cb = node._create_unthrottled_callback("/scan")
    cb("a")
    cb("b")
    assert node._write_to_bag_at.call_count == 2


def test_unthrottled_callback_buffers_before_first_clock():
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    cb = node._create_unthrottled_callback("/scan")
    cb("a")
    assert node._pre_clock_buffer == [("/scan", "a")]
    node._write_to_bag_at.assert_not_called()


def test_unthrottled_callback_discards_message_when_clock_buffer_already_flushed():
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    node._pre_clock_buffer = [("/a", "m0")]
    node.clock_callback(Clock())
    node._write_to_bag_at.reset_mock()
    node.current_time = None
    cb = node._create_unthrottled_callback("/scan")
    cb("a")
    node._write_to_bag_at.assert_not_called()


# ---------------------------------------------------------------------------
# Clock handling
# ---------------------------------------------------------------------------


def test_clock_callback_computes_sim_time_ns():
    node = _bare_node()
    msg = Clock()
    msg.clock.sec = 10
    msg.clock.nanosec = 5
    node.clock_callback(msg)
    assert node.current_time == 10 * 10**9 + 5
    assert node._clock_received_count == 1


def test_clock_callback_backward_jump_clears_throttle_timers():
    node = _bare_node(current_time=10_000_000_000, last_recorded_times={"/t": 5})
    msg = Clock()
    msg.clock.sec = 5
    node.clock_callback(msg)
    assert node.current_time == 5 * 10**9
    assert node.last_recorded_times == {}
    assert any("Backward time jump" in w for w in node.logger.warns)


def test_clock_callback_flushes_pre_clock_buffer_once():
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    node._pre_clock_buffer = [("/a", "m1"), ("/b", "m2")]
    msg = Clock()
    msg.clock.sec = 1
    node.clock_callback(msg)
    assert node.current_time == 10**9
    assert node._pre_clock_buffer is None
    node._write_to_bag_at.assert_has_calls([call("/a", "m1", 10**9), call("/b", "m2", 10**9)])
    node._write_to_bag_at.reset_mock()
    node.clock_callback(msg)
    node._write_to_bag_at.assert_not_called()


# ---------------------------------------------------------------------------
# /tf merging: _tf_callback / _flush_tf
# ---------------------------------------------------------------------------


def _tf_writes(node):
    """(timestamp, {child_frame: x}) for every /tf write on the mocked writer."""
    out = []
    for call_args in node._write_to_bag_at.call_args_list:
        topic, msg, ts = call_args.args
        assert topic == "/tf"
        out.append((ts, {t.child_frame_id: t.transform.translation.x for t in msg.transforms}))
    return out


def test_tf_callback_merges_pairs_of_one_window_into_a_single_message():
    node = _bare_node(current_time=0)
    node._write_to_bag_at = MagicMock()
    node._tf_callback(_tf_message(("map", "odom", 1.0)))
    node.current_time = 10_000_000
    node._tf_callback(_tf_message(("odom", "base_link", 2.0)))

    node._flush_tf(10_000_000)  # inside the 20 ms window
    node._write_to_bag_at.assert_not_called()

    node._flush_tf(20_000_000)
    assert _tf_writes(node) == [(20_000_000, {"odom": 1.0, "base_link": 2.0})]
    assert node._tf_pending == {}
    assert node._tf_last_flush == 20_000_000


def test_tf_callback_keeps_the_newest_transform_per_pair():
    node = _bare_node(current_time=0)
    node._write_to_bag_at = MagicMock()
    node._tf_callback(_tf_message(("odom", "base_link", 1.0)))
    node.current_time = 5_000_000
    node._tf_callback(_tf_message(("odom", "base_link", 2.0)))

    node._flush_tf(20_000_000)
    assert _tf_writes(node) == [(20_000_000, {"base_link": 2.0})]


def test_tf_bone_frames_are_capped_while_root_frames_flush_every_window():
    node = _bare_node(current_time=0)
    node._write_to_bag_at = MagicMock()
    root = ("map", "body_env_0_agent_2", 0.0)
    bone = ("body_env_0_agent_2", "l_p_collar_env_0_agent_2", 0.0)

    for step in range(0, 400_000_001, 20_000_000):
        node.current_time = step
        node._tf_callback(_tf_message(root, bone))
        node._flush_tf(step)

    writes = _tf_writes(node)
    assert [ts for ts, _ in writes] == list(range(20_000_000, 400_000_001, 20_000_000))
    assert all("body_env_0_agent_2" in frames for _, frames in writes)
    bone_stamps = [ts for ts, frames in writes if "l_p_collar_env_0_agent_2" in frames]
    assert bone_stamps == [200_000_000, 400_000_000]


def test_tf_callback_buffers_before_first_clock_and_merges_on_first_tick():
    node = _bare_node()
    node._write_to_bag_at = MagicMock()
    node._tf_callback(_tf_message(("map", "odom", 1.0)))
    node._tf_callback(_tf_message(("map", "odom", 2.0), ("odom", "base_link", 3.0)))
    assert [topic for topic, _ in node._pre_clock_buffer] == ["/tf", "/tf"]
    node._write_to_bag_at.assert_not_called()

    msg = Clock()
    msg.clock.sec = 1
    node.clock_callback(msg)

    assert _tf_writes(node) == [(10**9, {"odom": 2.0, "base_link": 3.0})]
    assert node._pre_clock_buffer is None


def test_stop_episode_flushes_pending_tf_before_closing():
    node = _bare_node(current_sim_episode_id=3, current_time=50_000_000)
    node._close_current_writer = MagicMock()
    node._write_to_bag_at = MagicMock()
    node._tf_pending = {("map", "odom"): _transform("map", "odom", 1.0)}
    node._tf_last_flush = 45_000_000  # inside the window: the forced flush writes anyway

    node._stop_episode(3, outcome_state=2)

    assert _tf_writes(node) == [(50_000_000, {"odom": 1.0})]
    assert node._tf_pending == {}
    node._close_current_writer.assert_called_once()


def test_stop_episode_drops_pending_tf_when_no_clock_was_received():
    node = _bare_node(current_sim_episode_id=3)
    node._close_current_writer = MagicMock()
    node._write_to_bag_at = MagicMock()
    node._tf_pending = {("map", "odom"): _transform("map", "odom", 1.0)}

    node._stop_episode(3, outcome_state=2)

    node._write_to_bag_at.assert_not_called()
    assert node._tf_pending == {}


def test_clock_backward_jump_writes_pending_tf_at_the_pre_jump_time():
    node = _bare_node(current_time=10_000_000_000, last_recorded_times={"/t": 5})
    node._write_to_bag_at = MagicMock()
    node._tf_pending = {("map", "odom"): _transform("map", "odom", 1.0)}
    node._tf_last_written = {("map", "odom"): 9_000_000_000}
    node._tf_last_flush = 10_000_000_000
    msg = Clock()
    msg.clock.sec = 5

    node.clock_callback(msg)

    assert _tf_writes(node) == [(10_000_000_000, {"odom": 1.0})]
    assert node._tf_pending == {}
    assert node._tf_last_written == {}


def test_flush_tf_writes_nothing_when_nothing_is_pending():
    node = _bare_node(current_time=100_000_000)
    node._write_to_bag_at = MagicMock()
    node._flush_tf(100_000_000)
    node._flush_tf(100_000_000, force=True)
    node._write_to_bag_at.assert_not_called()
    assert node._tf_last_flush == 0


# ---------------------------------------------------------------------------
# Writer plumbing: _register_topic / _ensure_topic_in_bag / _write_to_bag_at
# ---------------------------------------------------------------------------


def test_register_topic_builds_ros_type_string():
    node = _bare_node()
    node._register_topic("/cmd_vel", Twist)
    meta = node._topic_registry["/cmd_vel"]
    assert isinstance(meta, rosbag2_py.TopicMetadata)
    assert meta.name == "cmd_vel"
    assert meta.type == "geometry_msgs/msg/Twist"
    assert meta.serialization_format == "cdr"


def test_register_topic_is_idempotent():
    node = _bare_node()
    node._register_topic("/cmd_vel", Twist)
    node._register_topic("/cmd_vel", String)  # second registration ignored
    assert node._topic_registry["/cmd_vel"].type == "geometry_msgs/msg/Twist"


def test_register_topic_fallback_to_str_when_no_module():
    node = _bare_node()
    node._register_topic("/mystery", object())
    assert node._topic_registry["/mystery"].type == str(object())


def test_ensure_topic_in_bag_creates_topic_once():
    node = _bare_node()
    node.writer = MagicMock()
    node._register_topic("/cmd_vel", Twist)
    node._ensure_topic_in_bag("/cmd_vel")
    node._ensure_topic_in_bag("/cmd_vel")
    node.writer.create_topic.assert_called_once()
    assert "cmd_vel" in node.topics_metadata


def test_ensure_topic_in_bag_unregistered_topic_is_ignored():
    node = _bare_node()
    node.writer = MagicMock()
    node._ensure_topic_in_bag("/not_registered")
    node.writer.create_topic.assert_not_called()


def test_write_to_bag_at_noop_when_shutting_down():
    node = _bare_node(is_shutting_down=True)
    node.writer = MagicMock()
    node._write_to_bag_at("/cmd_vel", String(), 1)
    node.writer.write.assert_not_called()


def test_write_to_bag_at_serialization_failure_logs_error(monkeypatch):
    node = _bare_node()
    node.writer = MagicMock()
    monkeypatch.setattr(recorder, "serialize_message", MagicMock(side_effect=RuntimeError("boom")))
    node._write_to_bag_at("/cmd_vel", String(), 1)
    node.writer.write.assert_not_called()
    assert any("Serialization failed" in e for e in node.logger.errors)


def test_write_to_bag_at_buffers_latched_topics_when_no_writer():
    node = _bare_node(latched_topic_names={"env_0/state/episode"})
    msg = String()
    msg.data = "x"
    node._write_to_bag_at("/env_0/state/episode", msg, 1)
    assert len(node._pre_episode_buffer) == 1
    topic, payload, ts = node._pre_episode_buffer[0]
    assert topic == "/env_0/state/episode"
    assert isinstance(payload, bytes)
    assert ts == 1
    assert node._write_drop_count == 0


def test_write_to_bag_at_drops_unlatched_topics_when_no_writer():
    node = _bare_node()
    node._write_to_bag_at("/cmd_vel", String(), 1)
    assert node._write_drop_count == 1
    assert node._pre_episode_buffer == []
    assert any("DROP (writer=None)" in w for w in node.logger.warns)


def test_write_to_bag_at_latched_buffer_is_capped():
    node = _bare_node(latched_topic_names={"state/episode"})
    node._pre_episode_buffer = [("t", b"b", 0)] * 10000
    node._write_to_bag_at("/state/episode", String(), 1)
    assert node._write_drop_count == 1


def test_write_to_bag_at_drop_warning_repeats_every_100_drops():
    node = _bare_node()
    for _ in range(100):
        node._write_to_bag_at("/cmd_vel", String(), 1)
    assert node._write_drop_count == 100
    drop_warns = [w for w in node.logger.warns if "DROP (writer=None)" in w]
    # drops 1..10 warn (drop_count <= 10), then drop #100 (drop_count % 100 == 0)
    assert len(drop_warns) == 11
    assert any("drop_count=100" in w for w in drop_warns)


def test_write_to_bag_at_serialization_failure_silent_while_shutting_down(monkeypatch):
    node = _bare_node(is_shutting_down=True)
    monkeypatch.setattr(recorder, "serialize_message", MagicMock(side_effect=RuntimeError("boom")))
    node._write_to_bag_at("/cmd_vel", String(), 1)
    assert not any("Serialization failed" in e for e in node.logger.errors)


def test_write_to_bag_at_writes_serialized_message_via_writer():
    node = _bare_node()
    node.writer = MagicMock()
    node._register_topic("/cmd_vel", Twist)
    node._write_to_bag_at("/cmd_vel", String(), 999)
    node.writer.create_topic.assert_called_once()
    node.writer.write.assert_called_once_with("cmd_vel", ANY, 999)
    assert "cmd_vel" in node.recorded_topics
    assert node._write_success_count == 1
    assert any("Write success count" in i for i in node.logger.infos)


def test_write_to_bag_at_unregistered_topic_skips_create_topic():
    node = _bare_node()
    node.writer = MagicMock()
    node._write_to_bag_at("/mystery", String(), 1)
    node.writer.create_topic.assert_not_called()
    node.writer.write.assert_called_once()


def test_write_to_bag_at_writer_error_is_logged():
    node = _bare_node()
    node.writer = MagicMock()
    node.writer.write.side_effect = RuntimeError("disk full")
    node._write_to_bag_at("/cmd_vel", String(), 1)
    assert any("WRITE ERROR" in e for e in node.logger.errors)


# ---------------------------------------------------------------------------
# Episode lifecycle: _begin_episode / _stop_episode / start_episode service
# ---------------------------------------------------------------------------


def test_begin_episode_opens_new_episode_once():
    node = _bare_node()
    node._start_episode_recording = MagicMock()
    assert node._begin_episode(7, source="episode_record") is True
    assert node._begin_episode(7) is False
    assert node.episodes_recorded == 1
    node._start_episode_recording.assert_called_once_with(7)


def test_stop_episode_ignores_stale_episode():
    node = _bare_node(current_sim_episode_id=3)
    node._close_current_writer = MagicMock()
    node._stop_episode(5, outcome_state=2)
    node._close_current_writer.assert_not_called()


def test_stop_episode_writes_outcome_and_closes(tmp_path):
    node = _bare_node(
        current_sim_episode_id=3,
        current_metadata=_make_run_metadata(),
        current_metadata_path=tmp_path / "episode_003.yaml",
        _pre_episode_buffer=[("t", b"b", 0)],
    )
    node._close_current_writer = MagicMock()
    node._stop_episode(3, outcome_state=2, outcome_info="goal reached")
    node._close_current_writer.assert_called_once()
    assert node._pre_episode_buffer == []
    data = yaml.safe_load((tmp_path / "episode_003.yaml").read_text())
    assert data["outcome_state"] == 2
    assert data["outcome_info"] == "goal reached"


def test_stop_episode_metadata_write_failure_still_closes(tmp_path, monkeypatch):
    node = _bare_node(
        current_sim_episode_id=3,
        current_metadata=_make_run_metadata(),
        current_metadata_path=tmp_path / "episode_003.yaml",
    )
    node._close_current_writer = MagicMock()
    monkeypatch.setattr(MetadataWriter, "write", MagicMock(side_effect=RuntimeError("boom")))
    node._stop_episode(3, outcome_state=2)
    node._close_current_writer.assert_called_once()
    assert any("Failed to write outcome metadata" in e for e in node.logger.errors)


def _service_request(**overrides):
    base = dict(
        episode_id=9,
        command=1,
        COMMAND_START=1,
        COMMAND_STOP=2,
        outcome_state=0,
        outcome_info="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_service_callback_start_command():
    node = _bare_node(current_episode_id=4)
    node._begin_episode = MagicMock(return_value=True)
    resp = SimpleNamespace()
    node._start_episode_service_callback(_service_request(command=1), resp)
    assert resp.success is True
    assert resp.message == "recording episode 4"
    node._begin_episode.assert_called_once_with(9, source="runner")


def test_service_callback_stop_command():
    node = _bare_node()
    node._stop_episode = MagicMock()
    resp = SimpleNamespace()
    node._start_episode_service_callback(_service_request(command=2, outcome_state=3, outcome_info="collision"), resp)
    assert resp.success is True
    assert resp.message == "stopped"
    node._stop_episode.assert_called_once_with(9, outcome_state=3, outcome_info="collision")


def test_service_callback_unknown_command():
    node = _bare_node()
    resp = SimpleNamespace()
    node._start_episode_service_callback(_service_request(command=7), resp)
    assert resp.success is False
    assert resp.message == "unknown command 7"


# ---------------------------------------------------------------------------
# Topic callbacks: episode_record / semantic_snapshot / robots_fleet
# ---------------------------------------------------------------------------


def _episode_message(episode_id=3, outcome=1):
    msg = EpisodeRecord()
    msg.episode_id = episode_id
    msg.outcome_state = outcome
    msg.outcome_info = ""
    return msg


def test_episode_record_callback_begins_new_episode():
    node = _bare_node(current_time=5000)
    node.get_namespace = lambda: "/env_0"
    node._begin_episode = MagicMock(return_value=True)
    node._update_metadata_from_episode = MagicMock()
    node._write_to_bag_at = MagicMock()
    node.episode_record_callback(_episode_message(outcome=1))
    node._begin_episode.assert_called_once_with(3, source="episode_record")
    node._update_metadata_from_episode.assert_called_once()
    node._write_to_bag_at.assert_called_once_with("/env_0/state/episode", ANY, 5000)


@pytest.mark.parametrize("outcome", [3, 9])
def test_episode_record_callback_does_not_begin_finished_episodes(outcome):
    node = _bare_node(current_time=1)
    node.get_namespace = lambda: ""
    node._begin_episode = MagicMock()
    node._update_metadata_from_episode = MagicMock()
    node._write_to_bag_at = MagicMock()
    node.episode_record_callback(_episode_message(outcome=outcome))
    node._begin_episode.assert_not_called()
    node._write_to_bag_at.assert_called_once_with("/state/episode", ANY, 1)


def test_episode_record_callback_does_not_reopen_seen_episode():
    node = _bare_node(current_time=1, _seen_episodes={3})
    node.get_namespace = lambda: ""
    node._begin_episode = MagicMock()
    node._update_metadata_from_episode = MagicMock()
    node._write_to_bag_at = MagicMock()
    node.episode_record_callback(_episode_message(outcome=0))
    node._begin_episode.assert_not_called()


def test_semantic_snapshot_callback_writes_state_semantics():
    node = _bare_node(current_time=777)
    node.get_namespace = lambda: "/env_0"
    node._write_to_bag_at = MagicMock()
    node.semantic_snapshot_callback("snap")
    node._write_to_bag_at.assert_called_once_with("/env_0/state/semantics", "snap", 777)


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


def test_robots_fleet_callback_writes_and_discovers_robots(tmp_path, monkeypatch):
    node = _bare_node(current_time=2000)
    node.get_namespace = lambda: "/env_0"
    node._write_to_bag_at = MagicMock()
    node.create_subscription = MagicMock()
    node.current_metadata = _make_run_metadata(robot_model=["unknown"])
    node.current_metadata_path = tmp_path / "episode_000.yaml"
    write_spy = MagicMock(side_effect=MetadataWriter.write)
    monkeypatch.setattr(MetadataWriter, "write", write_spy)

    node.robots_fleet_callback(_fleet_message([("robot_0", "jackal")]))

    node._write_to_bag_at.assert_called_once_with("/env_0/state/robots", ANY, 2000)
    assert "robot_0" in node.known_robots
    assert node.robot_model == "jackal"
    assert node.current_metadata.robot_model == ["jackal"]
    write_spy.assert_called_once()
    # 15 of the 21 per-robot topics (state/peds/tf topics skipped) + the model's controller odom and cmd_vel
    assert node.create_subscription.call_count == 17
    assert (tmp_path / "episode_000.yaml").exists()

    # second sighting of the same robot: no re-subscription
    node.robots_fleet_callback(_fleet_message([("robot_0", "jackal")]))
    assert node.create_subscription.call_count == 17

    # a new robot triggers a new subscription wave, no controller topics for a model without model_params
    node.robots_fleet_callback(_fleet_message([("robot_1", "turtlebot3")]))
    assert node.create_subscription.call_count == 32
    assert "robot_1" in node.known_robots
    assert node.current_metadata.robot_model == ["jackal", "turtlebot3"]


def test_robots_fleet_callback_without_metadata_does_not_write(monkeypatch):
    node = _bare_node(current_time=1)
    node.get_namespace = lambda: ""
    node._write_to_bag_at = MagicMock()
    node.create_subscription = MagicMock()
    write_spy = MagicMock()
    monkeypatch.setattr(MetadataWriter, "write", write_spy)
    node.robots_fleet_callback(_fleet_message([("robot_0", "jackal")]))
    write_spy.assert_not_called()
    assert node.robot_model == "jackal"


# ---------------------------------------------------------------------------
# discover_topics
# ---------------------------------------------------------------------------


def _discover_node(**overrides):
    node = _bare_node(**overrides)
    node.get_namespace = lambda: "/env_0"
    node.get_topic_names_and_types = lambda: []
    return node


def test_discover_topics_subscribes_odom_and_infers_robot_model(tmp_path, monkeypatch):
    node = _discover_node()
    node._subscribe_discovered = MagicMock()
    node.get_topic_names_and_types = lambda: [
        ("/env_0/robot/jackal/odom", ["nav_msgs/msg/Odometry"]),
    ]
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode_000.yaml"
    write_spy = MagicMock(side_effect=MetadataWriter.write)
    monkeypatch.setattr(MetadataWriter, "write", write_spy)

    node.discover_topics()

    node._subscribe_discovered.assert_called_once()
    assert node._subscribe_discovered.call_args.args[0] == "/env_0/robot/jackal/odom"
    assert node.robot_model == "jackal"
    assert node.current_metadata.robot_model == ["jackal"]
    write_spy.assert_called_once()


def test_discover_topics_skips_registered_and_foreign_topics():
    node = _discover_node(robot_model="jackal")
    node._subscribe_discovered = MagicMock()
    node._topic_registry["/env_0/odom"] = "registered"
    node.get_topic_names_and_types = lambda: [
        ("/env_0/odom", ["nav_msgs/msg/Odometry"]),
        ("/other/odom", ["nav_msgs/msg/Odometry"]),
    ]
    node.discover_topics()
    node._subscribe_discovered.assert_not_called()


def test_discover_topics_ignores_scan_topics():
    node = _discover_node()
    node._subscribe_discovered = MagicMock()
    node.get_topic_names_and_types = lambda: [
        ("/env_0/front_lidar", ["sensor_msgs/msg/LaserScan"]),
        ("/env_0/scan", ["sensor_msgs/msg/LaserScan"]),
    ]
    node.discover_topics()
    node._subscribe_discovered.assert_not_called()
    assert node.robot_model == "unknown"


def test_discover_topics_skips_robot_model_inference_when_known(tmp_path, monkeypatch):
    node = _discover_node(robot_model="jackal")
    node._subscribe_discovered = MagicMock()
    node.get_topic_names_and_types = lambda: [
        ("/env_0/robot/jackal/odom", ["nav_msgs/msg/Odometry"]),
    ]
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode_000.yaml"
    write_spy = MagicMock()
    monkeypatch.setattr(MetadataWriter, "write", write_spy)
    node.discover_topics()
    write_spy.assert_not_called()
    assert node.robot_model == "jackal"


# ---------------------------------------------------------------------------
# Episode metadata: write / finalize / update-from-EpisodeRecord
# ---------------------------------------------------------------------------


def test_write_episode_metadata_writes_yaml(tmp_path):
    node = _bare_node(
        benchmark_id="b1",
        planner="cont-teb",
        stage="s1",
        map_name="m1",
        robot_model="jackal",
        env_ns_root="/env_0",
        is_reference=True,
        reference_type="golden",
        suite_name="s",
        contest_name="c",
        episodes_requested=10,
        local_planner="",
        inter_planner="",
        current_sim_episode_id=7,
        current_metadata_path=tmp_path / "episode_000.yaml",
    )
    node._write_episode_metadata(0)
    data = yaml.safe_load((tmp_path / "episode_000.yaml").read_text())
    assert data["benchmark_id"] == "b1"
    assert data["planner"] == "cont-teb"
    assert data["episode_id"] == 0
    assert data["task_generator_episode_id"] == 7
    assert data["is_reference"] is True
    assert data["reference_type"] == "golden"
    assert data["env_ns_root"] == "/env_0"
    assert node.current_metadata is not None


def test_write_episode_metadata_failure_keeps_metadata(tmp_path, monkeypatch):
    node = _bare_node(
        benchmark_id="b1",
        planner="p",
        stage="s",
        map_name="m",
        robot_model="jackal",
        current_metadata_path=tmp_path / "episode_000.yaml",
    )
    monkeypatch.setattr(MetadataWriter, "write", MagicMock(side_effect=RuntimeError("boom")))
    node._write_episode_metadata(0)
    assert node.current_metadata is not None
    assert any("Failed to write initial episode metadata" in e for e in node.logger.errors)


def test_finalize_episode_metadata_sets_stats(tmp_path):
    node = _bare_node()
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode.yaml"
    node.recorded_topics = {"env_0/arena_peds", "env_0/cmd_vel"}
    node._finalize_episode_metadata()
    md = node.current_metadata
    assert md.pedsim_available is True
    assert md.recorded_topics == ["env_0/arena_peds", "env_0/cmd_vel"]
    assert md.recording_ended_at is not None
    data = yaml.safe_load((tmp_path / "episode.yaml").read_text())
    assert data["pedsim_available"] is True


def test_finalize_episode_metadata_agent_states_counts_as_pedsim(tmp_path):
    node = _bare_node()
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode.yaml"
    node.recorded_topics = {"robot_0/agent_states"}
    node._finalize_episode_metadata()
    assert node.current_metadata.pedsim_available is True


def test_finalize_episode_metadata_no_pedsim_topics(tmp_path):
    node = _bare_node()
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode.yaml"
    node.recorded_topics = {"env_0/odom"}
    node._finalize_episode_metadata()
    assert node.current_metadata.pedsim_available is False


def test_finalize_episode_metadata_noop_without_metadata(monkeypatch):
    node = _bare_node(current_metadata=None)
    write_spy = MagicMock()
    monkeypatch.setattr(MetadataWriter, "write", write_spy)
    node._finalize_episode_metadata()
    write_spy.assert_not_called()


def test_finalize_episode_metadata_write_failure_logged(tmp_path, monkeypatch):
    node = _bare_node()
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = tmp_path / "episode.yaml"
    monkeypatch.setattr(MetadataWriter, "write", MagicMock(side_effect=RuntimeError("boom")))
    node._finalize_episode_metadata()
    assert any("Failed to finalize episode metadata" in e for e in node.logger.errors)


def test_update_metadata_from_episode_populates_fields(tmp_path):
    node = _bare_node(current_metadata=_make_run_metadata(robot_model=["unknown"], map="unknown"))
    node.current_metadata_path = tmp_path / "episode.yaml"
    pv_double = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.4)
    pv_string = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value="turtlebot3")
    msg = EpisodeRecord()
    msg.episode_id = 7
    msg.world = "hospital_1"
    msg.robots = ["jackal"]
    msg.tm_robots = "tm_robots"
    msg.tm_obstacles = "tm_obstacles"
    msg.tm_modules = ["spawn", "nav"]
    msg.obstacles_params = [RclParameter(name="robot.radius", value=pv_double)]
    msg.robots_params = [RclParameter(name="model", value=pv_string)]

    node._update_metadata_from_episode(msg)

    md = node.current_metadata
    assert md.robot_model == ["jackal"]
    assert md.map == "hospital_1"
    assert md.tm_obstacles == "tm_obstacles"
    assert md.tm_robots == "tm_robots"
    assert md.tm_modules == ["spawn", "nav"]
    assert md.obstacles_params == {"robot": {"radius": 0.4}}
    assert md.robots_params == {"model": "turtlebot3"}
    assert yaml.safe_load((tmp_path / "episode.yaml").read_text())["map"] == "hospital_1"


def test_update_metadata_from_episode_noop_without_metadata(monkeypatch):
    node = _bare_node(current_metadata=None)
    write_spy = MagicMock()
    monkeypatch.setattr(MetadataWriter, "write", write_spy)
    node._update_metadata_from_episode(_episode_message())
    write_spy.assert_not_called()


def test_update_metadata_from_episode_write_failure_logged(tmp_path, monkeypatch):
    node = _bare_node(current_metadata=_make_run_metadata())
    node.current_metadata_path = tmp_path / "episode.yaml"
    monkeypatch.setattr(MetadataWriter, "write", MagicMock(side_effect=RuntimeError("boom")))
    node._update_metadata_from_episode(_episode_message())
    assert any("Failed to update episode metadata" in e for e in node.logger.errors)


# ---------------------------------------------------------------------------
# _param_value_to_py / _unflatten_dict
# ---------------------------------------------------------------------------


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


def test_param_value_to_py_unknown_type_falls_back_to_str():
    node = _bare_node()
    pv = ParameterValue(type=99)
    assert node._param_value_to_py(pv) == str(pv)


def test_unflatten_dict_nested_and_scalar():
    node = _bare_node()
    assert node._unflatten_dict({"a.b.c": 1, "a.b.d": 2, "e": 3}) == {
        "a": {"b": {"c": 1, "d": 2}},
        "e": 3,
    }


def test_unflatten_dict_empty():
    node = _bare_node()
    assert node._unflatten_dict({}) == {}


# ---------------------------------------------------------------------------
# _start_episode_recording
# ---------------------------------------------------------------------------


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


def test_start_episode_recording_opens_writer_and_flushes_buffer(tmp_path, monkeypatch):
    node = _recording_node(tmp_path, current_sim_episode_id=None)
    node._pre_episode_buffer = [("/env_0/state/episode", b"\x01\x02", 12345)]
    fake_writer = MagicMock()
    monkeypatch.setattr(recorder.rosbag2_py, "SequentialWriter", MagicMock(return_value=fake_writer))
    storage_calls = []

    class _FakeStorageOptions:
        def __init__(self, *args, **kwargs):
            storage_calls.append((args, kwargs))
            self.__dict__.update(kwargs)

    monkeypatch.setattr(recorder.rosbag2_py, "StorageOptions", _FakeStorageOptions)

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

    fake_writer.open.assert_called_once()
    assert storage_calls[0][1]["uri"] == str(ep_dir / "episode_000")
    assert storage_calls[0][1]["storage_id"] == "mcap"
    fake_writer.write.assert_called_once_with("env_0/state/episode", b"\x01\x02", 12345)
    assert "env_0/state/episode" in node.recorded_topics
    assert node._pre_episode_buffer == []
    assert node._write_success_count == 0


def test_start_episode_recording_uses_mcap_config_when_present(tmp_path, monkeypatch):
    node = _recording_node(tmp_path)
    config_dir = tmp_path / "share" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "mcap_writer_options.yaml").write_text("compression: zstd\n")
    fake_writer = MagicMock()
    monkeypatch.setattr(recorder.rosbag2_py, "SequentialWriter", MagicMock(return_value=fake_writer))
    storage_calls = []

    class _FakeStorageOptions:
        def __init__(self, *args, **kwargs):
            storage_calls.append((args, kwargs))
            self.__dict__.update(kwargs)

    monkeypatch.setattr(recorder.rosbag2_py, "StorageOptions", _FakeStorageOptions)

    node._start_episode_recording(1)

    assert storage_calls[0][1]["storage_config_uri"] == str(config_dir / "mcap_writer_options.yaml")


def test_constructor_tolerates_chmod_failure(tmp_path, fake_share, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "chmod", MagicMock(side_effect=OSError("chmod denied")))
    node = _build_full_node(monkeypatch, ["pytest", "--dir", str(tmp_path / "ep")])
    try:
        assert node.episodes_root == (tmp_path / "ep" / "episodes").resolve()
    finally:
        node.destroy_node()


def test_start_episode_recording_writer_failure_sets_writer_none(tmp_path, monkeypatch):
    node = _recording_node(tmp_path)
    fake_writer = MagicMock()
    fake_writer.open.side_effect = RuntimeError("mcap open failed")
    monkeypatch.setattr(recorder.rosbag2_py, "SequentialWriter", MagicMock(return_value=fake_writer))

    node._start_episode_recording(1)

    assert node.writer is None
    assert node._episode_id_offset == 1
    assert any("Failed to open MCAP writer" in e for e in node.logger.errors)
    assert (node.episodes_root / "episode_000" / "episode_000.yaml").exists()


def test_start_episode_recording_increments_global_episode_id(tmp_path, monkeypatch):
    node = _recording_node(tmp_path)
    fake_writer = MagicMock()
    monkeypatch.setattr(recorder.rosbag2_py, "SequentialWriter", MagicMock(return_value=fake_writer))

    node._start_episode_recording(1)
    node._start_episode_recording(2)

    assert node.current_episode_id == 1
    assert node._episode_id_offset == 2
    assert (node.episodes_root / "episode_001").is_dir()
    assert (node.episodes_root / "episode_001" / "episode_001.yaml").exists()
    assert fake_writer.open.call_count == 2
    assert fake_writer.close.call_count == 1  # previous writer closed first


# ---------------------------------------------------------------------------
# _close_current_writer (flush + flatten + rosbag metadata merge)
# ---------------------------------------------------------------------------


def _episode_dir_with_bag(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"mcap-bytes")
    (inner / "episode_000_1.mcap").write_bytes(b"mcap-bytes")
    (inner / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  message_count: 42\n  topics_with_message_count:\n    - topic_metadata:\n        name: /cmd_vel\n      message_count: 42\n")
    return ep_dir


def test_close_current_writer_flattens_mcap_files_and_merges_rosbag_metadata(tmp_path):
    ep_dir = _episode_dir_with_bag(tmp_path)
    node = _bare_node()
    writer = MagicMock()
    node.writer = writer
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"

    node._close_current_writer()

    writer.close.assert_called_once()
    assert node.writer is None
    # first mcap becomes episode_000.mcap; the second is suffixed to avoid collision
    assert (ep_dir / "episode_000.mcap").exists()
    assert (ep_dir / "episode_000_episode_000_1.mcap").exists()
    inner = ep_dir / "episode_000"
    assert not (inner / "metadata.yaml").exists()
    assert not inner.exists()
    assert node.current_metadata.rosbag2_message_count == 42
    assert node.current_metadata.rosbag2_topics[0]["message_count"] == 42
    data = yaml.safe_load((ep_dir / "episode_000.yaml").read_text())
    assert data["recording_ended_at"] is not None
    assert data["pedsim_available"] is False
    assert data["recorded_topics"] == []


def test_close_current_writer_noop_without_writer(tmp_path):
    ep_dir = _episode_dir_with_bag(tmp_path)
    node = _bare_node()
    node.writer = None
    node.current_episode_dir = ep_dir
    node._close_current_writer()
    assert (ep_dir / "episode_000" / "episode_000_0.mcap").exists()
    assert (ep_dir / "episode_000" / "metadata.yaml").exists()


def test_close_current_writer_close_error_is_logged(tmp_path):
    ep_dir = _episode_dir_with_bag(tmp_path)
    node = _bare_node()
    node.writer = MagicMock()
    node.writer.close.side_effect = RuntimeError("close failed")
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert node.writer is None
    assert any("Error closing MCAP writer" in e for e in node.logger.errors)
    assert (ep_dir / "episode_000.mcap").exists()


def test_close_current_writer_missing_rosbag_metadata_is_tolerated(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"x")
    node = _bare_node()
    node.writer = MagicMock()
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert (ep_dir / "episode_000.mcap").exists()
    assert node.current_metadata.rosbag2_message_count is None


def test_close_current_writer_move_failure_is_logged(tmp_path, monkeypatch):
    ep_dir = _episode_dir_with_bag(tmp_path)
    node = _bare_node()
    node.writer = MagicMock()
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    monkeypatch.setattr(recorder.shutil, "move", MagicMock(side_effect=OSError("move failed")))

    node._close_current_writer()

    assert any("Failed to flatten" in e for e in node.logger.errors)
    assert (ep_dir / "episode_000" / "episode_000_0.mcap").exists()  # untouched


def test_close_current_writer_rmdir_failure_is_tolerated(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"x")
    (inner / "stray.bin").write_bytes(b"junk")  # keeps inner dir non-empty
    node = _bare_node()
    node.writer = MagicMock()
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert any("Failed to remove inner dir" in w for w in node.logger.warns)
    assert (inner / "stray.bin").exists()
    assert (ep_dir / "episode_000.mcap").exists()


def test_close_current_writer_corrupt_rosbag_metadata_is_logged(tmp_path):
    ep_dir = tmp_path / "episode_000"
    inner = ep_dir / "episode_000"
    inner.mkdir(parents=True)
    (inner / "episode_000_0.mcap").write_bytes(b"x")
    (inner / "metadata.yaml").write_text("{{{{ not yaml")
    node = _bare_node()
    node.writer = MagicMock()
    node.current_episode_dir = ep_dir
    node.current_metadata = _make_run_metadata()
    node.current_metadata_path = ep_dir / "episode_000.yaml"
    node._close_current_writer()
    assert any("Failed to merge rosbag metadata" in e for e in node.logger.errors)
    assert node.current_metadata.rosbag2_message_count is None


# ---------------------------------------------------------------------------
# Shutdown / lifecycle
# ---------------------------------------------------------------------------


def test_finalize_closes_writer_once(capsys):
    node = _bare_node()
    node._close_current_writer = MagicMock()
    node.finalize()
    assert node.is_shutting_down is True
    node.finalize()  # second call is a no-op
    node._close_current_writer.assert_called_once()
    out = capsys.readouterr().out
    assert "finalize() called" in out


def test_destroy_node_calls_finalize(monkeypatch):
    node = _bare_node()
    monkeypatch.setattr(recorder.Node, "destroy_node", lambda self: None)
    node.destroy_node()
    assert node.is_shutting_down is True


def test_log_methods_print_when_rclpy_not_initialized(monkeypatch, capsys):
    node = _bare_node()
    monkeypatch.setattr(recorder.rclpy, "ok", lambda: False)
    node._log_info("info-msg")
    node._log_warn("warn-msg")
    node._log_error("error-msg")
    out = capsys.readouterr().out
    assert "[DataRecorder] [INFO] info-msg" in out
    assert "[DataRecorder] [WARN] warn-msg" in out
    assert "[DataRecorder] [ERROR] error-msg" in out


def test_log_methods_print_when_logger_raises(monkeypatch, capsys):
    node = _bare_node()
    node.get_logger = MagicMock(side_effect=RuntimeError("no logger"))
    node._log_warn("warn-msg")
    node._log_error("error-msg")
    out = capsys.readouterr().out
    assert "[DataRecorder] [WARN] warn-msg" in out
    assert "[DataRecorder] [ERROR] error-msg" in out


def test_log_methods_write_to_open_log_file(tmp_path):
    node = _bare_node()
    log_file = open(tmp_path / "recorder.log", "w", buffering=1)
    node.log_file = log_file
    try:
        node._log_info("info-msg")
        node._log_warn("warn-msg")
        node._log_error("error-msg")
        log_file.flush()
    finally:
        log_file.close()
    content = (tmp_path / "recorder.log").read_text()
    assert "[DataRecorder] [INFO] info-msg" in content
    assert "[DataRecorder] [WARN] warn-msg" in content
    assert "[DataRecorder] [ERROR] error-msg" in content


def test_finalize_tolerates_signal_registration_failure(monkeypatch):
    node = _bare_node()
    monkeypatch.setattr(recorder.signal, "signal", MagicMock(side_effect=ValueError("main thread")))
    node.finalize()
    assert node.is_shutting_down is True


def _fake_signal_module():
    fake = MagicMock()
    fake.SIGTERM = 15
    fake.SIGINT = 2
    fake.SIG_IGN = 1
    return fake


def test_main_success_path(monkeypatch):
    fake_rclpy = MagicMock()
    fake_rclpy.ok.return_value = True
    monkeypatch.setattr(recorder, "rclpy", fake_rclpy)
    node = MagicMock()
    monkeypatch.setattr(recorder, "DataRecorderNode", MagicMock(return_value=node))
    executor = MagicMock()
    executor.spin.side_effect = KeyboardInterrupt
    monkeypatch.setattr(recorder, "MultiThreadedExecutor", MagicMock(return_value=executor))
    monkeypatch.setattr(recorder, "signal", _fake_signal_module())
    exit_mock = MagicMock()
    monkeypatch.setattr(recorder.os, "_exit", exit_mock)

    recorder.main(args=["test-args"])

    fake_rclpy.init.assert_called_once_with(args=["test-args"])
    node.finalize.assert_called_once()
    executor.shutdown.assert_called_once()
    node.destroy_node.assert_called_once()
    fake_rclpy.shutdown.assert_called_once()
    exit_mock.assert_called_once_with(0)


def test_main_exception_path_exits_nonzero(monkeypatch):
    fake_rclpy = MagicMock()
    fake_rclpy.ok.return_value = True
    monkeypatch.setattr(recorder, "rclpy", fake_rclpy)
    monkeypatch.setattr(recorder, "DataRecorderNode", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(recorder, "signal", _fake_signal_module())
    # simulate a real process exit so main does not fall through to os._exit(0)
    monkeypatch.setattr(recorder.os, "_exit", MagicMock(side_effect=lambda code: (_ for _ in ()).throw(SystemExit(code))))

    with pytest.raises(SystemExit) as exc_info:
        recorder.main(args=None)

    assert exc_info.value.code == 1
    fake_rclpy.shutdown.assert_called_once()


def test_bind_to_parent_exits_when_orphaned(monkeypatch):
    exit_mock = MagicMock()
    monkeypatch.setattr(recorder.os, "_exit", exit_mock)
    monkeypatch.setattr(recorder.os, "getppid", lambda: 1)
    recorder._bind_to_parent()
    exit_mock.assert_called_once_with(0)


def test_bind_to_parent_tolerates_ctypes_failure(monkeypatch):
    import ctypes as _real_ctypes

    monkeypatch.setattr(_real_ctypes, "CDLL", MagicMock(side_effect=OSError("no libc")))
    monkeypatch.setattr(recorder.os, "getppid", lambda: 42)
    recorder._bind_to_parent()  # must not raise


# ---------------------------------------------------------------------------
# End-to-end service flow on a fully constructed node
# ---------------------------------------------------------------------------


def test_start_stop_episode_service_end_to_end(tmp_path, full_node, monkeypatch):
    fake_writer = MagicMock()
    monkeypatch.setattr(recorder.rosbag2_py, "SequentialWriter", MagicMock(return_value=fake_writer))

    start_req = _service_request(episode_id=11, command=1)
    start_resp = SimpleNamespace()
    full_node._start_episode_service_callback(start_req, start_resp)

    assert start_resp.success is True
    assert start_resp.message == "recording episode 0"
    assert full_node.current_episode_id == 0
    assert full_node.current_sim_episode_id == 11
    ep_dir = full_node.episodes_root / "episode_000"
    assert (ep_dir / "episode_000.yaml").exists()
    fake_writer.open.assert_called_once()

    stop_req = _service_request(episode_id=11, command=2, outcome_state=2, outcome_info="ok")
    stop_resp = SimpleNamespace()
    full_node._start_episode_service_callback(stop_req, stop_resp)

    assert stop_resp.success is True
    assert stop_resp.message == "stopped"
    fake_writer.close.assert_called_once()
    data = yaml.safe_load((ep_dir / "episode_000.yaml").read_text())
    assert data["outcome_state"] == 2
    assert data["outcome_info"] == "ok"
    assert data["recording_ended_at"] is not None
