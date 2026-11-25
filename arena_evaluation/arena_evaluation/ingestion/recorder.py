from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import re
import signal
import sys
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, TypedDict, Unpack

import rclpy
import rosbag2_py
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.params import Param, Range, declare
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.spin import create_executor, spin_node
from arena_runtime.env.task_layer import decode_modes
from arena_runtime_msgs.msg import EpisodeRecord, RecordedTopics, RobotFleet
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import serialize_message
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage

from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import caps_snapshot_path

from .episode_writer import EpisodeWriter
from .metadata import IngestionMetadata
from .topics import DEFAULT_THROTTLE_RATE_HZ

if TYPE_CHECKING:
    from arena_evaluation_msgs.srv import RecordEpisode
    from arena_runtime_msgs.msg import RecordedTopic
    from builtin_interfaces.msg import Time
    from rcl_interfaces.msg import ParameterValue
    from rclpy.context import Context

    from .topics import TopicDefinition

_NS_PER_SECOND: Final[int] = 1_000_000_000
_NS_PER_MILLISECOND: Final[int] = 1_000_000
_ODOM_THROTTLE_RATE_HZ: Final[float] = 50.0
_WORLD_FRAMES: Final[frozenset[str]] = frozenset({"map", "odom", "world"})


def min_interval_ns(rate_hz: float) -> int:
    """The smallest gap between two samples of a rate_hz stream in whole nanoseconds, 0 when unthrottled."""
    if rate_hz <= 0.0:
        return 0
    return math.ceil(_NS_PER_SECOND / rate_hz)


def resolve_rate_hz(overrides: Mapping[str, float], topic_name: str, default_hz: float) -> float:
    """The record_rates_hz entry of the exact topic name when present, else the topic's default rate."""
    return float(overrides.get(topic_name, default_hz))


def _checked_rates(rates: Mapping[str, object]) -> dict[str, float]:
    """A record_rates_hz mapping as floats, refusing a rate below 0 Hz."""
    checked: dict[str, float] = {}
    for topic_name, rate in rates.items():
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise TypeError(f"record_rates_hz[{topic_name!r}] must be numeric, got {type(rate).__name__}")
        value = float(rate)
        if value < 0.0:
            raise ValueError(f"record_rates_hz[{topic_name!r}] is {value}: a throttle rate is >= 0 Hz, 0 meaning unthrottled")
        checked[topic_name] = value
    return checked


def robot_from_frame(frame_id: str, env_namespace: str) -> str | None:
    """The robot segment before a frame name, None for a world frame or the environment itself."""
    parts = [part for part in frame_id.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    robot = parts[-2]
    env = env_namespace.strip("/").rsplit("/", 1)[-1]
    if robot in _WORLD_FRAMES or robot == env:
        return None
    return robot


def initialpose_topic(frame_id: str, env_namespace: str) -> str:
    """The topic an initial pose is recorded under: the robot its frame names, else the environment."""
    env = env_namespace.strip("/")
    env_root = f"/{env}" if env else ""
    robot = robot_from_frame(frame_id, env)
    return f"{env_root}/{robot}/initialpose" if robot else f"{env_root}/initialpose"


def _running_ns(history: dict[int, EpisodeRecord]) -> int | None:
    """The sim time the episode's RUNNING record names as its start, None before one with a stamp arrived."""
    running = history.get(EpisodeRecord.RUNNING)
    stamp = _stamp_ns(running.start_time) if running is not None else 0
    return stamp if stamp > 0 else None


def _stamp_ns(stamp: Time) -> int:
    """A ROS time stamp in nanoseconds."""
    return int(stamp.sec) * _NS_PER_SECOND + int(stamp.nanosec)


@dataclasses.dataclass(slots=True)
class RateGate:
    """Accept a nanosecond timestamp stream at most once per interval, both boundaries included."""

    interval_ns: int
    last_ns: int | None = None

    def accept(self, t_ns: int) -> bool:
        """Whether the sample at t_ns passes, opening the next interval from its own stamp."""
        if self.interval_ns <= 0:
            return True
        if self.last_ns is not None and t_ns - self.last_ns < self.interval_ns:
            return False
        self.last_ns = t_ns
        return True


def _state_types() -> dict[str, tuple[type, str, bool]]:
    """ros type name -> (msg class, topic basename, latched) of the state topics discovery subscribes in this env."""
    types: dict[str, tuple[type, str, bool]] = {}
    try:
        from arena_humansim_msgs.msg import AgentStates, InteractionFrame

        types["arena_humansim_msgs/msg/InteractionFrame"] = (
            InteractionFrame,
            "interactions",
            True,
        )
        types["arena_humansim_msgs/msg/AgentStates"] = (
            AgentStates,
            "agent_states",
            False,
        )
    except ImportError:
        pass
    try:
        from arena_people_msgs.msg import AnimationStates, Pedestrians

        types["arena_people_msgs/msg/AnimationStates"] = (
            AnimationStates,
            "animation_states",
            False,
        )
        types["arena_people_msgs/msg/Pedestrians"] = (Pedestrians, "arena_peds", False)
    except ImportError:
        pass
    return types


_OUTCOME_LABELS = {
    0: "QUEUED",
    1: "RUNNING",
    2: "SUCCESS",
    3: "FAILED",
    4: "SKIPPED",
    5: "FATAL",
}
_TERMINAL_OUTCOMES = {
    EpisodeRecord.SUCCESS,
    EpisodeRecord.FAILED,
    EpisodeRecord.SKIPPED,
    EpisodeRecord.FATAL,
}

_EPISODE_HISTORY_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    depth=20,
)
_DESERIALIZED_TOPIC_KEYS = frozenset({"episode_record", "robots_fleet", "tf", "tf_humans", "initialpose"})
_ENV_TOPIC_KEYS = frozenset(
    {
        "episode_record",
        "robots_fleet",
        "peds",
        "agent_states",
        "agent_meta",
        "interactions",
        "semantic_snapshot",
        "map",
        "door_mask",
        "tf",
        "tf_humans",
        "tf_static",
        "initialpose",
    }
)


class _NodeKwargs(TypedDict, total=False):
    context: Context | None
    cli_args: list[str] | None
    namespace: str | None
    use_global_arguments: bool
    enable_rosout: bool
    start_parameter_services: bool
    parameter_overrides: list[Parameter] | None
    allow_undeclared_parameters: bool
    enable_logger_service: bool


class _RecorderConfig(TypedDict, total=False):
    record_rates_hz: dict[str, float]
    tf_frames: list[tuple[str, float]]
    optional_topics: list[str]


_RECORDER_PARAMS = (
    Param[str](
        "record_data_dir",
        "",
        description="Directory the run is recorded into (empty or auto:/ = a new timestamped run).",
    ),
    Param[str](
        "benchmark_id",
        "unknown",
        description="Benchmark run id the recording belongs to.",
    ),
    Param[str]("contestant", "unknown", description="Contestant name of the recorded planner."),
    Param[str]("stage", "unknown", description="Benchmark stage the recording belongs to."),
    Param[str]("map", "unknown", description="Map the episodes run on."),
    Param[str]("world", "unknown", description="World the episodes run in."),
    Param[str]("suite_name", "", description="Benchmark suite the run belongs to."),
    Param[str]("contest_name", "", description="Contest the run belongs to."),
    Param[str]("local_planner", "", description="Local planner under test."),
    Param[str]("inter_planner", "", description="Intermediate planner under test."),
    Param[str]("robot", "", description="Robot model of the run."),
    Param[int](
        "episodes_requested",
        0,
        constraint=Range(lo=0),
        description="Number of episodes the run asked for (0 = unspecified).",
    ),
    Param[bool]("is_reference", False, description="Mark the recording as a reference run."),
    Param[str](
        "reference_type",
        "",
        description="Kind of reference the recording is (empty = none).",
    ),
    Param[int]("episode_id_offset", 0, description="Offset added to the recorded episode ids."),
    Param[str]("workspace_dir", "", description="Workspace directory the run was started from."),
    Param[str](
        "world.lighting",
        "authored",
        description="Lighting mode of the world the episodes run in.",
    ),
    Param[str](
        "kg.derive",
        "auto",
        description="Mode in which the knowledge graph of the world is derived.",
    ),
    Param[str](
        "data_recorder_autoprefix",
        "",
        description="Run id chosen for an automatic run directory (empty = generated at start).",
    ),
)


class DataRecorderNode(ROSParamServer, EpisodeWriter):
    def __init__(self, **node_kwargs: Unpack[_NodeKwargs]):
        super().__init__(
            "arena_evaluation_data_recorder",
            automatically_declare_parameters_from_overrides=True,
            **node_kwargs,
        )

        self.base_dir = get_package_share_directory("arena_evaluation")

        handles = {param.name: declare(self, param) for param in _RECORDER_PARAMS}

        record_data_dir = str(handles["record_data_dir"].value or "") or self._parse_cli_dir() or "auto:/"
        workspace_root = pathlib.Path(self.base_dir).parents[3]
        self.run_dir, record_data_dir_path = self._resolve_run_dirs(handles, record_data_dir, workspace_root)

        self.episodes_root = record_data_dir_path.resolve()
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        for directory in (self.run_dir, self.episodes_root):
            if directory.exists():
                try:
                    directory.chmod(0o777)
                except OSError:
                    pass

        self.benchmark_id = str(handles["benchmark_id"].value or "unknown")
        self.planner = str(handles["contestant"].value or "unknown")
        self.stage = str(handles["stage"].value or "unknown")
        self.map_name = str(handles["map"].value or "unknown")
        self.world = str(handles["world"].value or "unknown")
        self.suite_name = str(handles["suite_name"].value or "")
        self.contest_name = str(handles["contest_name"].value or "")
        self.local_planner = str(handles["local_planner"].value or "")
        self.inter_planner = str(handles["inter_planner"].value or "")
        self.episodes_requested = int(handles["episodes_requested"].value or 0)
        self.is_reference = bool(handles["is_reference"].value)
        ref_type = handles["reference_type"].value
        self.reference_type = ref_type if ref_type else None
        self._episode_id_offset = int(handles["episode_id_offset"].value or 0)
        self.workspace_dir = str(handles["workspace_dir"].value or "")
        self.world_lighting = str(handles["world.lighting"].value)
        self.kg_derive = str(handles["kg.derive"].value)
        self._snapshot_worlds: set[str] = set()

        env_namespace = self.get_namespace().strip("/")
        self.env_ns_root = f"/{env_namespace}" if env_namespace else ""

        self.robot_model = str(handles["robot"].value or "unknown")
        self.current_sim_episode_id: int | None = None
        self.known_robots = set()
        self._robot_params: dict[str, dict[str, str]] = {}
        self._robot_snapshots: dict[str, dict[str, Any]] = {}

        self.current_episode_id: int | None = None
        self.current_episode_dir: pathlib.Path | None = None
        self.current_metadata_path: pathlib.Path | None = None
        self.current_metadata = None

        self.log_file = None
        self.log_file_path: pathlib.Path | None = None
        self._open_log_file()

        self.config = self.read_config()
        self.record_rates_hz = self.config.get("record_rates_hz", {})
        self.tf_frames = [(re.compile(pattern), float(ms)) for pattern, ms in self.config.get("tf_frames", [])]

        self._log_info(f"Recorder ready. Episodes root: {self.episodes_root}")
        self._log_info(f"planner={self.planner!r} stage={self.stage!r} map={self.map_name!r} benchmark_id={self.benchmark_id!r}")

        self.current_time = None
        self._pre_clock_buffer = []
        self._clock_received_count = 0
        self._throttle_gates: dict[str, RateGate] = {}
        self.writer_lock = threading.Lock()
        # guards the tf frame-family state against the start_episode service callback group
        self._tf_lock = threading.Lock()
        self._tf_last_written: dict[tuple[str, str], int] = {}
        self.writer: rosbag2_py.SequentialWriter | None = None
        self.topics_metadata: dict[str, rosbag2_py.TopicMetadata] = {}
        self._topic_registry: dict[str, rosbag2_py.TopicMetadata] = {}
        self._write_drop_count = 0
        self._write_success_count = 0
        self._pre_episode_buffer = []
        self.latched_topic_names = set()

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.latched_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=100,
        )
        self.tf_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=100,
        )

        self.is_shutting_down = False
        self.episodes_recorded = 0
        self._seen_episodes = set()
        self._record_history: dict[int, dict[int, EpisodeRecord]] = {}
        self._written_records: set[tuple[int, int]] = set()
        self.recorded_topics: set[str] = set()
        self._recorded_rows: dict[str, RecordedTopic] = {}
        self._unresolved_types: set[str] = set()

        self._log_info("Subscribing to /clock for sim time")
        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, self.qos)

        self._log_info("Setting up topic subscriptions")
        self._setup_subscriptions()

        try:
            from arena_evaluation_msgs.srv import RecordEpisode
        except ImportError:
            RecordEpisode = None
        if RecordEpisode is not None:
            self.service_cb_group = MutuallyExclusiveCallbackGroup()
            self._start_service = self.create_service(
                RecordEpisode,
                "start_episode",
                self._start_episode_service_callback,
                callback_group=self.service_cb_group,
            )
            self._log_info("start_episode service ready")
        else:
            self._start_service = None
        self._log_info("Topic subscriptions ready. Waiting for first episode to open MCAP writer...")

    def _open_log_file(self):
        if self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.close()
            except OSError:
                pass
        if self.run_dir != self.episodes_root:
            self.log_file_path = self.run_dir / "recorder.log"
        else:
            self.log_file_path = self.episodes_root / "recorder.log"
        try:
            self.log_file = open(self.log_file_path, "a", buffering=1)
        except OSError as e:
            print(
                f"[DataRecorder] Failed to open log file at {self.log_file_path}: {e}",
                flush=True,
            )
            self.log_file = None

    @staticmethod
    def _parse_cli_dir() -> str:
        for idx, arg in enumerate(sys.argv):
            if arg in ("--dir", "-d") and idx + 1 < len(sys.argv):
                return sys.argv[idx + 1]
            if arg.startswith("--dir="):
                return arg[6:].strip()
        return ""

    @staticmethod
    def _generate_run_id(handles: dict[str, Any]) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bench_id_val = str(handles["benchmark_id"].value or "")
        if bench_id_val and bench_id_val != "unknown":
            return f"{ts}-{bench_id_val}" if not bench_id_val.startswith(ts) else bench_id_val
        map_val = str(handles["map"].value or handles["world"].value or "")
        robot_val = str(handles["robot"].value or "")
        planner_val = str(handles["local_planner"].value or handles["contestant"].value or "")
        details = [
            pathlib.Path(map_val).stem if map_val and map_val != "unknown" else "",
            robot_val if robot_val and robot_val != "unknown" else "",
            planner_val if planner_val and planner_val != "unknown" else "",
        ]
        parts = [item for item in details if item]
        return f"{ts}-{'-'.join(parts)}" if parts else f"{ts}-run"

    def _resolve_run_dirs(
        self,
        handles: dict[str, Any],
        record_data_dir: str,
        workspace_root: pathlib.Path,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        ws_data = (workspace_root / "data").resolve()
        ws_runs = (workspace_root / "data" / "runs").resolve()

        def _auto_dirs() -> tuple[pathlib.Path, pathlib.Path]:
            param_value = handles["data_recorder_autoprefix"].value
            if not param_value:
                run_id = self._generate_run_id(handles)
                self.set_parameters([Parameter("data_recorder_autoprefix", Parameter.Type.STRING, run_id)])
            else:
                run_id = str(param_value)
            run_dir = (workspace_root / "data" / "runs" / run_id).resolve()
            return run_dir, run_dir / "episodes"

        if record_data_dir.startswith("auto:/"):
            return _auto_dirs()
        p = pathlib.Path(record_data_dir)
        if not p.is_absolute():
            p = workspace_root / p
        p_resolved = p.resolve()
        if p_resolved in (ws_data, ws_runs) or (p.name in ("data", "runs") and p_resolved.parent in (workspace_root.resolve(), ws_data)):
            return _auto_dirs()
        if p.name == "episodes":
            return p_resolved.parent, p_resolved
        return p_resolved, p_resolved / "episodes"

    def _start_episode_recording(self, episode_id: int):
        """Close any current writer and open a fresh one for this episode."""

        global_episode_id = self._episode_id_offset
        self._episode_id_offset += 1
        self.current_sim_episode_id = episode_id

        self._close_current_writer()

        with self._tf_lock:
            self._tf_last_written = {}
        self._throttle_gates.clear()

        ep_name = f"episode_{global_episode_id:03d}"
        buffered = self._pre_clock_buffer
        if buffered:
            kept = [entry for entry in buffered if entry[0].strip("/") in self.latched_topic_names]
            self._log_info(f"Dropping {len(buffered) - len(kept)} pre-clock messages that arrived before {ep_name}")
            buffered[:] = kept

        self._enter_episode_dir(self.episodes_root / ep_name)
        self.current_episode_id = global_episode_id

        self._write_episode_metadata(global_episode_id)

        self._open_writer()

    def _setup_subscriptions(self):
        env_namespace = self.get_namespace().strip("/")
        env_prefix = f"/{env_namespace}" if env_namespace else ""

        self.subs = []

        from .topics import get_topics

        topics_dict = get_topics(namespace="", parent_namespace=env_namespace)

        for key, t_def in topics_dict.items():
            if key not in _ENV_TOPIC_KEYS or not self._records(key, t_def):
                continue
            msg_type = t_def.msg_type
            if (
                isinstance(msg_type, type)
                and msg_type.__name__
                in (
                    "Pedestrians",
                    "AgentFrame",
                    "AgentMeta",
                    "InteractionFrame",
                    "EpisodeRecord",
                    "RobotFleet",
                )
                and not msg_type.__module__.startswith(("arena_", "task_generator_"))
            ):
                continue

            topic_name = t_def.name_template
            if key != "tf_humans":
                self._register_topic(topic_name, msg_type)
            if t_def.qos_transient_local:
                self.latched_topic_names.add(topic_name.strip("/"))

            qos_profile, callback = self._env_topic_binding(key, t_def)
            raw = key not in _DESERIALIZED_TOPIC_KEYS
            sub = self.create_subscription(msg_type, topic_name, callback, qos_profile, raw=raw)  # pyright: ignore[reportArgumentType]
            self.subs.append(sub)
            if key == "episode_record":
                self.subs.append(self.create_subscription(msg_type, topic_name, self.episode_record_history_callback, _EPISODE_HISTORY_QOS))
                self.get_logger().info(f"Subscribed to EpisodeRecord on {topic_name}")
            elif key == "robots_fleet":
                self.get_logger().info(f"Subscribed to RobotFleet on {topic_name}")

        self._state_types = _state_types()
        recorded_topics_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        topic_name = f"{env_prefix}/state/recorded_topics"
        self.subs.append(
            self.create_subscription(
                RecordedTopics,
                topic_name,
                self.recorded_topics_callback,
                recorded_topics_qos,
            )
        )
        self.get_logger().info(f"Subscribed to RecordedTopics on {topic_name}")

        self.create_timer(1.0, self.discover_topics)

    def _env_topic_binding(self, key: str, t_def: TopicDefinition) -> tuple[QoSProfile, Callable[..., Any]]:
        qos_profile = self.latched_qos if t_def.qos_transient_local else self.qos
        if key == "episode_record":
            return QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            ), self.episode_record_callback
        if key == "robots_fleet":
            return qos_profile, self.robots_fleet_callback
        if key == "semantic_snapshot":
            return qos_profile, self.semantic_snapshot_callback
        if key == "initialpose":
            return qos_profile, self.initialpose_callback
        if key in ("tf", "tf_humans"):
            return self.tf_qos, self._tf_callback
        topic_name = t_def.name_template
        callback = self._create_throttled_callback(topic_name, rate_hz=t_def.throttle_rate_hz) if t_def.throttled else self._create_unthrottled_callback(topic_name)
        return qos_profile, callback

    def recorded_topics_callback(self, msg: RecordedTopics):
        for row in msg.topics:
            self._recorded_rows[row.key] = row
        self._subscribe_recorded_rows(None)
        for robot_ns in sorted(self.known_robots):
            self._subscribe_recorded_rows(robot_ns)

    def _subscribe_recorded_rows(self, robot_ns: str | None):
        """Subscribe the env rows (robot_ns None) or one robot's rows of the backend-declared recorded topics."""
        from .topics import recorded_topic_definition

        env_namespace = self.get_namespace().strip("/")
        for key, row in self._recorded_rows.items():
            if row.robot_scoped != (robot_ns is not None):
                continue
            t_def = recorded_topic_definition(row, env_namespace, robot_ns or "")
            if t_def is None:
                if row.msg_type not in self._unresolved_types:
                    self._unresolved_types.add(row.msg_type)
                    self._log_warn(f"Not recording {key}: message type {row.msg_type!r} does not import")
                continue
            topic_name = t_def.name_template
            if topic_name in self._topic_registry or not self._records(key, t_def):
                continue

            self._register_topic(topic_name, t_def.msg_type)

            default_qos = self.latched_qos if t_def.qos_transient_local else self.qos
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE if t_def.reliable else QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL if t_def.qos_transient_local else QoSDurabilityPolicy.VOLATILE,
                depth=t_def.depth or default_qos.depth,
            )
            if t_def.qos_transient_local:
                self.latched_topic_names.add(topic_name.strip("/"))

            if t_def.throttled:
                callback = self._create_throttled_callback(topic_name, rate_hz=t_def.throttle_rate_hz)
            else:
                callback = self._create_unthrottled_callback(topic_name)

            self.subs.append(self.create_subscription(t_def.msg_type, topic_name, callback, qos_profile, raw=True))
            self.get_logger().info(f"Subscribed to recorded topic: {topic_name}")

    def discover_topics(self):
        namespace = self.get_namespace().strip("/")
        ns_prefix = f"/{namespace}" if namespace else ""
        env_prefix = ns_prefix.rsplit("/", 1)[0] + "/"

        for name, types in self.get_topic_names_and_types():
            if name in self._topic_registry:
                continue
            if name.startswith(env_prefix):
                self._discover_state_topic(name, types)
            if not name.startswith(ns_prefix):
                continue

            name_lower = name.lower()
            is_odom = "odom" in name_lower

            if is_odom and "nav_msgs/msg/Odometry" in types:
                self._subscribe_discovered(name, Odometry, rate_hz=_ODOM_THROTTLE_RATE_HZ)
                if self.robot_model in ("unknown", ""):
                    for part in reversed(name.split("/")):
                        if part and part not in ("odom", "eval_sim", "arena_env_node", "arena") and "velocity_controller" not in part and not part.startswith("env_"):
                            self.robot_model = part
                            self._record_robot_model_metadata(part, replace=True)
                            break

    def _discover_state_topic(self, topic_name: str, types: list[str]) -> None:
        for type_name in types:
            entry = self._state_types.get(type_name)
            if entry is None:
                continue
            msg_type, basename, latched = entry
            if topic_name.rsplit("/", 1)[-1] != basename:
                continue
            self._register_topic(topic_name, msg_type)
            if latched:
                self.latched_topic_names.add(topic_name.strip("/"))
                sub = self.create_subscription(
                    msg_type,
                    topic_name,
                    self._create_unthrottled_callback(topic_name),
                    self.latched_qos,
                )
            else:
                sub = self.create_subscription(
                    msg_type,
                    topic_name,
                    self._create_throttled_callback(topic_name),
                    self.qos,
                )
            self.subs.append(sub)
            self._log_info(f"Discovered state topic: {topic_name} [{type_name}]{' latched' if latched else ''}")
            return

    def _subscribe_discovered(self, topic_name: str, msg_type: type, *, rate_hz: float) -> None:
        """Subscribe a discovered topic, throttled at rate_hz."""
        self._register_topic(topic_name, msg_type)
        sub = self.create_subscription(
            msg_type,
            topic_name,
            self._create_throttled_callback(topic_name, rate_hz=rate_hz),
            self.qos,
            raw=True,
        )
        self.subs.append(sub)
        self.get_logger().info(f"Dynamically subscribed to: {topic_name}")

    def clock_callback(self, msg: Clock):
        new_time = msg.clock.sec * int(1e9) + msg.clock.nanosec
        self._clock_received_count += 1
        if self._clock_received_count <= 5:
            self._log_info(f"/clock tick #{self._clock_received_count}: sim_time_ns={new_time} ({new_time / 1e9:.3f}s)")

        if self.current_time is not None and new_time < self.current_time:
            self._log_warn(f"Backward time jump detected: {self.current_time} -> {new_time}")
            self._throttle_gates.clear()
            with self._tf_lock:
                self._tf_last_written.clear()

        self.current_time = new_time

        if self._pre_clock_buffer is not None:
            if self._pre_clock_buffer:
                self._log_info(f"Flushing {len(self._pre_clock_buffer)} pre-clock buffered messages")
                for topic, buffered_msg in self._pre_clock_buffer:
                    if topic == "/tf":
                        self._tf_callback(buffered_msg)
                    else:
                        self._write_to_bag_at(topic, buffered_msg, self.current_time)
            self._pre_clock_buffer = None

    def _begin_episode(self, episode_id: int, source: str = "episode_record") -> bool:
        """Open a writer for a new episode unless one was already started for it."""
        if episode_id in self._seen_episodes:
            return False
        self._seen_episodes.add(episode_id)
        self.episodes_recorded += 1
        self._log_info(f"New episode detected ({source}): episode_id={episode_id}")
        self._start_episode_recording(episode_id)
        self._record_history = {key: records for key, records in self._record_history.items() if key >= episode_id}
        history = self._record_history.get(episode_id, {})
        for state in sorted(history):
            self._ingest_record(history[state], replay_before_ns=_running_ns(history))
        return True

    def _stop_episode(self, episode_id: int, outcome_state: int = 0, outcome_info: str = "") -> None:
        """Close the current episode writer and record its outcome."""
        if self.current_sim_episode_id != episode_id:
            self._log_info(f"stop_episode ignored for episode_id={episode_id} (current is {self.current_sim_episode_id})")
            return
        if self.current_metadata is not None and self.current_metadata_path is not None:
            try:
                self.current_metadata.outcome_state = outcome_state
                self.current_metadata.outcome_info = outcome_info
                MetadataWriter.write(self.current_metadata, self.current_metadata_path)
            except Exception as e:
                self._log_error(f"Failed to write outcome metadata: {e}")
        self._close_current_writer()
        self._pre_episode_buffer.clear()

    def _start_episode_service_callback(self, request: RecordEpisode.Request, response: RecordEpisode.Response) -> RecordEpisode.Response:
        """Authoritative episode lifecycle: START opens the writer, STOP closes it."""
        episode_id = int(request.episode_id)
        command = request.command
        if command == request.COMMAND_START:  # pyright: ignore[reportAttributeAccessIssue]
            self._log_info(f"start_episode service called: episode_id={episode_id}")
            self._begin_episode(episode_id, source="runner")
            response.success = True
            response.message = f"recording episode {self.current_episode_id}"
        elif command == request.COMMAND_STOP:  # pyright: ignore[reportAttributeAccessIssue]
            self._log_info(f"stop_episode service called: episode_id={episode_id} outcome={request.outcome_state}")
            self._stop_episode(
                episode_id,
                outcome_state=int(request.outcome_state),
                outcome_info=str(request.outcome_info),
            )
            response.success = True
            response.message = "stopped"
        else:
            response.success = False
            response.message = f"unknown command {command}"
        return response

    def episode_record_callback(self, msg: EpisodeRecord):
        outcome_state = msg.outcome_state
        outcome_label = _OUTCOME_LABELS.get(outcome_state, f"UNKNOWN({outcome_state})")
        self._log_info(f"EpisodeRecord received: episode_id={msg.episode_id} outcome={outcome_state} ({outcome_label}) info={msg.outcome_info!r}")

        self._record_history.setdefault(int(msg.episode_id), {})[int(outcome_state)] = msg
        if msg.episode_id not in self._seen_episodes and outcome_state == EpisodeRecord.RUNNING:
            self._begin_episode(msg.episode_id, source="episode_record")
        self._ingest_record(msg)

    def episode_record_history_callback(self, msg: EpisodeRecord):
        """Keep every record of the env's latched history, writing those of the episode being recorded."""
        self._record_history.setdefault(int(msg.episode_id), {})[int(msg.outcome_state)] = msg
        if int(msg.episode_id) == self.current_sim_episode_id and int(msg.episode_id) in self._seen_episodes:
            self._ingest_record(msg, replay_before_ns=_running_ns(self._record_history[int(msg.episode_id)]))

    def _ingest_record(self, msg: EpisodeRecord, replay_before_ns: int | None = None):
        """Write one record of the current episode once, at its RUNNING stamp when it arrives after the episode began."""
        key = (int(msg.episode_id), int(msg.outcome_state))
        if key in self._written_records or int(msg.episode_id) != self.current_sim_episode_id:
            return
        self._written_records.add(key)
        self._update_metadata_from_episode(msg)
        if msg.world:
            self._snapshot_world(msg.world)

        env_namespace = self.get_namespace().strip("/")
        topic = f"/{env_namespace}/state/episode" if env_namespace else "/state/episode"
        if self.current_time is None:
            if self._pre_clock_buffer is not None:
                self._pre_clock_buffer.append((topic, msg))
            return
        stamp = self.current_time
        if replay_before_ns is not None and int(msg.outcome_state) <= EpisodeRecord.RUNNING:
            stamp = min(stamp, replay_before_ns)
        self._write_to_bag_at(topic, msg, stamp)

    def semantic_snapshot_callback(self, msg: bytes):
        env_namespace = self.get_namespace().strip("/")
        now = self.current_time or self.get_clock().now().nanoseconds
        topic = f"/{env_namespace}/state/semantics" if env_namespace else "/state/semantics"
        self._write_to_bag_at(topic, msg, now)

    def initialpose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Record the env-level initial pose under the robot its header frame names, else env-scoped."""
        topic = initialpose_topic(msg.header.frame_id, self.env_ns_root)
        self._register_topic(topic, PoseWithCovarianceStamped)
        if self.current_time is None:
            if self._pre_clock_buffer is not None:
                self._pre_clock_buffer.append((topic, msg))
            return
        self._write_to_bag_at(topic, msg, self.current_time)

    def _records(self, key: str, t_def: TopicDefinition) -> bool:
        """Default topics always, the others only when listed under optional_topics in data_recorder_config.yaml."""
        return t_def.recorded or key in self.config.get("optional_topics", [])

    def robots_fleet_callback(self, msg: RobotFleet):
        env_namespace = self.get_namespace().strip("/")
        now = self.current_time or self.get_clock().now().nanoseconds
        topic = f"/{env_namespace}/state/robots" if env_namespace else "/state/robots"
        self._write_to_bag_at(topic, msg, now)

        for state in msg.robots:
            robot = state.descriptor
            robot_ns = robot.ns.strip("/")
            params = {item.key: item.value for item in state.params}
            if self._robot_params.get(robot_ns) != params:
                self._robot_params[robot_ns] = params
                snapshot = self._snapshot_robot_caps(robot_ns, robot.model, params)
                if snapshot is not None:
                    self._robot_snapshots[robot_ns] = snapshot
            if robot_ns not in self.known_robots:
                self._subscribe_new_robot(robot_ns, robot.model, env_namespace)

    def _subscribe_new_robot(self, robot_ns: str, robot_model: str, env_namespace: str) -> None:
        from .topics import get_topics

        self.get_logger().info(f"Discovered new robot from RobotFleet: {robot_ns} (Model: {robot_model})")
        self.known_robots.add(robot_ns)

        for key, t_def in get_topics(namespace=robot_ns, parent_namespace=env_namespace).items():
            if key in _ENV_TOPIC_KEYS or not self._records(key, t_def):
                continue
            msg_type = t_def.msg_type
            if isinstance(msg_type, type) and msg_type.__name__ == "CollisionEvents" and not msg_type.__module__.startswith(("arena_", "task_generator_")):
                continue
            topic_name = t_def.name_template
            self._register_topic(topic_name, msg_type)
            if t_def.qos_transient_local:
                self.latched_topic_names.add(topic_name.strip("/"))
            qos_profile, callback = self._env_topic_binding(key, t_def)
            self.subs.append(self.create_subscription(msg_type, topic_name, callback, qos_profile, raw=True))
            self.get_logger().info(f"Subscribed to robot topic: {topic_name}")

        self._subscribe_recorded_rows(robot_ns)
        ns_prefix = f"/{robot_ns}" if robot_ns else ""
        cmd_vel_topic, odom_topic = self._control_topics(self._robot_snapshots.get(robot_ns))
        for rel, msg_type, rate_hz in (
            (odom_topic, Odometry, _ODOM_THROTTLE_RATE_HZ),
            (cmd_vel_topic, TwistStamped, DEFAULT_THROTTLE_RATE_HZ),
        ):
            if rel and "/" in rel:
                topic = f"{ns_prefix}/{rel}"
                self._register_topic(topic, msg_type)
                self.subs.append(
                    self.create_subscription(
                        msg_type,
                        topic,
                        self._create_throttled_callback(topic, rate_hz=rate_hz),
                        self.qos,
                        raw=True,
                    )
                )

        if self.robot_model == "unknown":
            self.robot_model = robot_model
        self._record_robot_model_metadata(robot_model)

    def _record_robot_model_metadata(self, robot_model: str, *, replace: bool = False) -> None:
        if self.current_metadata is None or self.current_metadata_path is None:
            return
        if replace or "unknown" in self.current_metadata.robot_model:
            self.current_metadata.robot_model = []
        if robot_model not in self.current_metadata.robot_model:
            self.current_metadata.robot_model.append(robot_model)
        try:
            MetadataWriter.write(self.current_metadata, self.current_metadata_path)
        except Exception as e:
            self.get_logger().warn(f"Failed to write episode metadata: {e}")

    def _snapshot_robot_caps(self, robot_ns: str, model: str, params: dict[str, str]) -> dict[str, Any] | None:
        """Write the effective caps of a robot instance to its caps snapshot file."""
        from arena_robots.fleet import instance_caps

        runtime_hash = params.get("caps:hash")
        request = {key: value for key, value in params.items() if key != "caps:hash"}
        try:
            snapshot = instance_caps(model.partition("[")[0], request).snapshot()
        except Exception as e:
            self._log_warn(f"Robot {robot_ns}: caps of {model} do not resolve from {request}: {e}")
            return None
        if runtime_hash is not None and runtime_hash != snapshot["hash"]:
            self._log_warn(f"Robot {robot_ns}: caps hash {snapshot['hash']} differs from the runtime's {runtime_hash}")
            snapshot["runtime_hash"] = runtime_hash
        path = caps_snapshot_path(self.run_dir, robot_ns)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
        except Exception as e:
            self._log_error(f"Failed to write the caps snapshot of {robot_ns}: {e}")
        return snapshot

    @staticmethod
    def _control_topics(snapshot: dict[str, Any] | None) -> tuple[str, str]:
        """The (cmd_vel_topic, odom_topic) of a caps snapshot's body, relative to the robot namespace."""
        body = None if snapshot is None else snapshot["effective"].get("body")
        if body is None:
            return "", ""
        return str(body["cmd_vel_topic"]), str(body["odom_topic"])

    def _create_throttled_callback(self, topic_name: str, *, rate_hz: float = DEFAULT_THROTTLE_RATE_HZ) -> Callable[[object], None]:
        """A callback through the topic's rate gate, overridden by record_rates_hz by exact name."""
        interval_ns = min_interval_ns(resolve_rate_hz(self.record_rates_hz, topic_name, rate_hz))

        def callback(msg: object) -> None:
            if self.current_time is None:
                if self._pre_clock_buffer is not None:
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            if self._throttle_gate(topic_name, interval_ns).accept(now):
                self._write_to_bag_at(topic_name, msg, now)

        return callback

    def _throttle_gate(self, topic_name: str, interval_ns: int) -> RateGate:
        """The rate gate of a topic, created on its first message after the last reset."""
        gate = self._throttle_gates.get(topic_name)
        if gate is None:
            gate = RateGate(interval_ns)
            self._throttle_gates[topic_name] = gate
        return gate

    def _create_unthrottled_callback(self, topic_name: str) -> Callable[[object], None]:
        def callback(msg: object) -> None:
            if self.current_time is None:
                if self._pre_clock_buffer is not None:
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            self._write_to_bag_at(topic_name, msg, now)

        return callback

    def _tf_callback(self, msg: TFMessage) -> None:
        """Write each transform at its header stamp, one TFMessage per stamp, capped per frame pair."""
        if self.current_time is None:
            if self._pre_clock_buffer is not None:
                self._pre_clock_buffer.append(("/tf", msg))
            return
        by_stamp: dict[int, list[TransformStamped]] = {}
        with self._tf_lock:
            for transform in msg.transforms:
                stamp_ns = _stamp_ns(transform.header.stamp)
                if self._accept_transform(transform, stamp_ns):
                    by_stamp.setdefault(stamp_ns, []).append(transform)
        for stamp_ns in sorted(by_stamp):
            self._write_to_bag_at("/tf", TFMessage(transforms=by_stamp[stamp_ns]), stamp_ns)

    def _accept_transform(self, transform: TransformStamped, stamp_ns: int) -> bool:
        """Whether the transform passes its family interval, its stamp recorded when it does. Call under _tf_lock."""
        key = (transform.header.frame_id, transform.child_frame_id)
        last_ns = self._tf_last_written.get(key)
        if last_ns is not None and stamp_ns - last_ns < self._tf_interval_ns(transform.child_frame_id):
            return False
        self._tf_last_written[key] = stamp_ns
        return True

    def _tf_interval_ns(self, child_frame_id: str) -> int:
        """The tf_frames cap of a child frame, the /tf rate when no family pattern matches."""
        for pattern, ms in self.tf_frames:
            if pattern.search(child_frame_id):
                return math.ceil(float(ms) * _NS_PER_MILLISECOND)
        return min_interval_ns(resolve_rate_hz(self.record_rates_hz, "/tf", DEFAULT_THROTTLE_RATE_HZ))

    def _log_info(self, msg: str):
        full_msg = f"[DataRecorder] [INFO] {msg}"
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().info(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _log_warn(self, msg: str):
        full_msg = f"[DataRecorder] [WARN] {msg}"
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().warn(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _log_error(self, msg: str):
        full_msg = f"[DataRecorder] [ERROR] {msg}"
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.write(f"[{datetime.now().isoformat()}] {full_msg}\n")
        try:
            if rclpy.ok():
                self.get_logger().error(msg)
            else:
                print(full_msg, flush=True)
        except Exception:
            print(full_msg, flush=True)

    def _write_to_bag_at(self, topic_name: str, msg: object, timestamp_ns: int):
        if self.is_shutting_down:
            return

        if isinstance(msg, (bytes, bytearray)):
            serialized_msg = bytes(msg)
        else:
            try:
                serialized_msg = serialize_message(msg)
            except Exception as e:
                if not self.is_shutting_down:
                    self._log_error(f"Serialization failed for {topic_name}: {e}")
                return

        self._write_serialized(topic_name, serialized_msg, timestamp_ns)

    def _write_episode_metadata(self, episode_id: int):
        """Write initial episode_XXX.yaml for this episode."""
        assert self.current_metadata_path is not None
        metadata = IngestionMetadata.create_episode_metadata(
            benchmark_id=self.benchmark_id,
            planner=self.planner,
            stage=self.stage,
            map_name=self.map_name,
            episode_id=episode_id,
            robot_model=self.robot_model,
            env_ns_root=self.env_ns_root,
            is_reference=self.is_reference,
            reference_type=self.reference_type,
            suite_name=self.suite_name,
            contest_name=self.contest_name,
            episodes_requested=self.episodes_requested,
            local_planner=self.local_planner,
            inter_planner=self.inter_planner,
            task_generator_episode_id=self.current_sim_episode_id,
            agent_name=self.robot_model,
            workspace_dir=str(self.workspace_dir or self.episodes_root or "/opt/arena_ws"),
        )
        try:
            MetadataWriter.write(metadata, self.current_metadata_path)
            self.current_metadata = metadata
            self._log_info(f"Wrote episode metadata: {self.current_metadata_path}")
        except Exception as e:
            self._log_error(f"Failed to write initial episode metadata: {e}")
            self.current_metadata = metadata

    def _snapshot_world(self, world_name: str) -> None:
        """Write the world an episode runs in to run_dir/worlds/<name>, once per world and run, off the callback thread."""
        from arena_simulation_setup.tree.World import WorldIdentifier

        name, level_filter = WorldIdentifier.split_filter(world_name)
        if name in self._snapshot_worlds:
            return
        self._snapshot_worlds.add(name)
        threading.Thread(
            target=self._write_world_snapshot,
            args=(name, level_filter),
            name=f"world_snapshot_{name}",
            daemon=True,
        ).start()

    def _write_world_snapshot(self, name: str, level_filter: set[str] | None) -> None:
        from arena_simulation_setup.tree.World import WorldIdentifier, save_snapshot

        directory = self.run_dir / "worlds" / name
        if (directory / "world.yaml").exists():
            return
        try:
            view = WorldIdentifier(name).resolve_sync()
            world = view.load(validate=False, level_filter=level_filter, derive=self.kg_derive)
            save_snapshot(
                world,
                directory,
                self.world_lighting,
                level_origins=view.level_origins(),
            )
            self._log_info(f"Wrote world snapshot: {directory}")
        except Exception as e:
            self._log_error(f"Failed to write the world snapshot of {name}: {e}")

    def _update_metadata_from_episode(self, msg: EpisodeRecord):
        """Update the current episode's metadata with live EpisodeRecord data."""
        if self.current_metadata is None or self.current_metadata_path is None or int(msg.episode_id) != self.current_sim_episode_id:
            return
        try:
            if not self.current_metadata.robot_model or self.current_metadata.robot_model == ["unknown"]:
                self.current_metadata.robot_model = list(msg.robots)
            if not self.current_metadata.map or self.current_metadata.map == "unknown":
                self.current_metadata.map = msg.world

            self.current_metadata.task_layer = msg.task_layer
            self.current_metadata.modes = decode_modes(msg.modes)
            self.current_metadata.params = self._record_params(msg)

            self.current_metadata.task_generator_episode_id = int(msg.episode_id)
            if int(msg.outcome_state) in _TERMINAL_OUTCOMES:
                self.current_metadata.outcome_state = int(msg.outcome_state)
                self.current_metadata.outcome_info = str(msg.outcome_info)
            self.current_metadata.integrity = list(msg.integrity)

            MetadataWriter.write(self.current_metadata, self.current_metadata_path)
        except Exception as e:
            self._log_error(f"Failed to update episode metadata: {e}")

    def _record_params(self, msg: EpisodeRecord) -> dict[str, Any]:
        return {p.name: self._param_value_to_py(p.value) for p in msg.params}

    def _param_value_to_py(self, val: ParameterValue) -> bool | int | float | str | list[int] | list[bool] | list[float] | list[str]:
        from rcl_interfaces.msg import ParameterType

        mapping = {
            ParameterType.PARAMETER_BOOL: lambda v: v.bool_value,
            ParameterType.PARAMETER_INTEGER: lambda v: v.integer_value,
            ParameterType.PARAMETER_DOUBLE: lambda v: v.double_value,
            ParameterType.PARAMETER_STRING: lambda v: v.string_value,
            ParameterType.PARAMETER_BYTE_ARRAY: lambda v: list(v.byte_array_value),
            ParameterType.PARAMETER_BOOL_ARRAY: lambda v: list(v.bool_array_value),
            ParameterType.PARAMETER_INTEGER_ARRAY: lambda v: list(v.integer_array_value),
            ParameterType.PARAMETER_DOUBLE_ARRAY: lambda v: list(v.double_array_value),
            ParameterType.PARAMETER_STRING_ARRAY: lambda v: list(v.string_array_value),
        }
        return mapping.get(val.type, lambda v: str(v))(val)

    def read_config(self) -> _RecorderConfig:
        """The recorder config; a missing file keeps the defaults, a file still carrying record_frequencies is refused."""
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {"record_rates_hz": {}, "tf_frames": []}
        if "record_frequencies" in config:
            raise ValueError("data_recorder_config.yaml still sets record_frequencies, which the recorder no longer reads: replace it with record_rates_hz (exact topic name -> rate in Hz, 0 = unthrottled) and rebuild arena_evaluation so the installed share copy is refreshed")
        return {
            "record_rates_hz": _checked_rates(config.get("record_rates_hz") or {}),
            "tf_frames": list(config.get("tf_frames") or []),
        }

    def finalize(self):
        """Close the current episode writer and write final metadata."""
        if self.is_shutting_down:
            return
        self.is_shutting_down = True

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except Exception:
            pass

        print(
            f"[DataRecorder] finalize() called. clock_ticks={self._clock_received_count} writes_ok={self._write_success_count} writes_dropped={self._write_drop_count} episodes={self.episodes_recorded}",
            flush=True,
        )
        self._log_info("Finalizing recording - closing last episode writer...")

        self._close_current_writer()

        self._log_info(f"Recording finished. {self.episodes_recorded} episodes recorded to {self.episodes_root}")

    def destroy_node(self):
        self.finalize()
        super().destroy_node()


def _bind_to_parent():
    """SIGTERM the recorder when its parent dies instead of orphaning to init."""
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass

    if os.getppid() == 1:
        os._exit(0)


def main(args: list[str] | None = None):
    import os

    _bind_to_parent()

    rclpy.init(args=args)

    try:
        spin_node(DataRecorderNode(), executor=create_executor())
    except Exception as e:
        print(f"[DataRecorder] Exception in main: {e}", flush=True)
        import traceback

        traceback.print_exc()
        if rclpy.ok():
            rclpy.shutdown()
        os._exit(1)

    if rclpy.ok():
        rclpy.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
