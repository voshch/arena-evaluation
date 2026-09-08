import argparse
from typing import Any
import os
import re
import shutil
import threading
import traceback
from datetime import datetime, timezone
import yaml
import pathlib
import sys
import signal

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import serialize_message

import rosbag2_py
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan, JointState
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from tf2_msgs.msg import TFMessage

try:
    from arena_people_msgs.msg import Pedestrians
    HAS_PEDESTRIANS = True
except ImportError:
    class Pedestrians:
        pass
    HAS_PEDESTRIANS = False

try:
    from arena_robots_msgs.msg import CollisionEvents
    HAS_COLLISION = True
except ImportError:
    class CollisionEvents:
        pass
    HAS_COLLISION = False

try:
    from arena_robots_msgs.msg import Power, Energy
    HAS_POWER = True
except ImportError:
    class Power:
        pass
    class Energy:
        pass
    HAS_POWER = False

try:
    from task_generator_msgs.msg import EpisodeRecord, RobotFleet, SemanticSnapshot
    HAS_TASK_GEN = True
except ImportError:
    class EpisodeRecord:
        pass
    class RobotFleet:
        pass
    HAS_TASK_GEN = False

_OUTCOME_LABELS = {0: "QUEUED", 1: "RUNNING", 2: "SUCCESS", 3: "FAILED", 4: "SKIPPED", 5: "FATAL"}
_TERMINAL_OUTCOMES = {
    EpisodeRecord.SUCCESS,
    EpisodeRecord.FAILED,
    EpisodeRecord.SKIPPED,
    EpisodeRecord.FATAL,
}

import arena_evaluation_msgs.srv as arena_evaluation_srvs

from .metadata import IngestionMetadata
from ..storage.manifest import MetadataWriter


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('arena_evaluation_data_recorder', automatically_declare_parameters_from_overrides=True)

        self.base_dir = get_package_share_directory("arena_evaluation")

        for name, default in [
            ("record_data_dir", ""),
            ("benchmark_id", "unknown"),
            ("contestant", "unknown"),
            ("stage", "unknown"),
            ("map", "unknown"),
            ("world", "unknown"),
            ("suite_name", ""),
            ("contest_name", ""),
            ("local_planner", ""),
            ("inter_planner", ""),
            ("robot", ""),
            ("episodes_requested", 0),
            ("is_reference", False),
            ("reference_type", ""),
            ("episode_id_offset", 0),
            ("workspace_dir", ""),
        ]:
            if not self.has_parameter(name):
                try:
                    self.declare_parameter(name, default)
                except Exception:
                    pass

        record_data_dir = str(self.get_parameter("record_data_dir").value or "")

        if not record_data_dir:
            for idx, arg in enumerate(sys.argv):
                if arg in ("--dir", "-d") and idx + 1 < len(sys.argv):
                    record_data_dir = sys.argv[idx + 1]
                    break
                elif arg.startswith("--dir="):
                    record_data_dir = arg[6:].strip()
                    break

        if not record_data_dir:
            record_data_dir = "auto:/"

        if record_data_dir.startswith("auto:/"):
            if not self.has_parameter("data_recorder_autoprefix"):
                try:
                    self.declare_parameter("data_recorder_autoprefix", "")
                except Exception:
                    pass
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            param_value = self.get_parameter("data_recorder_autoprefix").value
            if not param_value:
                self.set_parameters([Parameter("data_recorder_autoprefix", Parameter.Type.STRING, timestamp)])
            else:
                timestamp = param_value
            workspace_root = pathlib.Path(self.base_dir).parents[3]
            record_data_dir_path = workspace_root / "data" / timestamp / "episodes"
        else:
            record_data_dir_path = pathlib.Path(record_data_dir)
            if not record_data_dir_path.is_absolute():
                workspace_root = pathlib.Path(self.base_dir).parents[3]
                record_data_dir_path = workspace_root / record_data_dir_path

        self.episodes_root = record_data_dir_path.resolve()
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        try:
            self.episodes_root.chmod(0o777)
        except Exception:
            pass

        self.benchmark_id = str(self.get_parameter("benchmark_id").value or "unknown")
        self.planner = str(self.get_parameter("contestant").value or "unknown")
        self.stage = str(self.get_parameter("stage").value or "unknown")
        self.map_name = str(self.get_parameter("map").value or "unknown")
        self.world = str(self.get_parameter("world").value or "unknown")
        self.suite_name = str(self.get_parameter("suite_name").value or "")
        self.contest_name = str(self.get_parameter("contest_name").value or "")
        self.local_planner = str(self.get_parameter("local_planner").value or "")
        self.inter_planner = str(self.get_parameter("inter_planner").value or "")
        self.episodes_requested = int(self.get_parameter("episodes_requested").value or 0)
        self.is_reference = bool(self.get_parameter("is_reference").value)
        ref_type = self.get_parameter("reference_type").value
        self.reference_type = ref_type if ref_type else None
        self._episode_id_offset = int(self.get_parameter("episode_id_offset").value or 0)
        self.workspace_dir = str(self.get_parameter("workspace_dir").value or "")

        env_namespace = self.get_namespace().strip('/')
        self.env_ns_root = f"/{env_namespace}" if env_namespace else ""

        self.robot_model = str(self.get_parameter("robot").value or "unknown")
        self.current_sim_episode_id: int | None = None
        self.known_robots = set()

        self.current_episode_id: int | None = None
        self.current_episode_dir: pathlib.Path | None = None
        self.current_metadata_path: pathlib.Path | None = None
        self.current_metadata = None

        self.log_file = None
        self.log_file_path: pathlib.Path | None = None
        self._open_log_file()

        self.config = self.read_config()
        self.freqs = self.config.get("record_frequencies", {"default": 20.0})
        self.tf_frames = [(re.compile(pattern), float(ms)) for pattern, ms in self.config.get("tf_frames", [])]

        self._log_info(f"Recorder ready. Episodes root: {self.episodes_root}")
        self._log_info(f"planner={self.planner!r} stage={self.stage!r} map={self.map_name!r} benchmark_id={self.benchmark_id!r}")

        self.current_time = None
        self._pre_clock_buffer = []
        self._clock_received_count = 0
        self.last_recorded_times: dict[str, int] = {}
        self.writer_lock = threading.Lock()
        # guards the tf merge state against the start_episode service callback group
        self._tf_lock = threading.Lock()
        self._tf_pending: dict[tuple[str, str], Any] = {}
        self._tf_last_written: dict[tuple[str, str], int] = {}
        self._tf_last_flush = 0
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
        self.reliable_volatile_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.tf_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=100,
        )
        self.audio_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1000,
        )

        self.is_shutting_down = False
        self.episodes_recorded = 0
        self._seen_episodes = set()
        self.recorded_topics: set[str] = set()

        self._log_info(f"Subscribing to /clock for sim time")
        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_callback, self.qos)

        self._log_info(f"Setting up topic subscriptions")
        self._setup_subscriptions()

        try:
            from arena_evaluation_msgs.srv import RecordEpisode
        except ImportError:
            RecordEpisode = None
        if RecordEpisode is not None:
            self.service_cb_group = MutuallyExclusiveCallbackGroup()
            self._start_service = self.create_service(
                RecordEpisode, "start_episode", self._start_episode_service_callback,
                callback_group=self.service_cb_group
            )
            self._log_info("start_episode service ready")
        else:
            self._start_service = None
        self._log_info(f"Topic subscriptions ready. Waiting for first episode to open MCAP writer...")

    def _open_log_file(self):
        if self.log_file is not None and not self.log_file.closed:
            try:
                self.log_file.close()
            except Exception:
                pass
        self.log_file_path = self.episodes_root / "recorder.log"
        try:
            self.log_file = open(self.log_file_path, "a", buffering=1)
        except Exception as e:
            print(f"[DataRecorder] Failed to open log file at {self.log_file_path}: {e}", flush=True)
            self.log_file = None

    def _start_episode_recording(self, episode_id: int):
        """Close any current writer and open a fresh one for this episode."""

        global_episode_id = self._episode_id_offset
        self._episode_id_offset += 1
        self.current_sim_episode_id = episode_id

        self._close_current_writer()

        self._write_success_count = 0
        self._write_drop_count = 0
        self.recorded_topics = set()

        with self._tf_lock:
            self._tf_pending = {}
            self._tf_last_written = {}
            self._tf_last_flush = 0

        ep_name = f"episode_{global_episode_id:03d}"
        ep_dir = self.episodes_root / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)
        try:
            ep_dir.chmod(0o777)
        except Exception:
            pass

        self.current_episode_id = global_episode_id
        self.current_episode_dir = ep_dir
        self.current_metadata_path = ep_dir / f"{ep_name}.yaml"

        self._write_episode_metadata(global_episode_id)

        mcap_uri = str(ep_dir / ep_name)
        self._log_info(f"Opening MCAP writer for {ep_name}: uri={mcap_uri}")

        mcap_config_path = os.path.join(self.base_dir, "config", "mcap_writer_options.yaml")
        if not os.path.exists(mcap_config_path):
            self._log_warn(f"mcap_writer_options.yaml not found - recording without compression")
            mcap_config_path = ""

        storage_options = rosbag2_py.StorageOptions(
            uri=mcap_uri,
            storage_id='mcap',
            max_bagfile_size=0,
            max_cache_size=0,
            storage_config_uri=mcap_config_path,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )

        try:
            with self.writer_lock:
                self.writer = rosbag2_py.SequentialWriter()
                self.writer.open(storage_options, converter_options)
                self.topics_metadata = {}
                self.last_recorded_times = {}

                if self._pre_episode_buffer:
                    self._log_info(f"Flushing {len(self._pre_episode_buffer)} pre-episode buffered messages to writer")
                    for topic_name, serialized_msg, timestamp_ns in self._pre_episode_buffer:
                        self._ensure_topic_in_bag(topic_name)
                        self.writer.write(topic_name.strip('/'), serialized_msg, timestamp_ns)
                        self.recorded_topics.add(topic_name.strip('/'))
                    self._pre_episode_buffer.clear()
            self._log_info(f"MCAP writer opened for {ep_name}")
        except Exception as e:
            self._log_error(f"Failed to open MCAP writer for {ep_name}: {e}")
            self.writer = None

    def _close_current_writer(self):
        """Flush, close the current writer, and flatten the rosbag2 subdirectory."""

        with self.writer_lock:
            if self.writer is None:
                return
            try:
                self.writer.close()
            except Exception as e:
                self._log_error(f"Error closing MCAP writer: {e}")
            finally:
                self.writer = None

        if self.current_episode_dir is not None:
            ep_name = self.current_episode_dir.name
            inner_dir = self.current_episode_dir / ep_name
            if inner_dir.is_dir():
                for mcap_file in sorted(inner_dir.glob("*.mcap")):
                    dest = self.current_episode_dir / f"{ep_name}.mcap"
                    if dest.exists():
                        dest = self.current_episode_dir / f"{ep_name}_{mcap_file.stem}.mcap"
                    try:
                        shutil.move(str(mcap_file), str(dest))
                        self._log_info(f"Flattened {mcap_file.name} -> {dest.name}")
                    except Exception as e:
                        self._log_error(f"Failed to flatten {mcap_file}: {e}")

                rosbag_meta = inner_dir / "metadata.yaml"
                if rosbag_meta.exists():
                    try:
                        import yaml
                        with open(rosbag_meta, 'r') as f:
                            bag_info = yaml.safe_load(f).get("rosbag2_bagfile_information", {})

                        if self.current_metadata is not None:
                            self.current_metadata.rosbag2_message_count = bag_info.get("message_count")
                            self.current_metadata.rosbag2_topics = bag_info.get("topics_with_message_count")
                            self._log_info(f"Merged rosbag2 metadata into episode metadata.")

                        rosbag_meta.unlink()
                    except Exception as e:
                        self._log_error(f"Failed to merge rosbag metadata: {e}")

                try:
                    inner_dir.rmdir()
                except Exception as e:
                    self._log_warn(f"Failed to remove inner dir {inner_dir.name}: {e}")

        self._finalize_episode_metadata()

    def _setup_subscriptions(self):
        env_namespace = self.get_namespace().strip('/')
        env_prefix = f"/{env_namespace}" if env_namespace else ""

        self.subs = []

        from .topics import get_topics
        topics_dict = get_topics(namespace="", parent_namespace=env_namespace)

        for key, t_def in topics_dict.items():
            if key not in ("episode_record", "robots_fleet", "peds", "agent_states", "semantic_snapshot", "map", "door_mask", "tf", "tf_static"):
                continue

            topic_name = t_def.name_template
            msg_type = t_def.msg_type

            if isinstance(msg_type, type) and msg_type.__name__ in ("Pedestrians", "AgentStates", "EpisodeRecord", "RobotFleet") and not msg_type.__module__.startswith("arena_") and not msg_type.__module__.startswith("task_generator_"):
                continue

            self._register_topic(topic_name, msg_type)

            qos_profile = self.latched_qos if t_def.qos_transient_local else self.qos
            if t_def.qos_transient_local:
                self.latched_topic_names.add(topic_name.strip('/'))

            if key == "episode_record":
                qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    depth=1,
                )
                callback = self.episode_record_callback
            elif key == "robots_fleet":
                callback = self.robots_fleet_callback
            elif key == "semantic_snapshot":
                callback = self.semantic_snapshot_callback
            elif key == "tf":
                qos_profile = self.tf_qos
                callback = self._tf_callback
            elif t_def.throttled:
                callback = self._create_throttled_callback(topic_name)
            else:
                callback = self._create_unthrottled_callback(topic_name)

            sub = self.create_subscription(msg_type, topic_name, callback, qos_profile)
            self.subs.append(sub)
            if key == "episode_record":
                self.get_logger().info(f"Subscribed to EpisodeRecord on {topic_name}")
            elif key == "robots_fleet":
                self.get_logger().info(f"Subscribed to RobotFleet on {topic_name}")

        self.create_timer(1.0, self.discover_topics)

    def discover_topics(self):
        namespace = self.get_namespace().strip('/')
        ns_prefix = f"/{namespace}" if namespace else ""

        for name, types in self.get_topic_names_and_types():
            if name in self._topic_registry:
                continue
            if not name.startswith(ns_prefix):
                continue

            name_lower = name.lower()
            is_scan = "scan" in name_lower or "lidar" in name_lower
            is_odom = "odom" in name_lower

            if is_odom and "nav_msgs/msg/Odometry" in types:
                self._subscribe_discovered(name, Odometry)
                if self.robot_model in ("unknown", ""):
                    for part in reversed(name.split("/")):
                        if part and part not in ("odom", "eval_sim", "task_generator_node", "arena") \
                                and "velocity_controller" not in part \
                                and not part.startswith("env_"):
                            self.robot_model = part
                            if self.current_metadata is not None:
                                self.current_metadata.robot_model = [part]
                                MetadataWriter.write(self.current_metadata, self.current_metadata_path)
                            break

    def _subscribe_discovered(self, topic_name: str, msg_type):
        self._register_topic(topic_name, msg_type)
        sub = self.create_subscription(msg_type, topic_name, self._create_throttled_callback(topic_name), self.qos)
        self.subs.append(sub)
        self.get_logger().info(f"Dynamically subscribed to: {topic_name}")


    def clock_callback(self, msg: Clock):
        new_time = msg.clock.sec * int(1e9) + msg.clock.nanosec
        self._clock_received_count += 1
        if self._clock_received_count <= 5:
            self._log_info(f"/clock tick #{self._clock_received_count}: sim_time_ns={new_time} ({new_time/1e9:.3f}s)")

        if self.current_time is not None and new_time < self.current_time:
            self._log_warn(f"Backward time jump detected: {self.current_time} -> {new_time}")
            self._flush_tf(self.current_time, force=True)
            self.last_recorded_times.clear()
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

        self._flush_tf(self.current_time)

    def _begin_episode(self, episode_id: int, source: str = "episode_record") -> bool:
        """Open a writer for a new episode unless one was already started for it."""
        if episode_id in self._seen_episodes:
            return False
        self._seen_episodes.add(episode_id)
        self.episodes_recorded += 1
        self._log_info(f"New episode detected ({source}): episode_id={episode_id}")
        self._start_episode_recording(episode_id)
        return True

    def _stop_episode(self, episode_id: int, outcome_state: int = 0, outcome_info: str = "") -> None:
        """Close the current episode writer and record its outcome."""
        if self.current_sim_episode_id != episode_id:
            self._log_info(f"stop_episode ignored for episode_id={episode_id} (current is {self.current_sim_episode_id})")
            return
        if self.current_metadata is not None:
            try:
                self.current_metadata.outcome_state = outcome_state
                self.current_metadata.outcome_info = outcome_info
                MetadataWriter.write(self.current_metadata, self.current_metadata_path)
            except Exception as e:
                self._log_error(f"Failed to write outcome metadata: {e}")
        if self.current_time is None:
            with self._tf_lock:
                self._tf_pending.clear()
        else:
            self._flush_tf(self.current_time, force=True)
        self._close_current_writer()
        self._pre_episode_buffer.clear()

    def _start_episode_service_callback(self, request, response):
        """Authoritative episode lifecycle: START opens the writer, STOP closes it."""
        episode_id = int(request.episode_id)
        command = request.command
        if command == request.COMMAND_START:
            self._log_info(f"start_episode service called: episode_id={episode_id}")
            self._begin_episode(episode_id, source="runner")
            response.success = True
            response.message = f"recording episode {self.current_episode_id}"
        elif command == request.COMMAND_STOP:
            self._log_info(f"stop_episode service called: episode_id={episode_id} outcome={request.outcome_state}")
            self._stop_episode(episode_id, outcome_state=int(request.outcome_state), outcome_info=str(request.outcome_info))
            response.success = True
            response.message = "stopped"
        else:
            response.success = False
            response.message = f"unknown command {command}"
        return response

    def episode_record_callback(self, msg: EpisodeRecord):
        outcome_state = msg.outcome_state
        outcome_label = _OUTCOME_LABELS.get(outcome_state, f"UNKNOWN({outcome_state})")
        self._log_info(
            f"EpisodeRecord received: episode_id={msg.episode_id} "
            f"outcome={outcome_state} ({outcome_label}) info={msg.outcome_info!r}"
        )

        if msg.episode_id not in self._seen_episodes and outcome_state in (0, 1):
            self._begin_episode(msg.episode_id, source="episode_record")

        self._update_metadata_from_episode(msg)

        env_namespace = self.get_namespace().strip('/')
        topic = f"/{env_namespace}/state/episode" if env_namespace else "/state/episode"
        if self.current_time is None:
            self._pre_clock_buffer.append((topic, msg))
            return
        self._write_to_bag_at(topic, msg, self.current_time)

    def semantic_snapshot_callback(self, msg: SemanticSnapshot):
        env_namespace = self.get_namespace().strip('/')
        now = self.current_time or self.get_clock().now().nanoseconds
        topic = f"/{env_namespace}/state/semantics" if env_namespace else "/state/semantics"
        self._write_to_bag_at(topic, msg, now)

    def robots_fleet_callback(self, msg):
        env_namespace = self.get_namespace().strip('/')
        now = self.current_time or self.get_clock().now().nanoseconds
        topic = f"/{env_namespace}/state/robots" if env_namespace else "/state/robots"
        self._write_to_bag_at(topic, msg, now)

        from .topics import get_topics

        for state in msg.robots:
            robot = state.descriptor
            robot_ns = robot.ns.strip('/')

            if robot_ns not in self.known_robots:
                self.get_logger().info(f"Discovered new robot from RobotFleet: {robot_ns} (Model: {robot.model})")
                self.known_robots.add(robot_ns)

                topics_dict = get_topics(namespace=robot_ns, parent_namespace=env_namespace)

                for key, t_def in topics_dict.items():
                    if key in ("episode_record", "robots_fleet", "peds", "agent_states", "semantic_snapshot", "map", "door_mask", "tf", "tf_static"):
                        continue

                    topic_name = t_def.name_template
                    msg_type = t_def.msg_type

                    if isinstance(msg_type, type) and msg_type.__name__ in ("CollisionEvents") and not msg_type.__module__.startswith("arena_") and not msg_type.__module__.startswith("task_generator_"):
                        continue

                    self._register_topic(topic_name, msg_type)

                    qos_profile = self.latched_qos if t_def.qos_transient_local else self.qos
                    if key in ("audio_raw", "audio_rendered", "audio_stem_motor", "audio_render_inputs"):
                        qos_profile = self.audio_qos
                    if t_def.qos_transient_local:
                        self.latched_topic_names.add(topic_name.strip('/'))

                    if t_def.throttled:
                        callback = self._create_throttled_callback(topic_name)
                    else:
                        callback = self._create_unthrottled_callback(topic_name)

                    sub = self.create_subscription(msg_type, topic_name, callback, qos_profile)
                    self.subs.append(sub)
                    self.get_logger().info(f"Subscribed to robot topic: {topic_name}")

                ns_prefix = f"/{robot_ns}" if robot_ns else ""
                cmd_vel_topic, odom_topic = self._control_topics(robot.model)
                for rel, msg_type in ((odom_topic, Odometry), (cmd_vel_topic, TwistStamped)):
                    if rel and "/" in rel:
                        topic = f"{ns_prefix}/{rel}"
                        self._register_topic(topic, msg_type)
                        self.subs.append(self.create_subscription(msg_type, topic, self._create_throttled_callback(topic), self.qos))

                if self.robot_model == "unknown":
                    self.robot_model = robot.model

                if self.current_metadata is not None:
                    if "unknown" in self.current_metadata.robot_model:
                        self.current_metadata.robot_model = []

                    if robot.model not in self.current_metadata.robot_model:
                        self.current_metadata.robot_model.append(robot.model)

                    try:
                        from ..storage.manifest import MetadataWriter
                        MetadataWriter.write(self.current_metadata, self.current_metadata_path)
                    except Exception as e:
                        self.get_logger().warn(f"Failed to write episode metadata: {e}")

    @staticmethod
    def _control_topics(model: str) -> tuple[str, str]:
        """The model's controller-side (cmd_vel_topic, odom_topic), relative to the robot namespace."""
        try:
            path = os.path.join(get_package_share_directory("arena_robots"), "robots", model.partition("[")[0], "model_params.yaml")
            with open(path) as f:
                control = (yaml.safe_load(f) or {}).get("control", {})
        except Exception:
            return "", ""
        return str(control.get("cmd_vel_topic", "")), str(control.get("odom_topic", ""))

    def _resolve_throttle_ms(self, topic_name: str) -> float:
        topic_lower = topic_name.lower()
        for key, ms in self.freqs.items():
            if key == "default":
                continue
            if key.lower() in topic_lower:
                return float(ms)
        return float(self.freqs.get("default", 20.0))

    def _create_throttled_callback(self, topic_name: str):
        throttle_ms = self._resolve_throttle_ms(topic_name)

        def callback(msg):
            if self.current_time is None:
                if self._pre_clock_buffer is not None:
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            last_time = self.last_recorded_times.get(topic_name, 0)
            if (now - last_time) / 1e6 >= throttle_ms:
                self._write_to_bag_at(topic_name, msg, now)
                self.last_recorded_times[topic_name] = now
        return callback

    def _create_unthrottled_callback(self, topic_name: str):
        def callback(msg):
            if self.current_time is None:
                if self._pre_clock_buffer is not None:
                    self._pre_clock_buffer.append((topic_name, msg))
                return
            now = self.current_time
            self._write_to_bag_at(topic_name, msg, now)
        return callback

    def _tf_callback(self, msg: TFMessage):
        """Merge the per-window transforms of /tf, newest wins per (frame, child) pair."""
        if self.current_time is None:
            if self._pre_clock_buffer is not None:
                self._pre_clock_buffer.append(("/tf", msg))
            return
        now = self.current_time
        with self._tf_lock:
            for transform in msg.transforms:
                child = transform.child_frame_id
                interval_ms = 0.0
                for pattern, ms in self.tf_frames:
                    if pattern.search(child):
                        interval_ms = ms
                        break
                key = (transform.header.frame_id, child)
                if (now - self._tf_last_written.get(key, 0)) / 1e6 >= interval_ms:
                    self._tf_pending[key] = transform

    def _flush_tf(self, now: int, force: bool = False):
        """Write the merged transforms as one TFMessage. Never called under writer_lock."""
        window_ms = self._resolve_throttle_ms("/tf")
        with self._tf_lock:
            if not self._tf_pending:
                return
            if not force and (now - self._tf_last_flush) / 1e6 < window_ms:
                return
            taken = self._tf_pending
            self._tf_pending = {}
            for key in taken:
                self._tf_last_written[key] = now
            self._tf_last_flush = now
        self._write_to_bag_at("/tf", TFMessage(transforms=list(taken.values())), now)

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

    def _write_to_bag_at(self, topic_name: str, msg, timestamp_ns: int):
        if self.is_shutting_down:
            return

        try:
            serialized_msg = serialize_message(msg)
        except Exception as e:
            if not self.is_shutting_down:
                self._log_error(f"Serialization failed for {topic_name}: {e}")
            return

        with self.writer_lock:
            if self.writer is None:
                topic_stripped = topic_name.strip('/')
                if topic_stripped in self.latched_topic_names:
                    if len(self._pre_episode_buffer) < 10000:
                        self._pre_episode_buffer.append((topic_name, serialized_msg, timestamp_ns))
                        return

                self._write_drop_count += 1
                if self._write_drop_count <= 10 or self._write_drop_count % 100 == 0:
                    self._log_warn(f"DROP (writer=None) topic={topic_name} drop_count={self._write_drop_count}")
                return
            try:
                self._ensure_topic_in_bag(topic_name)
                self.writer.write(topic_name.strip('/'), serialized_msg, timestamp_ns)
                self.recorded_topics.add(topic_name.strip('/'))
                self._write_success_count += 1
                if self._write_success_count <= 5 or self._write_success_count % 500 == 0:
                    self._log_info(f"Write success count reached {self._write_success_count}")
            except Exception as e:
                self._log_error(f"WRITE ERROR topic={topic_name} ts={timestamp_ns} err={e}")


    def _register_topic(self, topic_name: str, msg_type):
        """Pre-register a topic so we know its type string before the first message."""
        if topic_name in self._topic_registry:
            return
        try:
            parts = msg_type.__module__.split('.')
            type_str = f"{parts[0]}/msg/{msg_type.__name__}"
        except AttributeError:
            type_str = str(msg_type)

        self._topic_registry[topic_name] = rosbag2_py.TopicMetadata(
            id=0,
            name=topic_name.strip('/'),
            type=type_str,
            serialization_format='cdr',
        )

    def _ensure_topic_in_bag(self, topic_name: str):
        """Lazily call create_topic on first write. Must be called under writer_lock."""
        strip = topic_name.strip('/')
        if strip not in self.topics_metadata and topic_name in self._topic_registry:
            self.writer.create_topic(self._topic_registry[topic_name])
            self.topics_metadata[strip] = self._topic_registry[topic_name]


    def _write_episode_metadata(self, episode_id: int):
        """Write initial episode_XXX.yaml for this episode."""
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

    def _finalize_episode_metadata(self):
        """Update episode yaml with final recording stats after writer is closed."""
        if self.current_metadata is None or self.current_metadata_path is None:
            return
        try:
            self.current_metadata.recording_ended_at = datetime.now(timezone.utc).isoformat()
            self.current_metadata.pedsim_available = any(
                "arena_peds" in t or "agent_states" in t for t in self.recorded_topics
            )
            self.current_metadata.recorded_topics = sorted(self.recorded_topics)
            MetadataWriter.write(self.current_metadata, self.current_metadata_path)
            self._log_info(f"Finalized episode metadata: {self.current_metadata_path}")
        except Exception as e:
            self._log_error(f"Failed to finalize episode metadata: {e}")

    def _update_metadata_from_episode(self, msg: EpisodeRecord):
        """Update the current episode's metadata with live EpisodeRecord data."""
        if self.current_metadata is None:
            return
        try:
            if not self.current_metadata.robot_model or self.current_metadata.robot_model == ["unknown"]:
                self.current_metadata.robot_model = list(msg.robots)
            if not self.current_metadata.map or self.current_metadata.map == "unknown":
                self.current_metadata.map = msg.world

            self.current_metadata.tm_obstacles = msg.tm_obstacles
            self.current_metadata.tm_robots = msg.tm_robots
            self.current_metadata.tm_modules = list(msg.tm_modules)

            self.current_metadata.task_generator_episode_id = int(msg.episode_id)
            if int(msg.outcome_state) in _TERMINAL_OUTCOMES:
                self.current_metadata.outcome_state = int(msg.outcome_state)
                self.current_metadata.outcome_info = str(msg.outcome_info)

            obstacles_params = {p.name: self._param_value_to_py(p.value) for p in msg.obstacles_params}
            robots_params = {p.name: self._param_value_to_py(p.value) for p in msg.robots_params}
            self.current_metadata.obstacles_params = self._unflatten_dict(obstacles_params)
            self.current_metadata.robots_params = self._unflatten_dict(robots_params)

            MetadataWriter.write(self.current_metadata, self.current_metadata_path)
        except Exception as e:
            self._log_error(f"Failed to update episode metadata: {e}")

    def _param_value_to_py(self, val):
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

    def _unflatten_dict(self, d: dict) -> dict:
        res: dict = {}
        for k, v in d.items():
            parts = k.split('.')
            curr = res
            for part in parts[:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = v
        return res



    def read_config(self):
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception:
            return {"record_frequencies": {"default": 20.0}, "tf_frames": []}

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
            f"[DataRecorder] finalize() called. "
            f"clock_ticks={self._clock_received_count} "
            f"writes_ok={self._write_success_count} "
            f"writes_dropped={self._write_drop_count} "
            f"episodes={self.episodes_recorded}",
            flush=True,
        )
        self._log_info("Finalizing recording - closing last episode writer...")

        self._close_current_writer()

        self._log_info(
            f"Recording finished. {self.episodes_recorded} episodes recorded to {self.episodes_root}"
        )

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


def main(args=None):
    import os

    _bind_to_parent()

    def _on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)

    rclpy.init(args=args)

    node = None
    try:
        node = DataRecorderNode()

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            print("[DataRecorder] Finalizing node...", flush=True)
            if node:
                node.finalize()
            print("[DataRecorder] Shutting down executor...", flush=True)
            executor.shutdown()
            print("[DataRecorder] Destroying node...", flush=True)
            if node:
                node.destroy_node()
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


if __name__ == '__main__':
    main()