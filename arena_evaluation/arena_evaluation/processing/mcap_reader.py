from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import re
from collections import defaultdict

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

from arena_evaluation.storage.schemas import TopicBundle

_log = logging.getLogger(__name__)
_BUNDLE_FIELDS = frozenset(f.name for f in dataclasses.fields(TopicBundle))


# Topic -> explicit PyArrow schema.  Only needed when RecordBatch.from_pydict
# cannot infer column types from data (e.g. semantic_snapshot where each row
# populates exactly one value_* column, leaving the rest None).
_TOPIC_SCHEMAS: dict[str, pa.Schema] = {
    "semantic_snapshot": pa.schema(
        [
            ("time_ns", pa.int64()),
            ("env_id", pa.int64()),
            ("world", pa.string()),
            ("entity", pa.string()),
            ("kind", pa.string()),
            ("field", pa.string()),
            ("field_kind", pa.string()),
            ("value_str", pa.string()),
            ("value_num", pa.float64()),
            ("value_bool", pa.bool_()),
            ("value_list", pa.list_(pa.string())),
        ]
    ),
    "collision_events": pa.schema(
        [
            ("time_ns", pa.int64()),
            ("collision_event", pa.int64()),
            ("collision_wall", pa.int64()),
            ("collision_static", pa.int64()),
            ("collision_pedestrian", pa.int64()),
            ("collision_obstacle_ids", pa.list_(pa.string())),
        ]
    ),
    "characterization_schedule": pa.schema(
        [
            ("phase_label", pa.string()),
            ("phase_kind", pa.string()),
            ("vx_target", pa.float64()),
            ("vy_target", pa.float64()),
            ("wz_target", pa.float64()),
            ("duration_s", pa.float64()),
            ("ramp_s", pa.float64()),
            ("turn_radius_m", pa.float64()),
        ]
    ),
}


class MCAPReader:
    """Reads an MCAP file and produces a TopicBundle of raw DataFrames."""

    def __init__(self, data_path: pathlib.Path):
        self.data_path = data_path

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _stamp_ns(header: object) -> int:
        """Header stamp (sim time) in ns."""
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    @staticmethod
    def _param_value_to_py(val: object) -> bool | int | float | str | list[int] | list[bool] | list[float] | list[str]:
        p_type = val.type
        if p_type == 1:
            return val.bool_value
        elif p_type == 2:
            return val.integer_value
        elif p_type == 3:
            return val.double_value
        elif p_type == 4:
            return val.string_value
        elif p_type == 5:
            return list(val.byte_array_value)
        elif p_type == 6:
            return list(val.bool_array_value)
        elif p_type == 7:
            return list(val.integer_array_value)
        elif p_type == 8:
            return list(val.double_array_value)
        elif p_type == 9:
            return list(val.string_array_value)
        return str(val)

    @staticmethod
    def _unflatten_dict(d: dict) -> dict:
        res: dict = {}
        for k, v in d.items():
            parts = k.split('.')
            curr = res
            for part in parts[:-1]:
                if part not in curr or not isinstance(curr[part], dict):
                    curr[part] = {}
                curr = curr[part]
            if isinstance(curr, dict):
                curr[parts[-1]] = v
        return res

    def read(self, out_dir: pathlib.Path, map_name_fallback: str | None = None) -> dict[str, TopicBundle]:
        """Stream the MCAP into per-topic parquet under out_dir and return lazy TopicBundles over it."""
        if self.data_path.is_dir():
            mcap_files = sorted(list(self.data_path.glob("*.mcap")))
            if not mcap_files:
                raise FileNotFoundError(f"No MCAP file found in directory: {self.data_path}")
        else:
            mcap_files = [self.data_path]

        for path_item in mcap_files:
            if not path_item.exists():
                raise FileNotFoundError(f"MCAP file not found: {path_item}")

        path = self.data_path.resolve()
        run_dir = path if path.is_dir() else path.parent

        def new_robot_data() -> dict[str, defaultdict[str, list]]:
            return {
                "odom": defaultdict(list),
                "odom_controller": defaultdict(list),
                "scan": defaultdict(list),
                "cmd_vel": defaultdict(list),
                "cmd_vel_controller": defaultdict(list),
                "joint_states": defaultdict(list),
                "collision_events": defaultdict(list),
                "collision_monitor_state": defaultdict(list),
                "power": defaultdict(list),
                "energy": defaultdict(list),
                "acoustics": defaultdict(list),
                "plan": defaultdict(list),
                "initialpose": defaultdict(list),
                "tf_gt": defaultdict(list),
                "characterization_phase": defaultdict(list),
                "characterization_schedule": defaultdict(list),
            }

        def new_env_data() -> dict[str, defaultdict[str, list]]:
            return {"peds": defaultdict(list), "peds_engine": defaultdict(list), "episode_record": defaultdict(list)}

        env_data = defaultdict(new_env_data)

        global_data = {
            "tf": defaultdict(list),
            "tf_static": defaultdict(list),
            "semantic_snapshot": defaultdict(list),
        }

        robot_data = defaultdict(new_robot_data)

        env_prefix = None

        out_dir.mkdir(parents=True, exist_ok=True)

        writers = {}
        accumulated_count = 0

        def flush_buffers():
            for topic_name, topic_data in global_data.items():
                if not topic_data or not any(len(column) for column in topic_data.values()):
                    continue

                schema_override = _TOPIC_SCHEMAS.get(topic_name)
                batch = pa.RecordBatch.from_pydict(dict(topic_data), schema=schema_override)
                writer_key = ("__global__", topic_name)

                if writer_key not in writers:
                    final_path = out_dir / f"{topic_name}.parquet"
                    writers[writer_key] = pq.ParquetWriter(final_path, schema=batch.schema, compression="zstd")
                writers[writer_key].write_batch(batch)
                topic_data.clear()

            for env_name, e_data in env_data.items():
                for topic_name, topic_data in e_data.items():
                    if not topic_data or not any(len(column) for column in topic_data.values()):
                        continue

                    batch = pa.RecordBatch.from_pydict(dict(topic_data))
                    writer_key = (env_name, topic_name)

                    if writer_key not in writers:
                        env_dir = out_dir / env_name
                        env_dir.mkdir(parents=True, exist_ok=True)
                        final_path = env_dir / f"{topic_name}.parquet"
                        writers[writer_key] = pq.ParquetWriter(final_path, schema=batch.schema, compression="zstd")
                    writers[writer_key].write_batch(batch)
                    topic_data.clear()

            for robot_name, r_data in robot_data.items():
                for topic_name, topic_data in r_data.items():
                    if not topic_data or not any(len(column) for column in topic_data.values()):
                        continue

                    batch = pa.RecordBatch.from_pydict(dict(topic_data), schema=_TOPIC_SCHEMAS.get(topic_name))
                    writer_key = (robot_name, topic_name)

                    if writer_key not in writers:
                        robot_dir = out_dir / robot_name
                        robot_dir.mkdir(parents=True, exist_ok=True)
                        final_path = robot_dir / f"{topic_name}.parquet"
                        writers[writer_key] = pq.ParquetWriter(final_path, schema=batch.schema, compression="zstd")
                    writers[writer_key].write_batch(batch)
                    topic_data.clear()

        try:
            for mfile in mcap_files:
                with open(mfile, "rb") as f:
                    reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])

                    msg_count = 0
                    decoders = {}
                    collision_kind_schema_cache: dict[int, bool] = {}

                    for schema, channel, message in reader.iter_messages(log_time_order=False):
                        try:
                            decoder = decoders.get(message.channel_id)
                            if decoder is None:
                                for factory in reader._decoder_factories:
                                    decoder = factory.decoder_for(channel.message_encoding, schema)
                                    if decoder is not None:
                                        decoders[message.channel_id] = decoder
                                        break
                            if decoder is None:
                                continue
                            ros_msg = decoder(message.data)
                            msg_count += 1
                        except Exception as e:
                            import logging

                            logging.getLogger(__name__).warning(f"Error reading MCAP message on {channel.topic} at index {msg_count}: {e}")
                            continue

                        topic = channel.topic
                        ts_ns = message.log_time
                        appended = False

                        parts = [p for p in topic.strip('/').split('/') if p]

                        env_key = "env_0"
                        match = re.search(r'(env_\d+)', topic)
                        if match:
                            env_key = match.group(1)

                        def get_robot_name(parts: list[str], env_key: str) -> str:
                            if len(parts) < 2:
                                return f"{env_key}_unknown"
                            base = parts[-2]
                            if base == env_key:
                                return env_key
                            if base.endswith("_controller"):
                                base = parts[-3] if len(parts) >= 3 else base
                            elif base == "power_publisher":
                                base = parts[-3] if len(parts) >= 3 else base
                            return f"{env_key}_{base}"

                        # Odom
                        if topic.endswith("/odom"):
                            robot_name = get_robot_name(parts, env_key)

                            target = robot_data[robot_name]["odom_controller" if parts[-2].endswith("_controller") else "odom"]
                            target["time_ns"].append(ts_ns)
                            target["stamp_ns"].append(self._stamp_ns(ros_msg.header))
                            target["pos_x"].append(ros_msg.pose.pose.position.x)
                            target["pos_y"].append(ros_msg.pose.pose.position.y)

                            yaw = self._quaternion_to_yaw(ros_msg.pose.pose.orientation.x, ros_msg.pose.pose.orientation.y, ros_msg.pose.pose.orientation.z, ros_msg.pose.pose.orientation.w)
                            target["yaw"].append(yaw)
                            target["vel_linear"].append(ros_msg.twist.twist.linear.x)
                            target["vel_angular"].append(ros_msg.twist.twist.angular.z)
                            appended = True

                        # Scan
                        elif topic.endswith("/scan") or topic.endswith("/lidar"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["scan"]
                            target["time_ns"].append(ts_ns)
                            target["scan_ranges"].append(list(ros_msg.ranges))
                            target["scan_min"].append(ros_msg.range_min)
                            target["scan_range_max"].append(ros_msg.range_max)
                            appended = True

                        # Cmd_vel
                        elif topic.endswith("/cmd_vel"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["cmd_vel_controller" if parts[-2].endswith("_controller") else "cmd_vel"]
                            twist = ros_msg.twist if schema.name == "geometry_msgs/msg/TwistStamped" else ros_msg
                            target["time_ns"].append(ts_ns)
                            target["linear_x"].append(twist.linear.x)
                            target["linear_y"].append(twist.linear.y)
                            target["linear_z"].append(twist.linear.z)
                            target["angular_x"].append(twist.angular.x)
                            target["angular_y"].append(twist.angular.y)
                            target["angular_z"].append(twist.angular.z)
                            appended = True
                        # Joint states
                        elif topic.endswith("/joint_states"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["joint_states"]
                            target["time_ns"].append(ts_ns)
                            target["name"].append(list(ros_msg.name))
                            target["position"].append(list(ros_msg.position))
                            target["velocity"].append(list(ros_msg.velocity))
                            target["effort"].append(list(ros_msg.effort))
                            appended = True

                        # Pedestrians
                        elif topic.endswith("/arena_peds") or topic.endswith("/peds") or topic.endswith("/agent_states"):
                            target = env_data[env_key]["peds_engine" if topic.endswith("/agent_states") else "peds"]
                            target["time_ns"].append(ts_ns)

                            if schema.name == "arena_people_msgs/msg/Pedestrians":
                                agents = ros_msg.pedestrians
                                is_pose2d = False
                            else:
                                agents = [a for a in ros_msg.agents if a.kind == 0]
                                is_pose2d = True

                            target["num_pedestrians"].append(len(agents))

                            positions = []
                            headings = []
                            twists = []

                            for p in agents:
                                if is_pose2d:
                                    positions.extend([p.pose.x, p.pose.y, 0.0])
                                    headings.append(p.pose.theta)
                                    twists.extend([p.velocity.x, p.velocity.y, p.velocity.z])
                                else:
                                    # Positions: flattened list [x1, y1, z1, x2, y2, z2, ...]
                                    positions.extend([p.pose.position.x, p.pose.position.y, p.pose.position.z])

                                    # Headings: calculate yaw from quaternion
                                    yaw = self._quaternion_to_yaw(p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w)
                                    headings.append(yaw)

                                    # Twists: flattened list of linear velocities [vx1, vy1, vz1, vx2, vy2, vz2, ...]
                                    twists.extend([p.twist.linear.x, p.twist.linear.y, p.twist.linear.z])

                            target["peds_positions"].append(positions)
                            target["peds_headings"].append(headings)
                            target["peds_twists"].append(twists)

                            appended = True

                        # Acoustics (ego-noise estimates from the M4 model)
                        elif topic.endswith("/acoustics"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["acoustics"]
                            target["time_ns"].append(ts_ns)
                            target["total_level_af_dba"].append(ros_msg.total_level_af_dba)
                            target["total_level_zf_db"].append(ros_msg.total_level_zf_db)
                            target["baseline_level_dba"].append(ros_msg.baseline_level_dba)
                            target["drivetrain_level_dba"].append(ros_msg.drivetrain_level_dba)
                            target["uncertainty_1sigma_dba"].append(ros_msg.uncertainty_1sigma_dba)
                            target["validity_flags"].append(ros_msg.validity_flags)
                            appended = True

                        # Characterization phase markers (open-loop sweeps)
                        elif topic.endswith("/characterization_phase"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["characterization_phase"]
                            target["time_ns"].append(ts_ns)
                            target["label"].append(str(ros_msg.data))
                            appended = True

                        # Characterization schedule: the phase table the sweep actually ran, latest wins
                        elif topic.endswith("/characterization_schedule"):
                            rows = self._schedule_rows(str(ros_msg.data))
                            if rows is not None:
                                robot_name = get_robot_name(parts, env_key)
                                target = robot_data[robot_name]["characterization_schedule"]
                                target.clear()
                                for column, values in rows.items():
                                    target[column].extend(values)
                                appended = True

                        # Episode records
                        elif topic.endswith("/state/episode"):
                            target = env_data[env_key]["episode_record"]
                            target["time_ns"].append(ts_ns)
                            target["episode_id"].append(ros_msg.episode_id)
                            target["outcome_state"].append(ros_msg.outcome_state)
                            target["outcome_info"].append(ros_msg.outcome_info)
                            target["goal_uuid"].append(ros_msg.goal_uuid)

                            robots_yaml = ""
                            try:
                                import yaml

                                robots_dict = {}
                                for p in ros_msg.robots_params:
                                    robots_dict[p.name] = self._param_value_to_py(p.value)
                                robots_dict = self._unflatten_dict(robots_dict)
                                robots_yaml = yaml.dump(robots_dict)
                            except Exception:
                                pass
                            target["robots_params"].append(robots_yaml)
                            appended = True

                        # Semantic snapshot (latched, one row per annotated entity)
                        elif topic.endswith("/state/semantics"):
                            target = global_data["semantic_snapshot"]
                            for ent in ros_msg.entities:
                                self._append_semantic_entity(target, ts_ns, ros_msg.env_id, ros_msg.world, ent)
                            appended = True

                        elif topic.endswith("/collision_events"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["collision_events"]
                            target["time_ns"].append(ts_ns)
                            target["collision_event"].append(len(ros_msg.events))

                            has_kind = collision_kind_schema_cache.get(schema.id)
                            if has_kind is None:
                                has_kind = re.search(r"^\s*string\s+kind\b", schema.data.decode(), re.M) is not None
                                collision_kind_schema_cache[schema.id] = has_kind

                            wall_count = None
                            static_count = None
                            pedestrian_count = None
                            if has_kind:
                                wall_count = 0
                                static_count = 0
                                pedestrian_count = 0
                                for ev in ros_msg.events:
                                    if ev.kind == "wall":
                                        wall_count += 1
                                    elif ev.kind == "static":
                                        static_count += 1
                                    elif ev.kind == "pedestrian":
                                        pedestrian_count += 1
                            target["collision_wall"].append(wall_count)
                            target["collision_static"].append(static_count)
                            target["collision_pedestrian"].append(pedestrian_count)
                            target["collision_obstacle_ids"].append(sorted({ev.obstacle_id for ev in ros_msg.events}))
                            appended = True

                        # Collision Monitor State (Nav2)
                        elif topic.endswith("/collision_monitor_state"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["collision_monitor_state"]
                            target["time_ns"].append(ts_ns)
                            target["action_type"].append(ros_msg.action_type)
                            target["polygon_name"].append(ros_msg.polygon_name)
                            appended = True

                        # Power
                        elif topic.endswith("/power_publisher/power"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["power"]
                            target["time_ns"].append(ts_ns)
                            target["total_power_w"].append(ros_msg.total_power_w)
                            target["static_power_w"].append(ros_msg.static_power_w)
                            target["total_mechanical_power_w"].append(ros_msg.total_mechanical_power_w)
                            target["total_thermal_power_w"].append(ros_msg.total_thermal_power_w)
                            target["joint_names"].append(list(ros_msg.joint_names))
                            target["joint_mechanical_power_w"].append(list(ros_msg.joint_mechanical_power_w))
                            target["joint_thermal_power_w"].append(list(ros_msg.joint_thermal_power_w))
                            target["joint_total_power_w"].append(list(ros_msg.joint_total_power_w))
                            appended = True

                        # Energy
                        elif topic.endswith("/power_publisher/energy"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["energy"]
                            target["time_ns"].append(ts_ns)
                            target["total_energy_consumed_wh"].append(ros_msg.total_energy_consumed_wh)
                            target["battery_soc_percent"].append(ros_msg.battery_soc_percent)
                            appended = True

                        # Global plan
                        elif topic.endswith("/plan") or topic.endswith("/global_plan"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["plan"]
                            target["time_ns"].append(ts_ns)
                            poses_x = []
                            poses_y = []
                            poses_yaw = []
                            for pose_stamped in ros_msg.poses:
                                poses_x.append(pose_stamped.pose.position.x)
                                poses_y.append(pose_stamped.pose.position.y)
                                poses_yaw.append(self._quaternion_to_yaw(pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w))
                            target["poses_x"].append(poses_x)
                            target["poses_y"].append(poses_y)
                            target["poses_yaw"].append(poses_yaw)
                            appended = True

                        # Initialpose
                        elif topic.endswith("/initialpose"):
                            robot_name = get_robot_name(parts, env_key)
                            target = robot_data[robot_name]["initialpose"]
                            target["time_ns"].append(ts_ns)
                            target["pos_x"].append(ros_msg.pose.pose.position.x)
                            target["pos_y"].append(ros_msg.pose.pose.position.y)
                            target["yaw"].append(self._quaternion_to_yaw(ros_msg.pose.pose.orientation.x, ros_msg.pose.pose.orientation.y, ros_msg.pose.pose.orientation.z, ros_msg.pose.pose.orientation.w))
                            appended = True

                        # TF processing
                        elif topic in ("/tf", "tf", "/tf_static", "tf_static"):
                            target_dict = global_data["tf_static"] if "static" in topic else global_data["tf"]
                            for t in ros_msg.transforms:
                                target_dict["time_ns"].append(ts_ns)
                                target_dict["frame_id"].append(t.header.frame_id)
                                target_dict["child_frame_id"].append(t.child_frame_id)
                                target_dict["trans_x"].append(t.transform.translation.x)
                                target_dict["trans_y"].append(t.transform.translation.y)
                                target_dict["trans_z"].append(t.transform.translation.z)
                                target_dict["rot_x"].append(t.transform.rotation.x)
                                target_dict["rot_y"].append(t.transform.rotation.y)
                                target_dict["rot_z"].append(t.transform.rotation.z)
                                target_dict["rot_w"].append(t.transform.rotation.w)
                                appended = True

                                # Detect ground-truth map/world/odom -> base_link transform if available
                                parent = t.header.frame_id.strip('/')
                                child = t.child_frame_id.strip('/')
                                parent_lower = parent.lower()
                                child_lower = child.lower()

                                is_world_frame = parent_lower in ("map", "world", "odom") or parent_lower.endswith("/map") or parent_lower.endswith("/world") or parent_lower.endswith("/odom")
                                is_base_frame = child_lower.endswith("base_link") or child_lower.endswith("base_footprint") or "base_link" in child_lower or "base_footprint" in child_lower

                                # Filter by env_prefix if detected
                                if env_prefix and env_prefix not in child_lower:
                                    continue

                                if is_world_frame and is_base_frame:
                                    child_parts = child.split('/')
                                    # /tf carries every env, so the env comes from the frame, not the topic.
                                    robot_name = get_robot_name(child_parts, self._frame_env(child, env_key))

                                    yaw_val = self._quaternion_to_yaw(t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
                                    target = robot_data[robot_name]["tf_gt"]
                                    target["time_ns"].append(ts_ns)
                                    target["stamp_ns_gt"].append(self._stamp_ns(t.header))
                                    target["pos_x_gt"].append(t.transform.translation.x)
                                    target["pos_y_gt"].append(t.transform.translation.y)
                                    target["yaw_gt"].append(yaw_val)
                                    target["frame_id"].append(parent)
                                    appended = True

                        if appended:
                            accumulated_count += 1
                            if accumulated_count >= 10000:
                                flush_buffers()
                                accumulated_count = 0
        finally:
            flush_buffers()
            for writer in writers.values():
                writer.close()
            for env_dir in (d for d in out_dir.iterdir() if d.is_dir()):
                engine_peds = env_dir / "peds_engine.parquet"
                arena_peds = env_dir / "peds.parquet"
                if not engine_peds.exists():
                    continue
                if arena_peds.exists():
                    engine_peds.unlink()
                else:
                    engine_peds.replace(arena_peds)

        return self.load_bundles(out_dir, run_dir, map_name_fallback=map_name_fallback)

    @staticmethod
    def _frame_env(frame: str, default: str) -> str:
        """The env_N segment of a frame id, else default."""
        match = re.search(r"(env_\d+)", frame)
        return match.group(1) if match else default

    @staticmethod
    def _schedule_rows(payload: str) -> dict[str, list] | None:
        """Columns of a recorded characterization schedule, None when the payload is not one."""
        rows: dict[str, list] = defaultdict(list)
        try:
            for phase in json.loads(payload)["phases"]:
                rows["phase_label"].append(str(phase["name"]))
                rows["phase_kind"].append(str(phase["kind"]))
                rows["vx_target"].append(float(phase["vx_target"]))
                rows["vy_target"].append(float(phase["vy_target"]))
                rows["wz_target"].append(float(phase["wz_target"]))
                rows["duration_s"].append(float(phase["duration_s"]))
                rows["ramp_s"].append(float(phase["ramp_s"]))
                rows["turn_radius_m"].append(float(phase["radius_m"]))
        except (ValueError, KeyError, TypeError) as e:
            _log.warning(f"characterization_schedule payload rejected: {e!r}")
            return None
        return dict(rows)

    @staticmethod
    def _append_semantic_field(
        target: dict,
        ts_ns: int,
        env_id: int,
        world: str,
        entity: str,
        kind: str,
        field: str,
        field_kind: str,
        value_str: str | None = None,
        value_num: float | None = None,
        value_bool: bool | None = None,
        value_list: list | None = None,
    ) -> None:
        """Append one long-format (entity, field) row of the flattened semantic snapshot."""
        target["time_ns"].append(ts_ns)
        target["env_id"].append(env_id)
        target["world"].append(world)
        target["entity"].append(entity)
        target["kind"].append(kind)
        target["field"].append(field)
        target["field_kind"].append(field_kind)
        target["value_str"].append(value_str)
        target["value_num"].append(value_num)
        target["value_bool"].append(value_bool)
        target["value_list"].append(value_list)

    @staticmethod
    def _append_semantic_entity(target: dict, ts_ns: int, env_id: int, world: str, ent: object) -> None:
        """Flatten one SemanticEntityState into long-format rows."""
        for name, value in zip(ent.discrete_names, ent.discrete_values, strict=True):
            MCAPReader._append_semantic_field(target, ts_ns, env_id, world, ent.entity, ent.kind, name, "discrete", value_str=str(value))
        for name, value in zip(ent.continuous_names, ent.continuous_values, strict=True):
            MCAPReader._append_semantic_field(target, ts_ns, env_id, world, ent.entity, ent.kind, name, "continuous", value_num=float(value))
        for name, value in zip(ent.predicate_names, ent.predicate_values, strict=True):
            MCAPReader._append_semantic_field(target, ts_ns, env_id, world, ent.entity, ent.kind, name, "predicate", value_bool=bool(value))
        MCAPReader._append_semantic_field(target, ts_ns, env_id, world, ent.entity, ent.kind, "members", "members", value_list=list(ent.members))

    @staticmethod
    def load_bundles(topics_dir: pathlib.Path, run_dir: pathlib.Path, map_name_fallback: str | None = None) -> dict[str, TopicBundle]:
        # Reconstruct dict[str, TopicBundle]
        bundles = {}

        def load_parquet(path: pathlib.Path) -> pl.LazyFrame | None:
            if path.exists():
                lf = pl.scan_parquet(path)
                if "time_ns" in lf.collect_schema().names():
                    lf = lf.sort("time_ns")
                return lf
            return None

        # Load global data
        global_bundle = TopicBundle()
        for t_name in ("tf", "tf_static", "semantic_snapshot"):
            lf = load_parquet(topics_dir / f"{t_name}.parquet")
            if lf is not None:
                setattr(global_bundle, t_name, lf)

        # Build each robot's bundle
        robot_dirs = [d for d in topics_dir.iterdir() if d.is_dir()]

        # Calculate env offsets
        env_offsets = {}
        if global_bundle.tf_static is not None:
            try:
                tf_df = global_bundle.tf_static.collect()
                for row in tf_df.iter_rows(named=True):
                    parent = row["frame_id"].strip('/').lower()
                    child = row["child_frame_id"].strip('/').lower()
                    parent_is_world = parent in ("map", "world", "odom", "") or parent.endswith("/map") or parent.endswith("/world") or parent.endswith("/odom")
                    is_robot_base = child.endswith("base_link") or child.endswith("base_footprint") or "base_link" in child or "base_footprint" in child
                    if parent_is_world and not is_robot_base:
                        match = re.match(r'^(env_\d+)(?:/map)?$', child)
                        if match and match.group(1) not in env_offsets:
                            env_offsets[match.group(1)] = (row["trans_x"], row["trans_y"])
            except Exception:
                pass

        if global_bundle.tf is not None:
            try:
                tf_df = global_bundle.tf.collect()
                for row in tf_df.iter_rows(named=True):
                    parent = row["frame_id"].strip('/').lower()
                    child = row["child_frame_id"].strip('/').lower()
                    parent_is_world = parent in ("map", "world", "odom", "") or parent.endswith("/map") or parent.endswith("/world") or parent.endswith("/odom")
                    is_robot_base = child.endswith("base_link") or child.endswith("base_footprint") or "base_link" in child or "base_footprint" in child
                    if parent_is_world and not is_robot_base:
                        match = re.match(r'^(env_\d+)/map$', child)
                        if match and match.group(1) not in env_offsets:
                            env_offsets[match.group(1)] = (row["trans_x"], row["trans_y"])
            except Exception:
                pass

        # Fallback for env_0 if tf_static recording missed it
        if "env_0" not in env_offsets:
            env_offsets["env_0"] = (5.0, 5.0)

        # If no explicit robot dirs but we have data, maybe it was named "unknown"
        for robot_dir in robot_dirs:
            robot_name = robot_dir.name

            # Skip pure environment directories (e.g. env_0) if odom.parquet is absent and other robot directories exist
            has_odom = (robot_dir / "odom.parquet").exists() or (robot_dir / "odom_controller.parquet").exists()
            if not has_odom and any((d / "odom.parquet").exists() or (d / "odom_controller.parquet").exists() for d in robot_dirs if d != robot_dir):
                continue

            rb = TopicBundle()

            match = re.search(r'(env_\d+)', robot_name)
            env_key = match.group(1) if match else "env_0"
            ox, oy = env_offsets.get(env_key, (0.0, 0.0))

            # Copy global references
            rb.tf = global_bundle.tf
            rb.tf_static = global_bundle.tf_static
            rb.semantic_snapshot = global_bundle.semantic_snapshot

            # Load env references
            env_dir = topics_dir / env_key
            rb.peds = load_parquet(env_dir / "peds.parquet")
            rb.episode_record = load_parquet(env_dir / "episode_record.parquet")

            mx, my = 0.0, 0.0
            map_name = None
            if rb.episode_record is not None:
                try:
                    ep_df = rb.episode_record.collect() if isinstance(rb.episode_record, pl.LazyFrame) else rb.episode_record
                    if "map" in ep_df.columns and len(ep_df) > 0:
                        map_name = str(ep_df["map"][0])
                except Exception:
                    pass

            if not map_name and map_name_fallback:
                map_name = map_name_fallback

            if map_name:
                try:
                    from arena_evaluation.processing.map_registry import MapRegistry

                    map_meta = MapRegistry.get_map_metadata(map_name, run_dir=run_dir)
                    if map_meta and "origin" in map_meta and map_meta["origin"]:
                        mx, my = float(map_meta["origin"][0]), float(map_meta["origin"][1])
                except Exception:
                    pass

            total_ox = ox + mx
            total_oy = oy + my

            if rb.episode_record is not None and (total_ox != 0.0 or total_oy != 0.0):
                try:
                    import json

                    ep_df = rb.episode_record.collect() if isinstance(rb.episode_record, pl.LazyFrame) else rb.episode_record

                    if len(ep_df) > 0:
                        new_starts = []
                        new_goals = []
                        for row in ep_df.iter_rows(named=True):
                            s = json.loads(row.get("start_pos", "[]"))
                            g = json.loads(row.get("goal_pos", "[]"))
                            if len(s) >= 2:
                                s[0] -= total_ox
                                s[1] -= total_oy
                            if len(g) >= 2:
                                g[0] -= total_ox
                                g[1] -= total_oy
                            new_starts.append(json.dumps(s))
                            new_goals.append(json.dumps(g))

                        ep_df = ep_df.with_columns([pl.Series("start_pos", new_starts), pl.Series("goal_pos", new_goals)])
                        rb.episode_record = ep_df.lazy() if isinstance(rb.episode_record, pl.LazyFrame) else ep_df
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to offset episode_record coordinates: {e}")
                    pass

            if rb.peds is not None and (total_ox != 0.0 or total_oy != 0.0):
                try:
                    rb.peds = rb.peds.with_columns([pl.col("peds_positions").list.eval(pl.when(pl.int_range(0, pl.element().len()) % 3 == 0).then(pl.element() - total_ox).when(pl.int_range(0, pl.element().len()) % 3 == 1).then(pl.element() - total_oy).otherwise(pl.element()))])
                except Exception:
                    pass

            for parquet in sorted(robot_dir.glob("*.parquet")):
                t_name = parquet.stem
                if t_name not in _BUNDLE_FIELDS:
                    continue
                lf = load_parquet(parquet)
                if lf is None:
                    continue
                if total_ox != 0.0 or total_oy != 0.0:
                    if t_name == "initialpose":
                        try:
                            lf = lf.with_columns([(pl.col("pos_x") - total_ox).alias("pos_x"), (pl.col("pos_y") - total_oy).alias("pos_y")])
                        except Exception:
                            pass
                    elif t_name == "tf_gt":
                        try:
                            lf = lf.with_columns([(pl.col("pos_x_gt") - total_ox).alias("pos_x_gt"), (pl.col("pos_y_gt") - total_oy).alias("pos_y_gt")])
                        except Exception:
                            pass
                    elif t_name == "plan":
                        try:
                            lf = lf.with_columns([pl.col("poses_x").list.eval(pl.element() - total_ox), pl.col("poses_y").list.eval(pl.element() - total_oy)])
                        except Exception:
                            pass
                setattr(rb, t_name, lf)

            # TEMP: until fixed in isaac, the bare odom topic is shared with Isaac's zero-twist publisher
            controller_odom = load_parquet(robot_dir / "odom_controller.parquet")
            if controller_odom is not None:
                rb.odom = controller_odom

            if rb.cmd_vel is None:
                rb.cmd_vel = load_parquet(robot_dir / "cmd_vel_controller.parquet")

            bundles[robot_name] = rb

        return bundles
