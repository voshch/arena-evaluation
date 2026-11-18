"""Episode lifecycle, world semantics, plans, characterization and pose markers, and transforms as flat rows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import yaml

from arena_evaluation.processing.telemetry.messages import Envelope, RobotRef, env_row, robot_row, stamp_ns, yaw_from_orientation
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_runtime_msgs.msg import EpisodeRecord, SemanticSnapshot
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Path
    from rcl_interfaces.msg import ParameterValue
    from std_msgs.msg import String as RosString
    from tf2_msgs.msg import TFMessage

_log = logging.getLogger(__name__)
_NOT_SET, _BOOL, _INTEGER, _DOUBLE, _STRING = 0, 1, 2, 3, 4
_BYTE_ARRAY, _BOOL_ARRAY, _INTEGER_ARRAY, _DOUBLE_ARRAY, _STRING_ARRAY = 5, 6, 7, 8, 9


def record_episode(sink: TableSink, envelope: Envelope[EpisodeRecord]) -> None:
    record = envelope.body
    sink.append(
        "episode_lifecycle",
        env_row(
            envelope,
            episode_id=int(record.episode_id),
            outcome_state=int(record.outcome_state),
            outcome_info=str(record.outcome_info),
            goal_uuid=str(record.goal_uuid),
            task_layer=str(record.task_layer),
            modes_json=json.dumps({m.key: m.value for m in record.modes}),
            robots_json=json.dumps(list(record.robots)),
            goal_dist_start_m=float(record.goal_dist_start),
            goal_dist_min_m=float(record.goal_dist_min),
            path_length_m=float(record.path_length),
            start_t_ns=int(record.start_time.sec) * 1_000_000_000 + int(record.start_time.nanosec),
            phases_yaml=str(record.phases) or None,
            params_yaml=_params_yaml(record),
            conditions_yaml=str(record.conditions) or None,
            world=str(record.world) if record.world else None,
            seed=int(record.seed) if record.seed >= 0 else None,
            human_params=_human_params_json(record),
        ),
    )
    for flag_idx, flag in enumerate(record.integrity):
        sink.append("episode_integrity", env_row(envelope, flag_idx=flag_idx, flag=str(flag)))


def record_semantics(sink: TableSink, envelope: Envelope[SemanticSnapshot]) -> None:
    snapshot = envelope.body
    env_id = f"env_{int(snapshot.env_id)}"
    for entity in snapshot.entities:
        base = {"t_ns": envelope.t_ns, "env_id": env_id, "world": str(snapshot.world), "entity_name": str(entity.entity), "kind": str(entity.kind)}
        for name, value in zip(entity.discrete_names, entity.discrete_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "discrete", "value_str": str(value), "value_num": None, "value_bool": None})
        for name, value in zip(entity.continuous_names, entity.continuous_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "continuous", "value_str": None, "value_num": float(value), "value_bool": None})
        for name, value in zip(entity.predicate_names, entity.predicate_values, strict=True):
            sink.append("world_semantics", {**base, "field_name": str(name), "field_kind": "predicate", "value_str": None, "value_num": None, "value_bool": bool(value)})
        members = ",".join(sorted(str(member) for member in entity.members))
        sink.append("world_semantics", {**base, "field_name": "members", "field_kind": "members", "value_str": members, "value_num": None, "value_bool": None})


class PlanRecorder:
    """Writes each global plan as numbered waypoint rows, one plan per message and robot."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._plan_counts: defaultdict[RobotRef, int] = defaultdict(int)

    def record(self, envelope: Envelope[Path]) -> None:
        plan_id = self._plan_counts[envelope.ref]
        self._plan_counts[envelope.ref] += 1
        for waypoint_idx, stamped in enumerate(envelope.body.poses):
            pose = stamped.pose
            self._sink.append(
                "robot_plans",
                robot_row(
                    envelope,
                    plan_id=plan_id,
                    waypoint_idx=waypoint_idx,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=yaw_from_orientation(pose.orientation),
                ),
            )


def record_characterization_phase(sink: TableSink, envelope: Envelope[RosString]) -> None:
    sink.append("robot_characterization_phase", robot_row(envelope, label=str(envelope.body.data)))


class ScheduleRecorder:
    """Keeps the characterization schedule the sweep actually ran: the latest valid payload per robot wins."""

    def __init__(self) -> None:
        self._latest: dict[RobotRef, tuple[int, list[dict[str, Any]]]] = {}

    def record(self, envelope: Envelope[RosString]) -> None:
        phases = _schedule_phases(str(envelope.body.data))
        if phases is not None:
            self._latest[envelope.ref] = (envelope.t_ns, phases)

    def write(self, sink: TableSink) -> None:
        for ref, (t_ns, phases) in self._latest.items():
            for phase_idx, phase in enumerate(phases):
                sink.append(
                    "robot_characterization_schedule",
                    {"t_ns": t_ns, "env_id": ref.env_id, "robot_id": ref.robot_id, "phase_idx": phase_idx, **phase},
                )


def record_initial_pose(sink: TableSink, envelope: Envelope[PoseWithCovarianceStamped]) -> None:
    pose = envelope.body.pose.pose
    sink.append(
        "robot_initialpose",
        robot_row(envelope, pos_x=float(pose.position.x), pos_y=float(pose.position.y), yaw=yaw_from_orientation(pose.orientation)),
    )


def record_task_pose(sink: TableSink, envelope: Envelope[PoseStamped]) -> None:
    message = envelope.body
    sink.append(
        "robot_task_pose",
        robot_row(
            envelope,
            stamp_ns=stamp_ns(message.header),
            pos_x=float(message.pose.position.x),
            pos_y=float(message.pose.position.y),
            yaw=yaw_from_orientation(message.pose.orientation),
        ),
    )


def record_transforms(sink: TableSink, table: str, envelope: Envelope[TFMessage]) -> None:
    for transform in envelope.body.transforms:
        translation, rotation = transform.transform.translation, transform.transform.rotation
        sink.append(
            table,
            {
                "t_ns": envelope.t_ns,
                "parent_frame": str(transform.header.frame_id),
                "child_frame": str(transform.child_frame_id),
                "tx": float(translation.x),
                "ty": float(translation.y),
                "tz": float(translation.z),
                "qx": float(rotation.x),
                "qy": float(rotation.y),
                "qz": float(rotation.z),
                "qw": float(rotation.w),
            },
        )


def _params_yaml(record: EpisodeRecord) -> str:
    return yaml.dump({param.name: _param_value(param.value) for param in record.params})


def _human_params_json(record: EpisodeRecord) -> str:
    return json.dumps({param.name: _param_value(param.value) for param in record.human_params})


def _param_value(value: ParameterValue) -> object:
    kind = value.type
    if kind == _NOT_SET:
        return None
    if kind == _BOOL:
        return value.bool_value
    if kind == _INTEGER:
        return value.integer_value
    if kind == _DOUBLE:
        return value.double_value
    if kind == _STRING:
        return value.string_value
    if kind == _BYTE_ARRAY:
        return [byte[0] if isinstance(byte, bytes) else int(byte) for byte in value.byte_array_value]
    if kind == _BOOL_ARRAY:
        return list(value.bool_array_value)
    if kind == _INTEGER_ARRAY:
        return list(value.integer_array_value)
    if kind == _DOUBLE_ARRAY:
        return list(value.double_array_value)
    if kind == _STRING_ARRAY:
        return list(value.string_array_value)
    raise ValueError(f"unsupported ROS parameter type {kind}")


def _schedule_phases(payload: str) -> list[dict[str, Any]] | None:
    try:
        return [
            {
                "phase_label": str(phase["name"]),
                "phase_kind": str(phase["kind"]),
                "vx_target": float(phase["vx_target"]),
                "vy_target": float(phase["vy_target"]),
                "wz_target": float(phase["wz_target"]),
                "duration_s": float(phase["duration_s"]),
                "ramp_s": float(phase["ramp_s"]),
                "turn_radius_m": float(phase["radius_m"]),
            }
            for phase in json.loads(payload)["phases"]
        ]
    except (ValueError, KeyError, TypeError) as error:
        _log.warning(f"characterization_schedule payload rejected: {error!r}")
        return None
