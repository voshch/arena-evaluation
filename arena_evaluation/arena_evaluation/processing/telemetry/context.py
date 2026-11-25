"""Episode lifecycle, world semantics, plans, characterization and pose markers, and transforms as flat rows."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

from arena_evaluation.processing.telemetry.messages import (
    Envelope,
    RobotRef,
    env_row,
    robot_row,
    stamp_ns,
    yaw_from_orientation,
)
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_runtime_msgs.msg import SemanticSnapshot
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import String as RosString
    from tf2_msgs.msg import TFMessage

    from arena_evaluation.processing.legacy_schemas import EpisodeFields, FleetRobot


class _KeyValueProto(Protocol):
    key: object
    value: object


_log = logging.getLogger(__name__)


def record_episode(sink: TableSink, envelope: Envelope[object], fields: EpisodeFields) -> None:
    """One lifecycle row and its integrity flags from the generation-normalised episode record."""
    sink.append(
        "episode_lifecycle",
        env_row(
            envelope,
            episode_id=fields.episode_id,
            outcome_state=fields.outcome_state,
            outcome_info=fields.outcome_info,
            goal_uuid=fields.goal_uuid,
            task_layer=fields.task_layer,
            modes_json=fields.modes_json,
            robots_json=json.dumps(list(fields.robots)),
            goal_dist_start_m=fields.goal_dist_start,
            goal_dist_min_m=fields.goal_dist_min,
            path_length_m=fields.path_length,
            start_t_ns=fields.start_t_ns,
            phases_yaml=fields.phases,
            tasks_json=fields.tasks,
            params_yaml=fields.params_yaml,
            conditions_yaml=fields.conditions,
            world=fields.world,
            seed=fields.seed,
            human_params=fields.human_params_json,
        ),
    )
    for flag_idx, flag in enumerate(fields.integrity):
        sink.append("episode_integrity", env_row(envelope, flag_idx=flag_idx, flag=flag))


class FleetRecorder:
    """Each robot's identity and resolved parameters from the latched fleet snapshot, once per robot and environment."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._recorded: set[tuple[str, str]] = set()

    def record(self, t_ns: int, env_id: str, robots: Sequence[FleetRobot]) -> None:
        for state in robots:
            name = str(state.descriptor.name)
            if (env_id, name) in self._recorded:
                continue
            self._recorded.add((env_id, name))
            robot_id = f"{env_id}_{name}"  # the name robot_id_of gives this robot's own topics
            self._sink.append(
                "robot_descriptor",
                {
                    "t_ns": t_ns,
                    "env_id": env_id,
                    "robot_id": robot_id,
                    "name": name,
                    "model": str(state.descriptor.model),
                    "ns": str(state.descriptor.ns),
                    "frame": str(state.descriptor.frame),
                    "base_frame": str(state.descriptor.base_frame),
                },
            )
            for param in cast(Iterable[_KeyValueProto], state.params):
                self._sink.append(
                    "robot_params",
                    {
                        "t_ns": t_ns,
                        "env_id": env_id,
                        "robot_id": robot_id,
                        "param_key": str(param.key),
                        "param_value": str(param.value),
                    },
                )


def record_semantics(sink: TableSink, envelope: Envelope[SemanticSnapshot]) -> None:
    snapshot = envelope.body
    env_id = f"env_{int(snapshot.env_id)}"
    for entity in snapshot.entities:
        base = {
            "t_ns": envelope.t_ns,
            "env_id": env_id,
            "world": str(snapshot.world),
            "entity_name": str(entity.entity),
            "kind": str(entity.kind),
        }
        for name, value in zip(entity.discrete_names, entity.discrete_values, strict=True):
            sink.append(
                "world_semantics",
                {
                    **base,
                    "field_name": str(name),
                    "field_kind": "discrete",
                    "value_str": str(value),
                    "value_num": None,
                    "value_bool": None,
                },
            )
        for name, value in zip(entity.continuous_names, entity.continuous_values, strict=True):
            sink.append(
                "world_semantics",
                {
                    **base,
                    "field_name": str(name),
                    "field_kind": "continuous",
                    "value_str": None,
                    "value_num": float(value),
                    "value_bool": None,
                },
            )
        for name, value in zip(entity.predicate_names, entity.predicate_values, strict=True):
            sink.append(
                "world_semantics",
                {
                    **base,
                    "field_name": str(name),
                    "field_kind": "predicate",
                    "value_str": None,
                    "value_num": None,
                    "value_bool": bool(value),
                },
            )
        members = ",".join(sorted(str(member) for member in entity.members))
        sink.append(
            "world_semantics",
            {
                **base,
                "field_name": "members",
                "field_kind": "members",
                "value_str": members,
                "value_num": None,
                "value_bool": None,
            },
        )


class PlanRecorder:
    """Writes each global plan as numbered waypoint rows, one plan per message and robot."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._plan_counts: defaultdict[RobotRef, int] = defaultdict(int)

    def record(self, envelope: Envelope[Path]) -> None:
        if envelope.ref is None:
            raise ValueError("PlanRecorder requires an envelope with a RobotRef")
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
    sink.append(
        "robot_characterization_phase",
        robot_row(envelope, label=str(envelope.body.data)),
    )


def record_hearing_search_state(sink: TableSink, envelope: Envelope[RosString]) -> None:
    sink.append(
        "robot_hearing_search_state",
        robot_row(envelope, state=str(envelope.body.data)),
    )


class ScheduleRecorder:
    """Keeps the characterization schedule the sweep actually ran: the latest valid payload per robot wins."""

    def __init__(self) -> None:
        self._latest: dict[RobotRef, tuple[int, list[dict[str, Any]]]] = {}

    def record(self, envelope: Envelope[RosString]) -> None:
        if envelope.ref is None:
            raise ValueError("ScheduleRecorder requires an envelope with a RobotRef")
        phases = _schedule_phases(str(envelope.body.data))
        if phases is not None:
            self._latest[envelope.ref] = (envelope.t_ns, phases)

    def write(self, sink: TableSink) -> None:
        for ref, (t_ns, phases) in self._latest.items():
            for phase_idx, phase in enumerate(phases):
                sink.append(
                    "robot_characterization_schedule",
                    {
                        "t_ns": t_ns,
                        "env_id": ref.env_id,
                        "robot_id": ref.robot_id,
                        "phase_idx": phase_idx,
                        **phase,
                    },
                )


def record_initial_pose(sink: TableSink, envelope: Envelope[PoseWithCovarianceStamped]) -> None:
    pose = envelope.body.pose.pose
    sink.append(
        "robot_initialpose",
        robot_row(
            envelope,
            pos_x=float(pose.position.x),
            pos_y=float(pose.position.y),
            yaw=yaw_from_orientation(pose.orientation),
        ),
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


def record_goal_pose(sink: TableSink, envelope: Envelope[PoseStamped]) -> None:
    message = envelope.body
    sink.append(
        "robot_goal_pose",
        robot_row(
            envelope,
            pos_x=float(message.pose.position.x),
            pos_y=float(message.pose.position.y),
            pos_z=float(message.pose.position.z),
            yaw=yaw_from_orientation(message.pose.orientation),
            frame_id=str(message.header.frame_id),
        ),
    )


def record_occupancy_grid(sink: TableSink, envelope: Envelope[OccupancyGrid], kind: str) -> None:
    """One metadata row and one row per non-free cell of a latched occupancy-grid snapshot."""
    grid = envelope.body
    info = grid.info
    origin = info.origin
    sink.append(
        "environment_map",
        env_row(
            envelope,
            kind=kind,
            frame_id=str(grid.header.frame_id),
            resolution_m=float(info.resolution),
            width_cells=int(info.width),
            height_cells=int(info.height),
            origin_x=float(origin.position.x),
            origin_y=float(origin.position.y),
            origin_yaw=yaw_from_orientation(origin.orientation),
        ),
    )
    width = int(info.width)
    for index, value in enumerate(grid.data):
        occupancy = int(value)
        if occupancy == 0:
            continue
        sink.append(
            "environment_map_cells",
            env_row(
                envelope,
                kind=kind,
                row=index // width,
                col=index % width,
                occupancy=occupancy,
            ),
        )


def record_transforms(sink: TableSink, table: str, envelope: Envelope[TFMessage]) -> None:
    for transform in envelope.body.transforms:
        translation, rotation = (
            transform.transform.translation,
            transform.transform.rotation,
        )
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
