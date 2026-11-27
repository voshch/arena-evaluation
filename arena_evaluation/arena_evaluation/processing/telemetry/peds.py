"""Pedestrian telemetry: each recorded pedestrian becomes one flat row per timestamp, never a stride-packed list."""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from arena_evaluation.processing import legacy_schemas
from arena_evaluation.processing.telemetry.messages import (
    Envelope,
    env_row,
    pedestrian_id,
    yaw_from_orientation,
)
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import AgentFrame, AgentStates
    from arena_humansim_msgs.msg import AgentMeta as RosAgentMeta
    from arena_people_msgs.msg import Pedestrians

HUMAN_KIND = 0
# The crowd engine reports "no interaction" as these values; the recorded row carries NULL instead.
NO_INTERACTION_ID = -1
NO_INTERACTION_TYPE = 0
ARENA_TABLE = "pedestrian_arena_peds"
ENGINE_TABLE = "pedestrian_agent_states"
GESTURE_TABLE = "pedestrian_gestures"
PEDESTRIANS_SCHEMA = "arena_people_msgs/msg/Pedestrians"
AGENT_FRAME_SCHEMA = "arena_humansim_msgs/msg/AgentFrame"
AGENT_STATES_SCHEMA = "arena_humansim_msgs/msg/AgentStates"


@dataclasses.dataclass(frozen=True, slots=True)
class CrowdSettings:
    """The robot-interaction settings latched in the agent metadata of one pedestrian; None where the generation did not carry one."""

    robot_range_offset: float | None
    robot_fov_offset: float | None
    robot_repulsion_gain: float
    robot_reaction_gain: float
    robot_blind: bool | None


@dataclasses.dataclass(frozen=True, slots=True)
class AgentMeta:
    """Crowd-engine metadata: policy names by index, and agent types and crowd settings by agent id."""

    policies: tuple[str, ...] = ()
    types_by_id: Mapping[int, str | None] = dataclasses.field(default_factory=dict)
    crowd_by_id: Mapping[int, CrowdSettings] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class _Agent:
    """One pedestrian of a frame, before it becomes a row of a pedestrian table."""

    number: int
    name: str
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    radius_m: float | None
    desired_speed: float | None
    policy: str | None
    agent_type: str | None
    animation_state: int | None
    interaction_id: int | None
    interaction_type: int | None
    gait_phase: float | None
    gait_cadence: float | None
    crowd: CrowdSettings | None

    def table_columns(self) -> dict[str, Any]:
        """The columns every pedestrian table carries; a value the message did not carry is None."""
        return {
            "ped_id": pedestrian_id(self.number),
            "ped_name": self.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "yaw": self.yaw,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "speed_mps": math.hypot(self.vx, self.vy),
            "radius_m": self.radius_m,
            "desired_speed": self.desired_speed,
            "policy": self.policy,
            "agent_type": self.agent_type,
            "animation_state": self.animation_state,
            "interaction_id": self.interaction_id,
            "interaction_type": self.interaction_type,
            "gait_phase": self.gait_phase,
            "gait_cadence": self.gait_cadence,
        }

    def crowd_columns(self) -> dict[str, Any]:
        """The five crowd settings of the agent; None where the agent metadata did not carry the agent."""
        if self.crowd is None:
            return {
                "robot_range_offset": None,
                "robot_fov_offset": None,
                "robot_repulsion_gain": None,
                "robot_reaction_gain": None,
                "robot_blind": None,
            }
        return {
            "robot_range_offset": self.crowd.robot_range_offset,
            "robot_fov_offset": self.crowd.robot_fov_offset,
            "robot_repulsion_gain": self.crowd.robot_repulsion_gain,
            "robot_reaction_gain": self.crowd.robot_reaction_gain,
            "robot_blind": self.crowd.robot_blind,
        }


class PedestrianRecorder:
    """Writes pedestrian frames as flat rows and remembers the frames each environment was stamped in."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._meta: dict[str, AgentMeta] = {}
        self._frames: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._names: set[str] = set()

    def record_meta(self, envelope: Envelope[RosAgentMeta], meta: AgentMeta) -> None:
        """Keep the normalised agent metadata of one environment."""
        if envelope.ref is None:
            raise ValueError("PedestrianRecorder.record_meta requires an envelope with a RobotRef")
        self._meta[envelope.ref.env_id] = meta

    def record_frame(self, table: str, envelope: Envelope[Pedestrians | AgentFrame | AgentStates]) -> None:
        if envelope.ref is None:
            raise ValueError("PedestrianRecorder.record_frame requires an envelope with a RobotRef")
        env_id = envelope.ref.env_id
        self._frames[(env_id, table)].add(envelope.body.header.frame_id.strip("/"))
        for agent in _agents(envelope, self._meta.get(env_id, AgentMeta())):
            self._names.add(agent.name.strip("/"))
            columns = agent.table_columns()
            if table == ENGINE_TABLE:
                columns |= agent.crowd_columns()
            self._sink.append(table, env_row(envelope, **columns))
        for row in _gesture_rows(envelope):
            self._sink.append(GESTURE_TABLE, row)

    def frames(self) -> dict[tuple[str, str], frozenset[str]]:
        return {key: frozenset(frames) for key, frames in self._frames.items()}

    def names(self) -> frozenset[str]:
        """Every recorded pedestrian name, which is also the namespace of its tf frame."""
        return frozenset(self._names)


class _CrowdAgentMetaProto(Protocol):
    policies: Sequence[str]
    agent_id: Sequence[int]
    agent_type: Sequence[str]
    robot_range_offset: Sequence[float]
    robot_fov_offset: Sequence[float]
    robot_repulsion_gain: Sequence[float]
    robot_reaction_gain: Sequence[float]
    robot_blind: Sequence[bool]


class _GainsAgentMetaProto(Protocol):
    policies: Sequence[str]
    agent_id: Sequence[int]
    agent_type: Sequence[str]
    robot_repulsion_gain: Sequence[float]
    robot_reaction_gain: Sequence[float]


def agent_meta(meta: _CrowdAgentMetaProto) -> AgentMeta:
    """The normalised metadata of the native generation, whose AgentMeta carries the five robot settings."""
    policies = tuple(str(policy) for policy in meta.policies)
    types = {int(agent_id): _text_or_none(agent_type) for agent_id, agent_type in zip(meta.agent_id, meta.agent_type, strict=True)}
    crowd = {
        int(agent_id): CrowdSettings(float(range_offset), float(fov_offset), float(repulsion), float(reaction), bool(blind))
        for agent_id, range_offset, fov_offset, repulsion, reaction, blind in zip(
            meta.agent_id,
            meta.robot_range_offset,
            meta.robot_fov_offset,
            meta.robot_repulsion_gain,
            meta.robot_reaction_gain,
            meta.robot_blind,
            strict=True,
        )
    }
    return AgentMeta(policies=policies, types_by_id=types, crowd_by_id=crowd)


def agent_meta_gains_only(meta: _GainsAgentMetaProto) -> AgentMeta:
    """The normalised metadata of the 2026-11-11 AgentMeta: only the two gains carry over, the offsets and blindness stay absent."""
    policies = tuple(str(policy) for policy in meta.policies)
    types = {int(agent_id): _text_or_none(agent_type) for agent_id, agent_type in zip(meta.agent_id, meta.agent_type, strict=True)}
    crowd = {
        int(agent_id): CrowdSettings(None, None, float(repulsion), float(reaction), None)
        for agent_id, repulsion, reaction in zip(
            meta.agent_id,
            meta.robot_repulsion_gain,
            meta.robot_reaction_gain,
            strict=True,
        )
    }
    return AgentMeta(policies=policies, types_by_id=types, crowd_by_id=crowd)


def agent_meta_without_robot_fields(meta: RosAgentMeta, *, has_agent_type: bool = True) -> AgentMeta:
    """The normalised metadata of a Nov-8 AgentMeta: the crowd settings are absent, never invented."""
    types = {int(agent_id): _text_or_none(agent_type) for agent_id, agent_type in zip(meta.agent_id, meta.agent_type, strict=True)} if has_agent_type else {}
    return AgentMeta(
        policies=tuple(str(policy) for policy in meta.policies),
        types_by_id=types,
        crowd_by_id={},
    )


def _agents(envelope: Envelope[Pedestrians | AgentFrame | AgentStates], meta: AgentMeta) -> list[_Agent]:
    schema = envelope.schema.name
    if schema == PEDESTRIANS_SCHEMA:
        columns = _pedestrians_columns(cast("Envelope[Pedestrians]", envelope))
    elif schema == AGENT_FRAME_SCHEMA:
        columns = _agent_frame_columns(cast("Envelope[AgentFrame]", envelope), meta)
    elif schema == AGENT_STATES_SCHEMA:
        columns = _agent_states_columns(cast("Envelope[AgentStates]", envelope), meta)
    else:
        raise ValueError(f"unsupported pedestrian schema {schema!r}")
    return [_Agent(**dict(zip(columns, values, strict=True))) for values in zip(*columns.values(), strict=True)]


def _pedestrians_columns(envelope: Envelope[Pedestrians]) -> dict[str, list[Any]]:
    peds = envelope.body.pedestrians
    nulls = [None] * len(peds)
    fields = legacy_schemas.pedestrian_fields(envelope.schema)
    return {
        "number": [int(ped.id) for ped in peds],
        "name": [ped.name or str(ped.id) for ped in peds],
        "x": [float(ped.pose.position.x) for ped in peds],
        "y": [float(ped.pose.position.y) for ped in peds],
        "z": [float(ped.pose.position.z) for ped in peds],
        "yaw": [yaw_from_orientation(ped.pose.orientation) for ped in peds],
        "vx": [float(ped.twist.linear.x) for ped in peds],
        "vy": [float(ped.twist.linear.y) for ped in peds],
        "vz": [float(ped.twist.linear.z) for ped in peds],
        "radius_m": nulls,
        "desired_speed": nulls,
        "policy": nulls,
        "agent_type": ([_text_or_none(ped.agent_type) for ped in peds] if fields.carries("agent_type") else nulls),
        "animation_state": [int(ped.animation_state) for ped in peds],
        "interaction_id": ([int(ped.interaction_id) for ped in peds] if fields.carries("interaction_id") else nulls),
        "interaction_type": ([int(ped.interaction_type) for ped in peds] if fields.carries("interaction_type") else nulls),
        "gait_phase": ([_nonzero(float(ped.gait_phase)) for ped in peds] if fields.carries("gait_phase") else nulls),
        "gait_cadence": ([_nonzero(float(ped.gait_cadence)) for ped in peds] if fields.carries("gait_cadence") else nulls),
        "crowd": nulls,
    }


def _agent_frame_columns(envelope: Envelope[AgentFrame], meta: AgentMeta) -> dict[str, list[Any]]:
    frame = envelope.body
    schema_text = envelope.schema.data.decode("utf-8", "replace")
    humans = _uint8(frame.kind) == HUMAN_KIND
    agent_ids = np.asarray(frame.agent_id, dtype=np.int64)[humans].tolist()
    policy_indices = np.asarray(frame.policy_idx, dtype=np.int64)[humans].tolist()
    nulls: list[Any] = [None] * len(agent_ids)
    return {
        "number": agent_ids,
        "name": [str(agent_id) for agent_id in agent_ids],
        "x": _floats(frame.x, humans),
        "y": _floats(frame.y, humans),
        "z": [0.0] * len(agent_ids),
        "yaw": _floats(frame.theta, humans),
        "vx": _floats(frame.vx, humans),
        "vy": _floats(frame.vy, humans),
        "vz": [0.0] * len(agent_ids),
        "radius_m": [_nonzero(radius) for radius in _floats(frame.radius, humans)],
        "desired_speed": [_nonzero(speed) for speed in _floats(frame.desired_velocity, humans)],
        "policy": [_policy_name(meta.policies, index) for index in policy_indices],
        "agent_type": [meta.types_by_id.get(agent_id) for agent_id in agent_ids],
        "animation_state": _bytes(frame.animation_state, humans),
        "interaction_id": ([None if value == NO_INTERACTION_ID else value for value in np.asarray(frame.interaction_id, dtype=np.int64)[humans].tolist()] if "interaction_id" in schema_text else nulls),
        "interaction_type": ([None if value == NO_INTERACTION_TYPE else value for value in _bytes(frame.interaction_type, humans)] if "interaction_type" in schema_text else nulls),
        "gait_phase": ([_nonzero(phase) for phase in _floats(frame.gait_phase, humans)] if "gait_phase" in schema_text else nulls),
        "gait_cadence": ([_nonzero(cadence) for cadence in _floats(frame.gait_cadence, humans)] if "gait_cadence" in schema_text else nulls),
        "crowd": [meta.crowd_by_id.get(agent_id) for agent_id in agent_ids],
    }


def _agent_states_columns(envelope: Envelope[AgentStates], meta: AgentMeta) -> dict[str, list[Any]]:
    agents = [agent for agent in envelope.body.agents if agent.kind == HUMAN_KIND]
    nulls = [None] * len(agents)
    return {
        "number": [int(agent.agent_id) for agent in agents],
        "name": [agent.name or str(agent.agent_id) for agent in agents],
        "x": [float(agent.pose.x) for agent in agents],
        "y": [float(agent.pose.y) for agent in agents],
        "z": [0.0] * len(agents),
        "yaw": [float(agent.pose.theta) for agent in agents],
        "vx": [float(agent.velocity.x) for agent in agents],
        "vy": [float(agent.velocity.y) for agent in agents],
        "vz": [float(agent.velocity.z) for agent in agents],
        "radius_m": [_nonzero(float(agent.radius)) for agent in agents],
        "desired_speed": [_nonzero(float(agent.desired_velocity)) for agent in agents],
        "policy": [_text_or_none(agent.policy) for agent in agents],
        "agent_type": [_text_or_none(agent.agent_type) for agent in agents],
        "animation_state": [int(agent.animation_state) for agent in agents],
        "interaction_id": nulls,
        "interaction_type": nulls,
        "gait_phase": nulls,
        "gait_cadence": nulls,
        "crowd": [meta.crowd_by_id.get(int(agent.agent_id)) for agent in agents],
    }


def _gesture_rows(envelope: Envelope) -> list[dict[str, Any]]:
    if envelope.schema.name != PEDESTRIANS_SCHEMA:
        return []
    fields = legacy_schemas.pedestrian_fields(envelope.schema)
    if not fields.carries("gestures"):
        return []
    peds = cast("Envelope[Pedestrians]", envelope).body.pedestrians
    return [
        env_row(
            envelope,
            ped_id=pedestrian_id(int(ped.id)),
            slot=str(gesture.slot),
            clip=str(gesture.clip),
            hand=str(gesture.hand),
            at_x=float(gesture.at.x),
            at_y=float(gesture.at.y),
            at_z=float(gesture.at.z),
            render_pose_override=bool(gesture.render_pose_override),
        )
        for ped in peds
        for gesture in ped.gestures
    ]


def _floats(values: object, humans: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=np.float64)[humans].tolist()


def _uint8(values: object) -> np.ndarray:
    return np.frombuffer(values, dtype=np.uint8) if isinstance(values, (bytes, bytearray)) else np.asarray(values, dtype=np.uint8)


def _bytes(values: object, humans: np.ndarray) -> list[int]:
    return _uint8(values)[humans].tolist()


def _policy_name(policies: tuple[str, ...], index: int | None) -> str | None:
    return policies[index] if index is not None and 0 <= index < len(policies) else None


def _text_or_none(raw: object) -> str | None:
    return str(raw) or None


def _nonzero(value: float | None) -> float | None:
    """Zero is the producer's 'not provided' value for gait, radius and speed."""
    return value or None
