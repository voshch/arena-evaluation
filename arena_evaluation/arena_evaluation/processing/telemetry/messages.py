"""Decoded ROS message helpers, recorded schema declarations, and the robot and environment identities of messages."""

from __future__ import annotations

import dataclasses
import math
import re
import typing
from collections.abc import Sequence

if typing.TYPE_CHECKING:
    from geometry_msgs.msg import Quaternion
    from mcap.records import Schema
    from rcl_interfaces.msg import ParameterValue
    from std_msgs.msg import Header

NS_PER_SECOND = 1_000_000_000
_ENV_TOKEN = re.compile(r"(env_\d+)")
_CONTROLLER_SUFFIX = "_controller"
_POWER_PUBLISHER = "power_publisher"


@dataclasses.dataclass(frozen=True, slots=True)
class RobotRef:
    """Environment and robot of a message; environment-level topics name the environment as their robot."""

    env_id: str
    robot_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class Envelope[T]:
    """A decoded message with its log time, its recorded schema, and the robot it belongs to (None for a global transform)."""

    t_ns: int
    ref: RobotRef | None
    schema: Schema
    body: T


def stamp_ns(header: Header) -> int:
    """Header stamp of a message in nanoseconds."""
    return int(header.stamp.sec) * NS_PER_SECOND + int(header.stamp.nanosec)


def yaw_from_orientation(orientation: Quaternion) -> float:
    """Heading angle of a geometry_msgs quaternion."""
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def env_id_in(text: str) -> str | None:
    """The env_N token of a topic or frame name, or None when the name carries no environment."""
    match = _ENV_TOKEN.search(text)
    return match.group(1) if match else None


def topic_parts(topic: str) -> list[str]:
    """Non-empty path segments of a topic name."""
    return [part for part in topic.strip("/").split("/") if part]


def robot_id_of(parts: Sequence[str], env_id: str) -> str:
    """Robot named by the segment before the topic name; controller and power-publisher segments name their robot's parent."""
    if len(parts) < 2:
        raise ValueError(f"{'/'.join(parts)!r} names no robot; a robot topic has a robot segment before its name")
    segment = parts[-2]
    if segment == env_id:
        return env_id
    if segment.endswith(_CONTROLLER_SUFFIX) or segment == _POWER_PUBLISHER:
        segment = parts[-3] if len(parts) >= 3 else segment
    return f"{env_id}_{segment}"


def is_controller_topic(topic: str) -> bool:
    """Whether the topic is published by a robot's diff-drive controller."""
    parts = topic_parts(topic)
    return len(parts) >= 2 and parts[-2].endswith(_CONTROLLER_SUFFIX)


def pedestrian_id(number: int) -> str:
    """Canonical pedestrian identifier from the engine's numeric agent id."""
    return f"Pedestrian_{number}"


def robot_row(envelope: Envelope, **columns: typing.Any) -> dict[str, typing.Any]:
    """A row of a robot-scoped table."""
    return {
        "t_ns": envelope.t_ns,
        "env_id": envelope.ref.env_id,
        "robot_id": envelope.ref.robot_id,
        **columns,
    }


def env_row(envelope: Envelope, **columns: typing.Any) -> dict[str, typing.Any]:
    """A row of an environment-scoped table."""
    return {"t_ns": envelope.t_ns, "env_id": envelope.ref.env_id, **columns}


def value_at(values: Sequence[float], index: int) -> float | None:
    """The value at an index of a parallel array, None when the array is shorter."""
    return float(values[index]) if index < len(values) else None


_PARAM_EXTRACTORS: dict[int, typing.Callable[[ParameterValue], object]] = {
    0: lambda _value: None,
    1: lambda value: value.bool_value,
    2: lambda value: value.integer_value,
    3: lambda value: value.double_value,
    4: lambda value: value.string_value,
    5: lambda value: [byte[0] if isinstance(byte, bytes) else int(byte) for byte in value.byte_array_value],
    6: lambda value: list(value.bool_array_value),
    7: lambda value: list(value.integer_array_value),
    8: lambda value: list(value.double_array_value),
    9: lambda value: list(value.string_array_value),
}


def parameter_value(value: ParameterValue) -> object:
    """The Python value of one recorded ROS parameter; an unknown type is refused, never guessed."""
    extractor = _PARAM_EXTRACTORS.get(value.type)
    if extractor is None:
        raise ValueError(f"unsupported ROS parameter type {value.type}")
    return extractor(value)
