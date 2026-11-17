"""Schema-keyed tolerance for the September EpisodeRecord, fleet and pedestrian generations.

The September corpus predates ``human_params``, ``phases``, the fleet descriptor's
``base_frame`` and the pedestrian gestures/interaction fields. These tests drive the adapters with
recorded-shape schema texts, never with a real MCAP, so a regression shows up as a field access,
not as a silent NULL.
"""

from __future__ import annotations

import dataclasses
import types

import pytest

from arena_evaluation.processing import legacy_schemas

_SEPARATOR = "\n" + "=" * 80 + "\n"


@dataclasses.dataclass(frozen=True)
class _Schema:
    """The two attributes the adapters read from an ``mcap.records.Schema``."""

    name: str
    definition: str

    @property
    def data(self) -> bytes:
        return self.definition.encode("utf-8")


_SEPT_EPISODE = (
    "task_generator_msgs/msg/EpisodeRecord",
    "uint32 episode_id\n"
    "string world\n"
    "int64 seed\n"
    "string tm_robots\n"
    "string tm_obstacles\n"
    "string[] tm_modules\n"
    "string[] robots\n"
    "uint8 RUNNING = 1\n"
    "uint8 outcome_state\n"
    "string outcome_info\n"
    "string goal_uuid\n"
    "float32 goal_dist_start\n"
    "float32 goal_dist_min\n"
    "float32 path_length\n"
    "builtin_interfaces/Time start_time\n"
    "bool integrity  # False = manually mutated mid-episode.\n"
    "rcl_interfaces/Parameter[] obstacles_params\n"
    "rcl_interfaces/Parameter[] robots_params\n"
    "string conditions\n",
)


def _sept_record(**overrides: object) -> types.SimpleNamespace:
    values: dict[str, object] = {
        "episode_id": 433,
        "world": "library_acoustic_partition",
        "seed": 101,
        "tm_robots": "jackal",
        "tm_obstacles": "static",
        "tm_modules": [],
        "robots": ["jackal"],
        "outcome_state": 2,
        "outcome_info": "finished",
        "goal_uuid": "goal",
        "goal_dist_start": 1.0,
        "goal_dist_min": 0.5,
        "path_length": 2.0,
        "start_time": types.SimpleNamespace(sec=1, nanosec=0),
        "integrity": False,
        "obstacles_params": [],
        "robots_params": [],
        "conditions": '[{"op": "never", "p": "robot in restricted"}]',
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _sept_schema(
    definition: str | None = None, name: str = _SEPT_EPISODE[0]
) -> _Schema:
    return _Schema(name, definition if definition is not None else _SEPT_EPISODE[1])


def test_a_native_schema_missing_a_field_is_loud_and_named() -> None:
    definition = (
        "uint32 episode_id\n"
        "string world\n"
        "int64 seed\n"
        "string task_layer\n"
        "diagnostic_msgs/KeyValue[] modes\n"
        "string[] robots\n"
        "uint8 outcome_state\n"
        "string outcome_info\n"
        "string goal_uuid\n"
        "float32 goal_dist_start\n"
        "float32 goal_dist_min\n"
        "float32 path_length\n"
        "builtin_interfaces/Time start_time\n"
        "string[] integrity\n"
        "rcl_interfaces/Parameter[] params\n"
        "string conditions\n"
        "string phases\n"
    )

    with pytest.raises(legacy_schemas.LegacySchemaError, match="human_params"):
        legacy_schemas.episode_schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, definition)


_FLEET = "task_generator_msgs/msg/RobotFleet"
_ROBOT_STATE = (
    "MSG: task_generator_msgs/RobotState\n"
    "RobotDescriptor descriptor\n"
    "RobotCap[] caps\n"
    "diagnostic_msgs/KeyValue[] params\n"
)
_DESCRIPTOR = (
    "MSG: task_generator_msgs/RobotDescriptor\n"
    "string name\n"
    "string model\n"
    "string ns\n"
    "string frame\n"
)


def _fleet(descriptor: types.SimpleNamespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        robots=[types.SimpleNamespace(descriptor=descriptor, params=[])]
    )


def test_a_legacy_fleet_descriptor_without_base_frame_is_null() -> None:
    definition = (
        "RobotState[] robots\n" + _SEPARATOR + _ROBOT_STATE + _SEPARATOR + _DESCRIPTOR
    )
    descriptor = types.SimpleNamespace(
        name="jackal", model="jackal", ns="/env_3/jackal", frame="env_3/jackal"
    )

    robots = legacy_schemas.fleet_robots(
        _fleet(descriptor), _Schema(_FLEET, definition)
    )

    assert robots[0].descriptor.name == "jackal"
    assert robots[0].descriptor.base_frame is None


_PEDESTRIANS = "arena_people_msgs/msg/Pedestrians"
_SEPT_PEDESTRIAN = (
    "MSG: arena_people_msgs/Pedestrian\n"
    "string name\n"
    "uint32 id\n"
    "geometry_msgs/Pose pose\n"
    "geometry_msgs/Twist twist\n"
    "uint8 animation_state\n"
    "sensor_msgs/JointState joint_state\n"
    "float32 gait_phase\n"
    "string model_uri\n"
)


