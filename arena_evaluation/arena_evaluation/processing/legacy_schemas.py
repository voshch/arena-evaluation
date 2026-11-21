"""Recognise recorded message generations and normalise legacy ones into the native field set."""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib
import re
import typing
from collections.abc import Iterable, Mapping, Sequence

import yaml
from mcap.reader import NonSeekingReader, make_reader

if typing.TYPE_CHECKING:
    from mcap.records import Schema

NATIVE: typing.Final[str] = "native"
LEGACY_2026_09: typing.Final[str] = "legacy_2026_09"
LEGACY_2026_11: typing.Final[str] = "legacy_2026_11"
UNKNOWN: typing.Final[str] = "unknown"

NATIVE_EPISODE_SCHEMA: typing.Final[str] = "arena_runtime_msgs/msg/EpisodeRecord"
LEGACY_EPISODE_SCHEMA: typing.Final[str] = "task_generator_msgs/msg/EpisodeRecord"
_SUPPORTED_FLEETS: typing.Final[tuple[str, ...]] = (
    "arena_runtime_msgs/msg/RobotFleet",
    "task_generator_msgs/msg/RobotFleet",
    "arena_runtime_msgs/msg/RobotState",
    "task_generator_msgs/msg/RobotState",
)
_ROBOT_SETTINGS: typing.Final[tuple[str, ...]] = (
    "robot_awareness",
    "robot_fov_gain",
    "robot_repulsion_gain",
    "robot_reaction_gain",
)
_EPISODE_FIELD_LINE: typing.Final[re.Pattern[str]] = re.compile(
    r"(?P<type>[A-Za-z][\w/]*(?:\[[^\]]*\])?)\s+(?P<name>[A-Za-z_]\w*)"
    r"(?:\s+\S.*)?"
)
_NATIVE_EPISODE_FIELDS: typing.Final[frozenset[str]] = frozenset(
    {
        "episode_id",
        "world",
        "seed",
        "task_layer",
        "modes",
        "robots",
        "outcome_state",
        "outcome_info",
        "goal_uuid",
        "goal_dist_start",
        "goal_dist_min",
        "path_length",
        "start_time",
        "integrity",
        "params",
        "human_params",
        "conditions",
        "phases",
    }
)
_LEGACY_REQUIRED_FIELDS: typing.Final[frozenset[str]] = frozenset(
    {
        "episode_id",
        "world",
        "seed",
        "tm_robots",
        "tm_obstacles",
        "tm_modules",
        "robots",
        "outcome_state",
        "outcome_info",
        "goal_uuid",
        "goal_dist_start",
        "goal_dist_min",
        "path_length",
        "start_time",
        "obstacles_params",
        "robots_params",
    }
)
_LEGACY_OPTIONAL_FIELDS: typing.Final[tuple[str, ...]] = (
    "integrity",
    "human_params",
    "conditions",
    "phases",
)
_LEGACY_EPISODE_FIELDS: typing.Final[frozenset[str]] = _LEGACY_REQUIRED_FIELDS | frozenset(_LEGACY_OPTIONAL_FIELDS)
_DESCRIPTOR_REQUIRED_FIELDS: typing.Final[frozenset[str]] = frozenset({"name", "model", "ns", "frame"})
_DESCRIPTOR_FIELDS: typing.Final[frozenset[str]] = _DESCRIPTOR_REQUIRED_FIELDS | {"base_frame"}
_PEDESTRIAN_REQUIRED_FIELDS: typing.Final[frozenset[str]] = frozenset({"name", "id", "pose", "twist", "animation_state"})
_PEDESTRIAN_FIELDS: typing.Final[frozenset[str]] = _PEDESTRIAN_REQUIRED_FIELDS | {
    "joint_state",
    "gait_phase",
    "model_uri",
    "gestures",
    "body",
    "radius",
    "device",
    "interaction_id",
    "interaction_type",
    "agent_type",
    "gait_cadence",
}


class LegacySchemaError(ValueError):
    """A recorded schema the migration cannot map; the caller turns it into a hinted MigrationError."""


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeSchema:
    """The fields one recorded EpisodeRecord definition declares, by field name and ROS type."""

    name: str
    types: Mapping[str, str]

    def carries(self, field: str) -> bool:
        """Whether the recorded definition declares this field."""
        return field in self.types

    def is_single_bool(self, field: str) -> bool:
        """Whether the recorded definition stores this field as one bool rather than a list."""
        return self.types[field] == "bool"


@dataclasses.dataclass(frozen=True, slots=True)
class RecordingGeneration:
    """What the MCAP's channel and schema records say about the recording's generation."""

    name: str
    episode_schema: str | None
    fleet_schema: str | None
    agent_meta_schema: str | None
    agent_meta_has_crowd: bool
    unsupported: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class FleetDescriptor:
    """One recorded robot descriptor, normalised to the native identity fields."""

    name: str
    model: str
    ns: str
    frame: str
    base_frame: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class FleetRobot:
    """One fleet member: its normalised descriptor and its recorded parameters."""

    descriptor: FleetDescriptor
    params: tuple[object, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeFields:
    """The native EpisodeRecord fields, with a legacy generation's absent fields NULL and named."""

    episode_id: int
    world: str | None
    seed: int | None
    task_layer: str | None
    modes_json: str | None
    robots: tuple[str, ...]
    outcome_state: int
    outcome_info: str
    goal_uuid: str
    goal_dist_start: float
    goal_dist_min: float
    path_length: float
    start_t_ns: int | None
    integrity: tuple[str, ...]
    params_yaml: str | None
    human_params_json: str | None
    conditions: str | None
    phases: str | None
    not_recorded: tuple[str, ...]


def detect(recording: pathlib.Path) -> RecordingGeneration:
    """Classify a recording from its channel and schema records, naming every unsupported construct."""
    topics = _topic_schemas(recording)
    episode = _schema_of(topics, "/state/episode")
    fleet = _schema_of(topics, "/state/robots")
    agent_meta = _schema_of(topics, "/agent_meta")
    agent_text = topics.get(_topic_by_suffix(topics, "/agent_meta"), (None, ""))[1]
    episode_text = topics.get(_topic_by_suffix(topics, "/state/episode"), (None, ""))[1]
    crowd = "robot_awareness" in agent_text
    unsupported: list[str] = []
    if episode is None:
        unsupported.append("no /state/episode topic; the generation is inferred from the present schemas")
        if agent_meta is not None and not crowd:
            name = LEGACY_2026_11
        elif fleet is not None and fleet.startswith("task_generator_msgs/"):
            name = LEGACY_2026_09
        else:
            name = NATIVE
    elif episode == NATIVE_EPISODE_SCHEMA:
        name = NATIVE if crowd or agent_meta is None else LEGACY_2026_11
    elif episode == LEGACY_EPISODE_SCHEMA:
        name = LEGACY_2026_09
    else:
        unsupported.append(f"episode schema {episode!r} is not a supported generation")
        name = UNKNOWN
    if episode == LEGACY_EPISODE_SCHEMA and episode_text:
        try:
            parsed = episode_schema(episode, episode_text)
        except LegacySchemaError as error:
            unsupported.append(str(error))
            name = UNKNOWN
        else:
            unsupported.extend(f"EpisodeRecord does not carry {field}; the migrated value is NULL (not_recorded)" for field in _legacy_not_recorded(parsed))
    if fleet is None:
        unsupported.append("no /state/robots topic; the descriptor must be synthesised")
    elif fleet not in _SUPPORTED_FLEETS:
        unsupported.append(f"fleet schema {fleet!r} is not supported")
    if agent_meta is None:
        unsupported.append("no /agent_meta topic; the crowd settings are NULL")
    elif not crowd:
        unsupported.append("AgentMeta has no robot_* fields; the crowd settings are NULL")
    return RecordingGeneration(
        name=name,
        episode_schema=episode,
        fleet_schema=fleet,
        agent_meta_schema=agent_meta,
        agent_meta_has_crowd=crowd,
        unsupported=tuple(unsupported),
    )


def recording_topics(recording: pathlib.Path) -> tuple[str, ...]:
    """Every topic the recording's channel records name, from its summary when it exists."""
    return tuple(_topic_schemas(recording))


def agent_meta_has_crowd(schema_text: str) -> bool:
    """Whether the recorded AgentMeta schema declares the four robot interaction settings."""
    return all(setting in schema_text for setting in _ROBOT_SETTINGS)


def _validated_fields(
    name: str,
    fields: Mapping[str, str],
    known: frozenset[str],
    required: frozenset[str],
    noun: str,
) -> dict[str, str]:
    """One recorded message's fields, refusing any field the migration does not map and any missing required field."""
    unsupported = sorted(set(fields) - known)
    if unsupported:
        raise LegacySchemaError(f"the {name} recording declares unsupported {noun} field(s) {unsupported}; the migration maps only the known generation's fields")
    missing = sorted(required - set(fields))
    if missing:
        raise LegacySchemaError(f"the {name} recording lacks required {noun} field(s) {missing}; the migration cannot map this construction")
    return dict(fields)


@functools.cache
def episode_schema(name: str, definition: str) -> EpisodeSchema:
    """The declared fields of one recorded EpisodeRecord schema; loud on an unsupported construction."""
    if name == NATIVE_EPISODE_SCHEMA:
        known, required = _NATIVE_EPISODE_FIELDS, _NATIVE_EPISODE_FIELDS
    elif name == LEGACY_EPISODE_SCHEMA:
        known, required = _LEGACY_EPISODE_FIELDS, _LEGACY_REQUIRED_FIELDS
    else:
        raise LegacySchemaError(f"episode schema {name!r} is not a supported generation; the migration supports arena_runtime_msgs and task_generator_msgs EpisodeRecord")
    types = _field_types(definition.split("\n====", 1)[0].splitlines())
    if not types:
        raise LegacySchemaError("the EpisodeRecord schema declares no field")
    return EpisodeSchema(
        name=name,
        types=_validated_fields(name, types, known, required, "EpisodeRecord"),
    )


def _field_types(lines: Iterable[str]) -> dict[str, str]:
    """Field name -> ROS type of one message definition body, constants left out."""
    fields: dict[str, str] = {}
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "=" in stripped:
            continue
        match = _EPISODE_FIELD_LINE.fullmatch(stripped)
        if match is None:
            raise LegacySchemaError(f"cannot parse the message schema line {stripped!r}; the migration maps plain fields and constants only")
        fields[match["name"]] = match["type"]
    return fields


def _dependency_fields(definition: str, message: str) -> dict[str, str]:
    """The fields of one nested message definition the recorded schema carries."""
    for block in re.split(r"\n=+\n", definition):
        lines = block.strip().splitlines()
        if not lines or not lines[0].strip().startswith("MSG:"):
            continue
        header = lines[0].strip().removeprefix("MSG:").strip()
        if header.rsplit("/", 1)[-1] == message:
            return _field_types(lines[1:])
    raise LegacySchemaError(f"the recorded schema declares no {message} definition; the migration cannot map the fleet descriptor")


@functools.cache
def _descriptor_fields(schema_name: str, definition: str) -> frozenset[str]:
    """The RobotDescriptor fields the recording declared, loud on an unsupported identity shape."""
    return frozenset(
        _validated_fields(
            schema_name,
            _dependency_fields(definition, "RobotDescriptor"),
            _DESCRIPTOR_FIELDS,
            _DESCRIPTOR_REQUIRED_FIELDS,
            "RobotDescriptor",
        )
    )


def episode_fields(record: object, schema: Schema) -> EpisodeFields:
    """The native field set of one EpisodeRecord generation, absent legacy fields NULL and named."""
    parsed = episode_schema(schema.name, schema.data.decode("utf-8", "replace"))
    if parsed.name == NATIVE_EPISODE_SCHEMA:
        return _episode_fields(record, parsed, legacy=False)
    if parsed.name == LEGACY_EPISODE_SCHEMA:
        return _episode_fields(record, parsed, legacy=True)
    raise LegacySchemaError(f"episode schema {parsed.name!r} is not a supported generation; the migration supports arena_runtime_msgs and task_generator_msgs EpisodeRecord")


def _legacy_not_recorded(schema: EpisodeSchema) -> tuple[str, ...]:
    """The native fields one legacy generation never recorded, keyed on that generation's schema."""
    return tuple(field for field in ("task_layer", *_LEGACY_OPTIONAL_FIELDS) if not schema.carries(field))


def fleet_robots(fleet: object, schema: Schema) -> tuple[FleetRobot, ...]:
    """Normalise a fleet snapshot, its descriptor read as the recording itself declared it."""
    if schema.name.endswith("/RobotFleet"):
        states = tuple(fleet.robots)
    elif schema.name.endswith("/RobotState"):
        states = (fleet,)
    elif isinstance(fleet, Sequence):
        states = tuple(fleet)
    else:
        raise LegacySchemaError(f"fleet schema {schema.name!r} carries no RobotState sequence")
    fields = _descriptor_fields(schema.name, schema.data.decode("utf-8", "replace"))
    return tuple(
        FleetRobot(
            descriptor=FleetDescriptor(
                name=str(state.descriptor.name),
                model=str(state.descriptor.model),
                ns=str(state.descriptor.ns),
                frame=str(state.descriptor.frame),
                base_frame=(str(state.descriptor.base_frame) if "base_frame" in fields else None),
            ),
            params=tuple(state.params),
        )
        for state in states
    )


@functools.cache
def pedestrian_schema(name: str, definition: str) -> EpisodeSchema:
    """The Pedestrian fields the recording declared, loud on an unsupported construction."""
    return EpisodeSchema(
        name=name,
        types=_validated_fields(
            name,
            _dependency_fields(definition, "Pedestrian"),
            _PEDESTRIAN_FIELDS,
            _PEDESTRIAN_REQUIRED_FIELDS,
            "Pedestrian",
        ),
    )


def pedestrian_fields(schema: Schema) -> EpisodeSchema:
    """The Pedestrian fields of one recorded Pedestrians schema."""
    return pedestrian_schema(schema.name, schema.data.decode("utf-8", "replace"))


def legacy_modes(*, robots: str | None, obstacles: str | None, modules: Iterable[str] | None) -> dict[str, list[str]]:
    """`modes` of a record made before modes: the tm_* fields it carried, an absent field leaves its axis out."""
    from arena_runtime.env.task_layer import TM_AXES

    given = dict(zip(TM_AXES, (robots, obstacles, modules), strict=True))
    return {axis: [name for name in ([value] if isinstance(value, str) else value) if name] for axis, value in given.items() if value is not None}


def legacy_params(
    modes: Mapping[str, Sequence[str]],
    robots_params: Mapping[str, typing.Any],
    obstacles_params: Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    """Full-name `params` of a record made before modes: a leaf param of the robots or obstacles mode `m` was `task.<m>.<leaf>`."""
    from arena_runtime.env.task_layer import TM_AXES

    robots_axis, obstacles_axis, _ = TM_AXES
    return {
        f"task.{mode}.{leaf}": value
        for axis, leaves in (
            (robots_axis, robots_params),
            (obstacles_axis, obstacles_params),
        )
        for mode in list(modes.get(axis, []))[:1]
        for leaf, value in leaves.items()
    }


def flatten_params(nested: Mapping[str, typing.Any], prefix: str = "") -> dict[str, typing.Any]:
    """Dotted leaf names of a nested params mapping."""
    flat: dict[str, typing.Any] = {}
    for key, value in nested.items():
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def modes_json(modes: Mapping[str, Sequence[str]]) -> str | None:
    """The recorded `modes_json` payload, None when the generation carried no mode."""
    return json.dumps({axis: ",".join(names) for axis, names in modes.items()}) if modes else None


def params_yaml(params: Mapping[str, typing.Any]) -> str | None:
    """The recorded `params_yaml` payload, None when the generation carried no parameter."""
    return yaml.dump(dict(params)) if params else None


def _parameter_values(parameters: Iterable[object]) -> dict[str, object]:
    """The recorded name -> value of one rcl_interfaces Parameter sequence."""
    from arena_evaluation.processing.telemetry.messages import parameter_value

    return {parameter.name: parameter_value(parameter.value) for parameter in parameters}


def _integrity_flags(record: object, schema: EpisodeSchema) -> tuple[str, ...]:
    """The recorded integrity reasons, read as the generation's own declared field type."""
    if not schema.carries("integrity"):
        return ()
    if schema.is_single_bool("integrity"):
        return ("recording_integrity_flag",) if record.integrity else ()
    if schema.types["integrity"] != "string[]":
        raise LegacySchemaError(f"the {schema.name} recording stores integrity as {schema.types['integrity']!r}; the migration maps bool and string[] only")
    return tuple(str(flag) for flag in typing.cast(typing.Iterable[object], record.integrity))


def _episode_fields(record: object, schema: EpisodeSchema, *, legacy: bool) -> EpisodeFields:
    """The shared native field set, with the generation-specific modes/params/optional fields filled in."""
    robot_names = tuple(str(robot) for robot in record.robots)
    integrity = _integrity_flags(record, schema)
    if legacy:
        modes = legacy_modes(
            robots=record.tm_robots,
            obstacles=record.tm_obstacles,
            modules=record.tm_modules,
        )
        params = legacy_params(
            modes,
            flatten_params(_parameter_values(record.robots_params)),
            flatten_params(_parameter_values(record.obstacles_params)),
        )
        human = _parameter_values(record.human_params) if schema.carries("human_params") else {}
        task_layer, modes_payload, params_payload = (
            None,
            modes_json(modes),
            params_yaml(params),
        )
        conditions = (str(record.conditions) or None) if schema.carries("conditions") else None
        phases = (str(record.phases) or None) if schema.carries("phases") else None
        not_recorded = _legacy_not_recorded(schema)
    else:
        modes = {mode.key: mode.value for mode in record.modes}
        params = _parameter_values(record.params)
        human = _parameter_values(record.human_params)
        task_layer = str(record.task_layer) or None
        modes_payload = json.dumps(modes) if modes else None
        params_payload = yaml.dump(params) if params else None
        conditions = str(record.conditions) or None
        phases = str(record.phases) or None
        not_recorded = ()
    return EpisodeFields(
        episode_id=int(record.episode_id),
        world=str(record.world) if record.world else None,
        seed=int(record.seed) if record.seed >= 0 else None,
        task_layer=task_layer,
        modes_json=modes_payload,
        robots=robot_names,
        outcome_state=int(record.outcome_state),
        outcome_info=str(record.outcome_info),
        goal_uuid=str(record.goal_uuid),
        goal_dist_start=float(record.goal_dist_start),
        goal_dist_min=float(record.goal_dist_min),
        path_length=float(record.path_length),
        start_t_ns=int(record.start_time.sec) * 1_000_000_000 + int(record.start_time.nanosec),
        integrity=integrity,
        params_yaml=params_payload,
        human_params_json=json.dumps(human) if human else None,
        conditions=conditions,
        phases=phases,
        not_recorded=not_recorded,
    )


def _topic_schemas(recording: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Topic -> (schema name, schema text) of one recording, from the summary when it exists."""
    with recording.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    if summary is not None:
        by_id = dict(summary.schemas)
        return {
            channel.topic: (
                by_id[channel.schema_id].name,
                by_id[channel.schema_id].data.decode("utf-8", "replace"),
            )
            for channel in summary.channels.values()
            if channel.schema_id in by_id
        }
    topics: dict[str, tuple[str, str]] = {}
    with recording.open("rb") as handle:
        for schema, channel, _message in NonSeekingReader(handle).iter_messages(log_time_order=False):
            if schema is not None and channel.topic not in topics:
                topics[channel.topic] = (
                    schema.name,
                    schema.data.decode("utf-8", "replace"),
                )
    return topics


def _topic_by_suffix(topics: Mapping[str, tuple[str, str]], suffix: str) -> str | None:
    """The first topic that ends with the suffix, None when the recording has none."""
    return next((topic for topic in topics if topic.endswith(suffix)), None)


def _schema_of(topics: Mapping[str, tuple[str, str]], suffix: str) -> str | None:
    topic = _topic_by_suffix(topics, suffix)
    return None if topic is None else topics[topic][0]
