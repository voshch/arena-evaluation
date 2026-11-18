"""Recognise the recorded message generations and normalise the legacy ones into the native field set.

The tolerance lives here, keyed on the recorded schema, so the telemetry calculators only ever see
the native shape. A legacy recording never raises ``AttributeError`` from a calculator; a field its
generation did not record is NULL and named in ``not_recorded``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import typing
from collections.abc import Iterable, Mapping, Sequence

import yaml
from mcap.reader import NonSeekingReader, make_reader

NATIVE: typing.Final[str] = "native"
LEGACY_2026_09: typing.Final[str] = "legacy_2026_09"
LEGACY_2026_11: typing.Final[str] = "legacy_2026_11"
UNKNOWN: typing.Final[str] = "unknown"

NATIVE_EPISODE_SCHEMA: typing.Final[str] = "arena_runtime_msgs/msg/EpisodeRecord"
LEGACY_EPISODE_SCHEMA: typing.Final[str] = "task_generator_msgs/msg/EpisodeRecord"
AGENT_META_SCHEMA: typing.Final[str] = "arena_humansim_msgs/msg/AgentMeta"
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


class LegacySchemaError(ValueError):
    """A recorded schema the migration cannot map; the caller turns it into a hinted MigrationError."""


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
    crowd = "robot_awareness" in agent_text
    unsupported: list[str] = []
    if episode is None:
        unsupported.append(
            "no /state/episode topic; the generation is inferred from the present schemas"
        )
        name = _absent_episode_generation(fleet, agent_meta, crowd)
    elif episode == NATIVE_EPISODE_SCHEMA:
        name = NATIVE if crowd or agent_meta is None else LEGACY_2026_11
    elif episode == LEGACY_EPISODE_SCHEMA:
        name = LEGACY_2026_09
    else:
        unsupported.append(f"episode schema {episode!r} is not a supported generation")
        name = UNKNOWN
    if fleet is None:
        unsupported.append("no /state/robots topic; the descriptor must be synthesised")
    elif fleet not in _SUPPORTED_FLEETS:
        unsupported.append(f"fleet schema {fleet!r} is not supported")
    if agent_meta is None:
        unsupported.append("no /agent_meta topic; the crowd settings are NULL")
    elif not crowd:
        unsupported.append(
            "AgentMeta has no robot_* fields; the crowd settings are NULL"
        )
    return RecordingGeneration(
        name=name,
        episode_schema=episode,
        fleet_schema=fleet,
        agent_meta_schema=agent_meta,
        agent_meta_has_crowd=crowd,
        unsupported=tuple(unsupported),
    )


def _absent_episode_generation(
    fleet: str | None, agent_meta: str | None, crowd: bool
) -> str:
    """The generation of a recording without an EpisodeRecord, from the schemas it does carry."""
    if agent_meta is not None and not crowd:
        return LEGACY_2026_11
    if fleet is not None and fleet.startswith("task_generator_msgs/"):
        return LEGACY_2026_09
    return NATIVE


def recording_topics(recording: pathlib.Path) -> tuple[str, ...]:
    """Every topic the recording's channel records name, from its summary when it exists."""
    return tuple(_topic_schemas(recording))


def agent_meta_has_crowd(schema_text: str) -> bool:
    """Whether the recorded AgentMeta schema declares the four robot interaction settings."""
    return all(setting in schema_text for setting in _ROBOT_SETTINGS)


def episode_fields(record: object, schema_name: str) -> EpisodeFields:
    """The native field set of one EpisodeRecord generation, absent legacy fields NULL and named."""
    if schema_name == NATIVE_EPISODE_SCHEMA:
        return _native_episode(record)
    if schema_name == LEGACY_EPISODE_SCHEMA:
        return _legacy_episode(record)
    raise LegacySchemaError(
        f"episode schema {schema_name!r} is not a supported generation; "
        "the migration supports arena_runtime_msgs and task_generator_msgs EpisodeRecord"
    )


def fleet_robots(fleet: object, schema_name: str) -> tuple[object, ...]:
    """The RobotState sequence of a fleet snapshot, whichever supported schema carried it."""
    if schema_name.endswith("/RobotFleet"):
        return tuple(fleet.robots)
    if schema_name.endswith("/RobotState"):
        return (fleet,)
    if isinstance(fleet, Sequence):
        return tuple(fleet)
    raise LegacySchemaError(
        f"fleet schema {schema_name!r} carries no RobotState sequence"
    )


def legacy_modes(
    *, robots: str | None, obstacles: str | None, modules: Iterable[str] | None
) -> dict[str, list[str]]:
    """`modes` of a record made before modes: the tm_* fields it carried, an absent field leaves its axis out."""
    from arena_runtime.env.task_layer import TM_AXES

    given = dict(zip(TM_AXES, (robots, obstacles, modules), strict=True))
    return {
        axis: [name for name in ([value] if isinstance(value, str) else value) if name]
        for axis, value in given.items()
        if value is not None
    }


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


def flatten_params(
    nested: Mapping[str, typing.Any], prefix: str = ""
) -> dict[str, typing.Any]:
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
    return (
        json.dumps({axis: ",".join(names) for axis, names in modes.items()})
        if modes
        else None
    )


def params_yaml(params: Mapping[str, typing.Any]) -> str | None:
    """The recorded `params_yaml` payload, None when the generation carried no parameter."""
    return yaml.dump(dict(params)) if params else None


def _native_episode(record: object) -> EpisodeFields:
    from arena_evaluation.processing.telemetry.messages import parameter_value

    modes = {mode.key: mode.value for mode in record.modes}
    return EpisodeFields(
        episode_id=int(record.episode_id),
        world=str(record.world) if record.world else None,
        seed=int(record.seed) if record.seed >= 0 else None,
        task_layer=str(record.task_layer) or None,
        modes_json=json.dumps(modes) if modes else None,
        robots=tuple(str(robot) for robot in record.robots),
        outcome_state=int(record.outcome_state),
        outcome_info=str(record.outcome_info),
        goal_uuid=str(record.goal_uuid),
        goal_dist_start=float(record.goal_dist_start),
        goal_dist_min=float(record.goal_dist_min),
        path_length=float(record.path_length),
        start_t_ns=int(record.start_time.sec) * 1_000_000_000
        + int(record.start_time.nanosec),
        integrity=_integrity_flags(record.integrity),
        params_yaml=yaml.dump(
            {
                parameter.name: parameter_value(parameter.value)
                for parameter in record.params
            }
        )
        if record.params
        else None,
        human_params_json=json.dumps(
            {
                parameter.name: parameter_value(parameter.value)
                for parameter in record.human_params
            }
        )
        if record.human_params
        else None,
        conditions=str(record.conditions) or None,
        phases=str(record.phases) or None,
        not_recorded=(),
    )


def _integrity_flags(value: object) -> tuple[str, ...]:
    """The recorded integrity reasons; a pre-list recording stored one bool flag instead."""
    if isinstance(value, bool):
        return ("recording_integrity_flag",) if value else ()
    return tuple(str(flag) for flag in typing.cast(typing.Iterable[object], value))


def _legacy_episode(record: object) -> EpisodeFields:
    from arena_evaluation.processing.telemetry.messages import parameter_value

    modes = legacy_modes(
        robots=record.tm_robots,
        obstacles=record.tm_obstacles,
        modules=record.tm_modules,
    )
    robots_params = flatten_params(
        {
            parameter.name: parameter_value(parameter.value)
            for parameter in record.robots_params
        }
    )
    obstacles_params = flatten_params(
        {
            parameter.name: parameter_value(parameter.value)
            for parameter in record.obstacles_params
        }
    )
    params = legacy_params(modes, robots_params, obstacles_params)
    return EpisodeFields(
        episode_id=int(record.episode_id),
        world=str(record.world) if record.world else None,
        seed=int(record.seed) if record.seed >= 0 else None,
        task_layer=None,
        modes_json=modes_json(modes),
        robots=tuple(str(robot) for robot in record.robots),
        outcome_state=int(record.outcome_state),
        outcome_info=str(record.outcome_info),
        goal_uuid=str(record.goal_uuid),
        goal_dist_start=float(record.goal_dist_start),
        goal_dist_min=float(record.goal_dist_min),
        path_length=float(record.path_length),
        start_t_ns=int(record.start_time.sec) * 1_000_000_000
        + int(record.start_time.nanosec),
        integrity=_integrity_flags(record.integrity),
        params_yaml=params_yaml(params),
        human_params_json=json.dumps(
            {
                parameter.name: parameter_value(parameter.value)
                for parameter in record.human_params
            }
        )
        if record.human_params
        else None,
        conditions=str(record.conditions) or None,
        phases=str(record.phases) or None,
        not_recorded=("task_layer",),
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
        for schema, channel, _message in NonSeekingReader(handle).iter_messages(
            log_time_order=False
        ):
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
