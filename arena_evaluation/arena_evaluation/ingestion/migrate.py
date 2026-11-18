"""Migration of runs recorded before the native layout: generations, episode metadata, native tables, descriptors."""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import typing
from collections.abc import Iterable, Mapping, Sequence

import polars as pl
import yaml
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

from arena_evaluation.processing import legacy_schemas
from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.telemetry.messages import env_id_in, topic_parts
from arena_evaluation.storage.atomic import staged_temp_path
from arena_evaluation.storage.layout import retire_legacy_outputs
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.registry.telemetry import RobotDescriptor, RobotParamRows
from arena_evaluation.storage.schemas import RunMetadata

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_layer import TaskModes

_STAGED_TOPICS: typing.Final[str] = ".migrate-topics"
_TRANSFORM_TOPICS: typing.Final[tuple[str, ...]] = ("/tf", "/tf_static")
_REPORT_NAME: typing.Final[str] = "migration_report.yaml"
_BENCHMARK_YAMLS: typing.Final[frozenset[str]] = frozenset(
    {
        "manifest.yaml",
        "report_manifest.yaml",
        _REPORT_NAME,
        "benchmark_layout.yaml",
        "contract_report.yaml",
    }
)
_ENV_LEVEL_SEGMENTS: typing.Final[frozenset[str]] = frozenset(
    {
        "state",
        "arena_peds",
        "peds",
        "map",
        "door_mask",
        "tf",
        "initialpose",
        "agent_states",
        "agent_meta",
        "interactions",
        "humans",
        "acoustics",
        "audio",
        "heard_sound_events",
        "continuous_heard_sounds",
    }
)


class MigrationError(RuntimeError):
    """A recording the migration cannot map; the episode is left as it was unless a report entry says otherwise."""


@dataclasses.dataclass(frozen=True, slots=True)
class EpisodeMigration:
    """One episode's migration result, generation and the fields the generation did not carry."""

    episode: pathlib.Path
    generation: str
    mapped: tuple[str, ...]
    fallbacks: tuple[str, ...]
    unsupported: tuple[str, ...]
    migrated: bool
    error: str | None = None
    metadata: RunMetadata | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class MigrationReport:
    """The per-episode migration report of one benchmark, written to migration_report.yaml."""

    benchmark_dir: pathlib.Path
    episodes: tuple[EpisodeMigration, ...]
    dry_run: bool
    retired: tuple[pathlib.Path, ...] = ()

    @property
    def failures(self) -> tuple[EpisodeMigration, ...]:
        """The episodes that failed, the benchmark keeps migrating the rest."""
        return tuple(episode for episode in self.episodes if episode.error is not None)

    def failure_message(self) -> str:
        """One line per failed episode, for the caller's loud exit."""
        lines = [
            f"{episode.episode.name}: {episode.error}" for episode in self.failures
        ]
        return f"{len(lines)} episode(s) failed to migrate:\n" + "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        """The report as plain data for the YAML artifact."""
        return {
            "benchmark_dir": str(self.benchmark_dir),
            "dry_run": self.dry_run,
            "retired": [str(path) for path in self.retired],
            "episodes": [
                {
                    "episode": episode.episode.name,
                    "generation": episode.generation,
                    "migrated": episode.migrated,
                    "mapped": list(episode.mapped),
                    "fallbacks": list(episode.fallbacks),
                    "unsupported": list(episode.unsupported),
                    "error": episode.error,
                }
                for episode in self.episodes
            ],
        }


def legacy_modes(
    *, robots: str | None, obstacles: str | None, modules: Iterable[str] | None
) -> TaskModes:
    """`modes` of a record made before modes: the tm_* fields it carried, an absent field leaves its axis out."""
    return typing.cast(
        "TaskModes",
        legacy_schemas.legacy_modes(
            robots=robots, obstacles=obstacles, modules=modules
        ),
    )


def legacy_params(
    modes: TaskModes,
    robots_params: Mapping[str, typing.Any],
    obstacles_params: Mapping[str, typing.Any],
) -> dict[str, typing.Any]:
    """Full-name `params` of a record made before modes: a leaf param of the robots or obstacles mode `m` was `task.<m>.<leaf>`."""
    return legacy_schemas.legacy_params(modes, robots_params, obstacles_params)


def flatten_params(
    nested: Mapping[str, typing.Any], prefix: str = ""
) -> dict[str, typing.Any]:
    """Dotted leaf names of a nested params mapping."""
    return legacy_schemas.flatten_params(nested, prefix)


def environment_of(recording: pathlib.Path) -> str:
    """The one environment a recording names: its topic tokens first, its transform frames second; loud when none or several."""
    topic_environments = _topic_environments(recording)
    if len(topic_environments) == 1:
        return topic_environments.pop()
    if len(topic_environments) > 1:
        raise MigrationError(
            f"{recording}: its topics name several environments {sorted(topic_environments)}"
        )
    frame_environments = _frame_environments(recording)
    if len(frame_environments) == 1:
        return frame_environments.pop()
    if not frame_environments:
        raise MigrationError(
            f"{recording}: no topic or transform frame names an environment"
        )
    raise MigrationError(
        f"{recording}: its transform frames name several environments {sorted(frame_environments)}"
    )


def migrate_episode(episode_dir: pathlib.Path, *, force: bool = False) -> RunMetadata:
    """Map one episode to current metadata and native tables; loud on anything unmapped."""
    result = _migrate_episode(episode_dir, force=force, dry_run=False)
    assert result.metadata is not None
    return result.metadata


def migrate_run(
    run_dir: pathlib.Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    report_path: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Migrate every episode of a benchmark or one episode directory; a failure is reported and the rest continue."""
    report = migrate_benchmark(
        run_dir, dry_run=dry_run, force=force, report_path=report_path
    )
    if report.failures:
        raise MigrationError(report.failure_message())
    return [episode.episode for episode in report.episodes if episode.migrated]


def migrate_benchmark(
    run_dir: pathlib.Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    report_path: pathlib.Path | None = None,
) -> MigrationReport:
    """Migrate a benchmark root by its episodes/ layout, or one episode directory, and write the report."""
    root = pathlib.Path(run_dir)
    episode_dirs = _episode_directories(root)
    if not episode_dirs:
        raise MigrationError(
            f"{root}: no episode to migrate; expected {root}/episodes/episode_* or one episode directory "
            "holding its recording and its episode YAML"
        )
    retired = () if dry_run else retire_legacy_outputs(root)
    results: list[EpisodeMigration] = []
    for episode_dir in episode_dirs:
        try:
            results.append(_migrate_episode(episode_dir, force=force, dry_run=dry_run))
        except MigrationError as error:
            results.append(
                EpisodeMigration(
                    episode=episode_dir,
                    generation=legacy_schemas.UNKNOWN,
                    mapped=(),
                    fallbacks=(),
                    unsupported=(),
                    migrated=False,
                    error=str(error),
                )
            )
    report = MigrationReport(
        benchmark_dir=root,
        episodes=tuple(results),
        dry_run=dry_run,
        retired=tuple(retired),
    )
    destination = report_path
    if destination is None and not dry_run:
        destination = root / _REPORT_NAME
    if destination is not None:
        _write_report(report, pathlib.Path(destination))
    return report


def _migrate_episode(
    episode_dir: pathlib.Path, *, force: bool, dry_run: bool
) -> EpisodeMigration:
    metadata_path = _metadata_path(episode_dir)
    metadata = _current_metadata(metadata_path)
    recording = _recording_path(episode_dir)
    generation = legacy_schemas.detect(recording)
    if generation.name == legacy_schemas.UNKNOWN:
        raise MigrationError(
            f"{recording}: unsupported recording generation: "
            + "; ".join(generation.unsupported)
        )
    mapped = _mapped_names(metadata_path)
    topics = episode_dir / "topics"
    fallbacks: list[str] = []
    needs_extraction = force or not TopicParquetStore.is_current(topics)
    if dry_run:
        if generation.fleet_schema is None and not _descriptor_present(topics):
            for _robot, model in _descriptor_assignment(
                recording, metadata, environment_of(recording)
            ).items():
                _installed_parameters(model)
        return EpisodeMigration(
            episode=episode_dir,
            generation=generation.name,
            mapped=mapped,
            fallbacks=("re-extract the recording",) if needs_extraction else (),
            unsupported=generation.unsupported,
            migrated=False,
        )
    if needs_extraction:
        environment = environment_of(recording)
        staged = episode_dir / _STAGED_TOPICS
        if staged.exists():
            shutil.rmtree(staged)
        MCAPReader(recording, environment=environment).read(staged)
        if topics.exists():
            shutil.rmtree(topics)
        staged.rename(topics)
        TopicParquetStore.mark_complete(topics)
        fallbacks.append("re-extracted the recording into topics/")
    synthesized: tuple[str, ...] = ()
    if generation.fleet_schema is None:
        synthesized = _synthesize_descriptor(topics, recording, metadata)
        if synthesized:
            fallbacks.append(
                "synthesised robot_descriptor/robot_params from RunMetadata.robot_model and "
                "the installed arena_robots definition for " + ", ".join(synthesized)
            )
    changed = needs_extraction or bool(synthesized) or bool(mapped)
    if changed:
        MetadataWriter.write(metadata, metadata_path)
    return EpisodeMigration(
        episode=episode_dir,
        generation=generation.name,
        mapped=mapped,
        fallbacks=tuple(fallbacks),
        unsupported=generation.unsupported,
        migrated=changed,
        metadata=metadata,
    )


def _current_metadata(metadata_path: pathlib.Path) -> RunMetadata:
    raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MigrationError(f"{metadata_path}: the metadata is not a mapping")
    data = _mapped_fields(raw, metadata_path)
    try:
        return RunMetadata.model_validate(data)
    except ValueError as error:
        raise MigrationError(
            f"{metadata_path}: the mapped metadata is incomplete: {error}"
        ) from error


def _mapped_fields(
    raw: dict[str, typing.Any], metadata_path: pathlib.Path
) -> dict[str, typing.Any]:
    from arena_runtime.env.task_layer import TM_AXES

    legacy = any(f"tm_{axis}" in raw for axis in TM_AXES)
    if "modes" not in raw and not legacy:
        raise MigrationError(
            f"{metadata_path}: records neither modes nor the tm_* fields; "
            "this metadata predates both the migration and the native layout"
        )
    data = dict(raw)
    robots, obstacles, modules = (data.pop(f"tm_{axis}", None) for axis in TM_AXES)
    robots_params = flatten_params(data.pop("robots_params", None) or {})
    obstacles_params = flatten_params(data.pop("obstacles_params", None) or {})
    modes = data.get("modes") or legacy_modes(
        robots=robots, obstacles=obstacles, modules=modules
    )
    data["modes"] = modes
    data["params"] = data.get("params") or legacy_params(
        modes, robots_params, obstacles_params
    )
    return data


def _mapped_names(metadata_path: pathlib.Path) -> tuple[str, ...]:
    """The native fields the original metadata had to be mapped from tm_*."""
    from arena_runtime.env.task_layer import TM_AXES

    raw = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ()
    return ("modes", "params") if any(f"tm_{axis}" in raw for axis in TM_AXES) else ()


def _episode_directories(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The episode directories a benchmark root holds, by its episodes/ layout or as one episode."""
    if (root / "episodes").is_dir():
        candidates = sorted(
            path for path in (root / "episodes").iterdir() if path.is_dir()
        )
    else:
        candidates = [root]
    return tuple(path for path in candidates if _looks_like_episode(path))


def _looks_like_episode(directory: pathlib.Path) -> bool:
    """An episode directory holds one recording and one non-benchmark YAML."""
    return bool(list(directory.glob("*.mcap"))) and bool(
        _metadata_candidates(directory)
    )


def _metadata_candidates(episode_dir: pathlib.Path) -> list[pathlib.Path]:
    return (
        sorted(
            path
            for path in episode_dir.glob("*.yaml")
            if path.is_file() and path.name not in _BENCHMARK_YAMLS
        )
        if episode_dir.is_dir()
        else []
    )


def _metadata_path(episode_dir: pathlib.Path) -> pathlib.Path:
    candidates = _metadata_candidates(episode_dir)
    if not candidates:
        raise MigrationError(
            f"{episode_dir}: no episode metadata YAML; expected {episode_dir.name}.yaml "
            "or metadata.yaml"
        )
    named = [path for path in candidates if path.stem == episode_dir.name]
    chosen = named or candidates
    if len(chosen) != 1:
        raise MigrationError(
            f"{episode_dir}: expected one episode metadata file, found "
            f"{[path.name for path in chosen]}"
        )
    return chosen[0]


def _recording_path(episode_dir: pathlib.Path) -> pathlib.Path:
    recordings = sorted(episode_dir.glob("*.mcap"))
    if len(recordings) != 1:
        raise MigrationError(
            f"{episode_dir}: expected one recording, found {len(recordings)}"
        )
    return recordings[0]


def _descriptor_present(topics: pathlib.Path) -> bool:
    path = topics / "robot_descriptor.parquet"
    return path.is_file() and pl.read_parquet(path).height > 0


def _descriptor_assignment(
    recording: pathlib.Path, metadata: RunMetadata, environment: str
) -> dict[str, str]:
    """Robot name -> model for a fleet-less recording, from its robot topics and the recorded model list."""
    names = _robot_names(recording, environment)
    models = list(metadata.robot_model)
    if not models:
        raise MigrationError(
            f"{recording}: no /state/robots fleet and RunMetadata.robot_model is empty; "
            "the robot descriptor cannot be synthesised (record a fleet message or the model list)"
        )
    if not names:
        if len(models) == 1:
            # A single-robot recording with no fleet and no robot topic name: the recorded model is
            # the only robot identity there is, so the robot keeps the model's base name.
            base = models[0].partition("[")[0]
            return {base: models[0]}
        raise MigrationError(
            f"{recording}: no /state/robots fleet and no robot topic names a robot; "
            f"the recorded models {models} cannot be assigned to robots"
        )
    if len(models) == 1:
        return {name: models[0] for name in names}
    if len(names) == len(models):
        return dict(zip(names, models, strict=True))
    raise MigrationError(
        f"{recording}: {len(models)} recorded robot models {models} do not match "
        f"the {len(names)} robot topic names {list(names)}"
    )


def _synthesize_descriptor(
    topics: pathlib.Path, recording: pathlib.Path, metadata: RunMetadata
) -> tuple[str, ...]:
    """Write robot_descriptor/robot_params from the metadata model and the installed definition, once."""
    if _descriptor_present(topics):
        return ()
    environment = environment_of(recording)
    assignment = _descriptor_assignment(recording, metadata, environment)
    descriptor_rows: list[dict[str, typing.Any]] = []
    parameter_rows: list[dict[str, typing.Any]] = []
    for name, model in assignment.items():
        parameters = _installed_parameters(model)
        robot_id = f"{environment}_{name}"
        descriptor_rows.append(
            {
                "t_ns": 0,
                "env_id": environment,
                "robot_id": robot_id,
                "name": name,
                "model": model,
                "ns": f"/{environment}/{name}",
                "frame": "world",
                "base_frame": f"{environment}/{name}/base_link",
            }
        )
        parameter_rows.extend(
            {
                "t_ns": 0,
                "env_id": environment,
                "robot_id": robot_id,
                "param_key": key,
                "param_value": value,
            }
            for key, value in parameters.items()
        )
    _write_rows(topics / "robot_descriptor.parquet", descriptor_rows, RobotDescriptor)
    _write_rows(topics / "robot_params.parquet", parameter_rows, RobotParamRows)
    return tuple(f"{name}={model}" for name, model in assignment.items())


def _installed_parameters(model: str) -> dict[str, str]:
    """The recorded parameters of one model from the installed arena_robots definition; a missing one is loud."""
    from ament_index_python.packages import get_package_share_directory

    base = model.partition("[")[0]
    try:
        share = pathlib.Path(get_package_share_directory("arena_robots"))
    except (ImportError, OSError, KeyError) as error:
        raise MigrationError(
            f"arena_robots is not installed, so the parameters of {model!r} cannot be resolved; "
            "record a /state/robots fleet in the episode"
        ) from error
    root = share / "robots" / base
    radius = _yaml_value(root / "caps" / "mobile.yaml", ("radius",))
    mass = _yaml_value(root / "model_params.yaml", ("mass", "base_kg"))
    missing = [
        name
        for name, value in (("radius", radius), ("mass.base_kg", mass))
        if value is None
    ]
    if missing:
        raise MigrationError(
            f"the installed arena_robots definition of {model!r} lacks {missing}; "
            "the descriptor cannot be synthesised with an invented value"
        )
    parameters = {"robot_radius": str(radius), "robot_mass_kg": str(mass)}
    angular = _yaml_value(root / "caps" / "mobile.yaml", ("velocity_limits", "angular"))
    if isinstance(angular, list) and angular:
        parameters["max_angular_vel"] = str(max(abs(float(value)) for value in angular))
    return parameters


def _yaml_value(path: pathlib.Path, keys: Sequence[str]) -> typing.Any | None:
    """One value of a nested YAML mapping, None when the file or any key is absent."""
    try:
        value: typing.Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _robot_names(recording: pathlib.Path, environment: str) -> tuple[str, ...]:
    """The robot names the recording's topics carry, never an environment-level topic segment."""
    names: set[str] = set()
    for topic in legacy_schemas.recording_topics(recording):
        if env_id_in(topic) != environment:
            continue
        parts = topic_parts(topic)
        if len(parts) < 3 or parts[0] != environment:
            continue
        name = parts[1]
        if name in _ENV_LEVEL_SEGMENTS:
            continue
        names.add(name)
    return tuple(sorted(names))


def _write_rows(
    path: pathlib.Path, rows: Iterable[Mapping[str, typing.Any]], table: type
) -> None:
    """Write one telemetry table with its declared schema in one rename."""
    schema = table.polars_schema()
    frame = pl.DataFrame(list(rows)).select(list(schema.names()))
    frame = frame.cast({name: dtype for name, dtype in schema.items()})
    with staged_temp_path(path) as temporary:
        frame.write_parquet(temporary, compression="zstd")


def _write_report(report: MigrationReport, path: pathlib.Path) -> None:
    """Write the YAML migration report in one rename."""
    with staged_temp_path(path) as temporary:
        temporary.write_text(
            yaml.safe_dump(report.as_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _topic_environments(recording: pathlib.Path) -> set[str]:
    environments: set[str] = set()
    with recording.open("rb") as handle:
        for _schema, channel, _message in NonSeekingReader(handle).iter_messages(
            log_time_order=False
        ):
            token = env_id_in(channel.topic)
            if token is not None:
                environments.add(token)
    return environments


def _frame_environments(recording: pathlib.Path) -> set[str]:
    factory = DecoderFactory()
    environments: set[str] = set()
    with recording.open("rb") as handle:
        for schema, channel, message in NonSeekingReader(handle).iter_messages(
            log_time_order=False
        ):
            if schema is None or not _is_transform_topic(channel.topic):
                continue
            decoder = factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise MigrationError(
                    f"{recording}: transform schema {schema.name!r} cannot be decoded"
                )
            for transform in decoder(message.data).transforms:
                for frame in (transform.header.frame_id, transform.child_frame_id):
                    token = env_id_in(frame)
                    if token is not None:
                        environments.add(token)
    return environments


def _is_transform_topic(topic: str) -> bool:
    return topic in _TRANSFORM_TOPICS or topic.endswith(_TRANSFORM_TOPICS)
