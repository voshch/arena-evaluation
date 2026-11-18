"""Migration of runs recorded before the native layout: old episode metadata and recording to the L1 layout, loudly."""

from __future__ import annotations

import pathlib
import shutil
import typing
from collections.abc import Iterable, Mapping

import yaml
from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry.messages import env_id_in
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_layer import TaskModes

_STAGED_TOPICS: typing.Final[str] = ".migrate-topics"
_TRANSFORM_TOPICS: typing.Final[tuple[str, ...]] = ("/tf", "/tf_static")


class MigrationError(RuntimeError):
    """A recording the migration cannot map; the episode is left as it was."""


def legacy_modes(
    *, robots: str | None, obstacles: str | None, modules: Iterable[str] | None
) -> TaskModes:
    """`modes` of a record made before modes: the tm_* fields it carried, an absent field leaves its axis out."""
    from arena_runtime.env.task_layer import TM_AXES

    given = dict(zip(TM_AXES, (robots, obstacles, modules), strict=True))
    return {
        axis: [name for name in ([value] if isinstance(value, str) else value) if name]
        for axis, value in given.items()
        if value is not None
    }


def legacy_params(
    modes: TaskModes,
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
        for mode in modes.get(axis, [])[:1]
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


def migrate_episode(episode_dir: pathlib.Path) -> RunMetadata:
    """Map one episode recorded before the native layout to current metadata and native tables; loud on anything unmapped."""
    metadata_path = _metadata_path(episode_dir)
    metadata = _current_metadata(metadata_path)
    recording = _recording_path(episode_dir)
    environment = environment_of(recording)
    staged = episode_dir / _STAGED_TOPICS
    if staged.exists():
        shutil.rmtree(staged)
    MCAPReader(recording, environment=environment).read(staged)
    topics = episode_dir / "topics"
    if topics.exists():
        shutil.rmtree(topics)
    staged.rename(topics)
    MetadataWriter.write(metadata, metadata_path)
    return metadata


def migrate_run(run_dir: pathlib.Path) -> list[pathlib.Path]:
    """Migrate every episode folder of a run, stopping at the first episode that cannot be mapped."""
    episode_dirs = (
        [run_dir]
        if _metadata_candidates(run_dir)
        else sorted(path for path in run_dir.glob("episode_*") if path.is_dir())
    )
    if not episode_dirs:
        raise MigrationError(f"{run_dir}: no episode folder to migrate")
    migrated: list[pathlib.Path] = []
    for episode_dir in episode_dirs:
        migrate_episode(episode_dir)
        migrated.append(episode_dir)
    return migrated


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

    if "modes" not in raw and not any(f"tm_{axis}" in raw for axis in TM_AXES):
        raise MigrationError(
            f"{metadata_path}: records neither modes nor the tm_* fields"
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


def _metadata_candidates(episode_dir: pathlib.Path) -> list[pathlib.Path]:
    return (
        sorted(path for path in episode_dir.glob("*.yaml") if path.is_file())
        if episode_dir.is_dir()
        else []
    )


def _metadata_path(episode_dir: pathlib.Path) -> pathlib.Path:
    candidates = _metadata_candidates(episode_dir)
    named = [path for path in candidates if path.stem == episode_dir.name]
    chosen = named or candidates
    if len(chosen) != 1:
        raise MigrationError(
            f"{episode_dir}: expected one episode metadata file, found {len(chosen)}"
        )
    return chosen[0]


def _recording_path(episode_dir: pathlib.Path) -> pathlib.Path:
    recordings = sorted(episode_dir.glob("*.mcap"))
    if len(recordings) != 1:
        raise MigrationError(
            f"{episode_dir}: expected one recording, found {len(recordings)}"
        )
    return recordings[0]


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
