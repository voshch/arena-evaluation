"""The migration maps an episode recorded before the native layout to current metadata and tables, and refuses what it cannot map."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from mcap_ros2.writer import Writer

from arena_evaluation.ingestion.migrate import MigrationError, migrate_episode

SEC = 1_000_000_000
SEP = "\n" + "=" * 80 + "\n"
STRING = ("std_msgs/msg/String", "string data")
TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec"
HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
VEC3 = "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z"
QUAT = "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w"
TRANSFORM = "MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation"
TRANSFORM_STAMPED = "MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform"
TF_MESSAGE = (
    "tf2_msgs/msg/TFMessage",
    SEP.join(
        [
            "geometry_msgs/TransformStamped[] transforms",
            TRANSFORM_STAMPED,
            HEADER,
            TIME,
            TRANSFORM,
            VEC3,
            QUAT,
        ]
    ),
)

_OLD_METADATA = {
    "benchmark_id": "bench",
    "planner": "dwa",
    "robot_model": ["jackal"],
    "map": "hospital",
    "stage": "stage_1",
    "recording_started_at": "2026-01-01T00:00:00",
    "python_version": "3.12",
    "ros_distro": "jazzy",
    "tm_robots": "jackal",
    "tm_obstacles": "static_crowd",
    "robots_params": {"speed": 1.0},
    "obstacles_params": {"count": 3},
}


def _write_recording(path: pathlib.Path, records: list[tuple[str, tuple[str, str], dict, int]]) -> pathlib.Path:
    with path.open("wb") as handle:
        writer = Writer(handle)
        schemas: dict[tuple[str, str], int] = {}
        for topic, (name, text), message, log_time in records:
            if (name, text) not in schemas:
                schemas[name, text] = writer.register_msgdef(name, text)
            writer.write_message(topic, schemas[name, text], message, log_time=log_time)
        writer.finish()
    return path


def _old_episode(
    tmp_path: pathlib.Path,
    records: list[tuple[str, tuple[str, str], dict, int]],
    metadata: dict | None = None,
) -> pathlib.Path:
    episode = tmp_path / "episode_000"
    episode.mkdir()
    _write_recording(episode / "episode_000.mcap", records)
    (episode / "episode_000.yaml").write_text(
        yaml.safe_dump(_OLD_METADATA if metadata is None else metadata),
        encoding="utf-8",
    )
    return episode


def _phase(topic: str, log_time: int) -> tuple[str, tuple[str, str], dict, int]:
    return (topic, STRING, {"data": "ramp_up"}, log_time)


def _transform(parent: str, child: str, log_time: int) -> tuple[str, tuple[str, str], dict, int]:
    transform = {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": parent},
        "child_frame_id": child,
        "transform": {
            "translation": {"x": 1.0, "y": 2.0, "z": 0.0},
            "rotation": {"w": 1.0},
        },
    }
    return ("/tf", TF_MESSAGE, {"transforms": [transform]}, log_time)


def test_the_environment_falls_back_to_the_transform_frames_when_no_topic_names_one(
    tmp_path: pathlib.Path,
) -> None:
    episode = _old_episode(tmp_path, [_transform("env_0/map", "env_0/jackal/base_link", SEC)])

    migrate_episode(episode)

    assert (episode / "topics" / "tf.parquet").exists()
    assert (episode / "topics" / "tf_static.parquet").exists() is False


def test_a_recording_that_names_no_environment_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    episode = _old_episode(tmp_path, [("/odom", STRING, {"data": "bare"}, SEC)])

    with pytest.raises(MigrationError, match="no topic or transform frame names an environment"):
        migrate_episode(episode)
    assert not (episode / "topics").exists()


def test_a_recording_that_names_two_environments_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    episode = _old_episode(
        tmp_path,
        [
            _phase("/env_0/jackal/characterization_phase", SEC),
            _phase("/env_1/jackal/characterization_phase", 2 * SEC),
        ],
    )

    with pytest.raises(MigrationError, match="several environments"):
        migrate_episode(episode)


def test_metadata_without_task_modes_is_refused(tmp_path: pathlib.Path) -> None:
    metadata = {key: value for key, value in _OLD_METADATA.items() if not key.startswith("tm_")}
    episode = _old_episode(tmp_path, [_phase("/env_0/jackal/characterization_phase", SEC)], metadata)

    with pytest.raises(MigrationError, match="neither modes nor the tm_\\* fields"):
        migrate_episode(episode)
    assert (episode / "episode_000.yaml").read_text(encoding="utf-8") == yaml.safe_dump(metadata)
