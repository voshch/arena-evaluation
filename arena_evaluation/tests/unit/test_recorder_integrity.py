from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import yaml
from rclpy.parameter import Parameter
from task_generator_msgs.msg import EpisodeRecord

from arena_evaluation.ingestion.recorder import DataRecorderNode
from arena_evaluation.storage.manifest import MetadataWriter


@pytest.fixture
def recorder_node(tmp_path: pathlib.Path):
    node = DataRecorderNode(parameter_overrides=[Parameter("record_data_dir", value=str(tmp_path / "episodes"))])
    yield node
    node.destroy_node()


def _record(episode_id: int, outcome_state: int, *integrity: str) -> EpisodeRecord:
    msg = EpisodeRecord()
    msg.episode_id = episode_id
    msg.outcome_state = outcome_state
    msg.integrity = list(integrity)
    return msg


def _yaml_integrity(node: DataRecorderNode) -> list[str]:
    return yaml.safe_load(node.current_metadata_path.read_text())["integrity"]


def test_recorder_yaml_follows_integrity_reasons_as_they_accrue(recorder_node: DataRecorderNode) -> None:
    recorder_node.episode_record_callback(_record(3, EpisodeRecord.RUNNING))
    assert _yaml_integrity(recorder_node) == []

    recorder_node.episode_record_callback(_record(3, EpisodeRecord.RUNNING, "robot jackal spawned by hand"))
    assert _yaml_integrity(recorder_node) == ["robot jackal spawned by hand"]

    recorder_node.episode_record_callback(_record(3, EpisodeRecord.SUCCESS, "robot jackal spawned by hand", "task parameters changed mid-episode"))
    assert _yaml_integrity(recorder_node) == ["robot jackal spawned by hand", "task parameters changed mid-episode"]
    assert MetadataWriter.read(recorder_node.current_metadata_path).integrity == ["robot jackal spawned by hand", "task parameters changed mid-episode"]


def test_recorder_keeps_integrity_reasons_arriving_after_stop(recorder_node: DataRecorderNode) -> None:
    recorder_node.episode_record_callback(_record(4, EpisodeRecord.RUNNING))
    path = recorder_node.current_metadata_path
    recorder_node._stop_episode(4, outcome_state=EpisodeRecord.FAILED, outcome_info="timeout")

    recorder_node.episode_record_callback(_record(4, EpisodeRecord.FAILED, "env_0/jackal moved by hand"))

    assert yaml.safe_load(path.read_text())["integrity"] == ["env_0/jackal moved by hand"]


def test_queued_record_of_the_next_episode_leaves_integrity_of_the_current_one(recorder_node: DataRecorderNode) -> None:
    recorder_node.episode_record_callback(_record(5, EpisodeRecord.RUNNING, "jackal: elevator teleport 'a' -> 'b' failed: TimeoutError()"))
    recorder_node.episode_record_callback(_record(6, EpisodeRecord.QUEUED))

    assert recorder_node.current_sim_episode_id == 5
    assert _yaml_integrity(recorder_node) == ["jackal: elevator teleport 'a' -> 'b' failed: TimeoutError()"]


def test_episode_yaml_without_integrity_reads_as_intact(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "episode_000.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark_id": "b",
                "planner": "dwb",
                "map": "map_empty",
                "stage": "preflight",
                "outcome_state": 2,
                "recording_started_at": "2026-01-01T00:00:00+00:00",
                "python_version": "3.12",
                "ros_distro": "jazzy",
            }
        )
    )
    assert MetadataWriter.read(path).integrity == []
