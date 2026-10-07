from __future__ import annotations

from pathlib import Path

import pytest
import yaml

EPISODE = "episode_000"
TOPIC = "/arena/env_0/task_generator_node/jackal/cmd_vel"
MOVED = ("_enter_episode_dir", "_open_writer", "_close_current_writer", "_write_serialized", "_register_topic", "_ensure_topic_in_bag", "_finalize_episode_metadata")


def metadata(episode_id: int = 0):
    from arena_evaluation.ingestion.metadata import IngestionMetadata

    return IngestionMetadata.create_episode_metadata(benchmark_id="run", planner="teb", stage="stage", map_name="unknown", episode_id=episode_id, robot_model="jackal")


def record(writer, directory: Path, speeds: list[float]):
    from geometry_msgs.msg import Twist
    from rclpy.serialization import serialize_message

    meta = metadata()
    writer.open_episode(directory, meta)
    for n, speed in enumerate(speeds):
        msg = Twist()
        msg.linear.x = speed
        writer.write_message(TOPIC, Twist, serialize_message(msg), n * 100_000_000)
    writer.close_episode()
    return meta


@pytest.fixture
def writer():
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("rclpy")
    from arena_evaluation.ingestion.episode_writer import EpisodeWriter

    return EpisodeWriter()


def test_episode_is_flattened_beside_its_finalized_yaml(writer, tmp_path: Path):
    directory = tmp_path / "run" / "episodes" / EPISODE
    meta = record(writer, directory, [0.1, 0.2, 0.3])
    assert sorted(p.name for p in directory.iterdir()) == [f"{EPISODE}.mcap", f"{EPISODE}.yaml"]
    with open(directory / f"{EPISODE}.yaml") as f:
        stored = yaml.safe_load(f)
    assert stored["rosbag2_message_count"] == 3
    assert stored["recorded_topics"] == [TOPIC.strip("/")]
    assert stored["pedsim_available"] is False
    assert stored["recording_ended_at"] == meta.recording_ended_at
    (entry,) = stored["rosbag2_topics"]
    assert (entry["topic_metadata"]["name"], entry["topic_metadata"]["type"], entry["message_count"]) == (TOPIC.strip("/"), "geometry_msgs/msg/Twist", 3)
    assert writer.writer is None


def test_processing_reads_what_the_writer_wrote(writer, tmp_path: Path):
    from arena_evaluation.processing.mcap_reader import MCAPReader

    directory = tmp_path / "run" / "episodes" / EPISODE
    record(writer, directory, [0.1, 0.2, 0.3])
    bundles = MCAPReader(directory / f"{EPISODE}.mcap").read(tmp_path / "topics")
    command = bundles["env_0_jackal"].cmd_vel.collect()
    assert command["time_ns"].to_list() == [0, 100_000_000, 200_000_000]
    assert command["linear_x"].to_list() == pytest.approx([0.1, 0.2, 0.3])


def test_second_episode_starts_from_clean_counts(writer, tmp_path: Path):
    first = record(writer, tmp_path / "run" / "episodes" / EPISODE, [0.1, 0.2, 0.3])
    second = record(writer, tmp_path / "run" / "episodes" / "episode_001", [0.4])
    assert (first.rosbag2_message_count, second.rosbag2_message_count) == (3, 1)
    assert (tmp_path / "run" / "episodes" / "episode_001" / "episode_001.mcap").is_file()


def test_open_episode_raises_when_the_bag_cannot_open(tmp_path: Path):
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("rclpy")
    from arena_evaluation.ingestion.episode_writer import EpisodeWriter

    (tmp_path / "share" / "config").mkdir(parents=True)
    (tmp_path / "share" / "config" / "mcap_writer_options.yaml").write_text("compression: [unclosed\n")
    with pytest.raises(RuntimeError, match="no MCAP writer"):
        EpisodeWriter(base_dir=str(tmp_path / "share")).open_episode(tmp_path / "run" / "episodes" / EPISODE, metadata())


def test_bag_an_earlier_writer_left_unclosed_is_kept_aside(writer, tmp_path: Path):
    directory = tmp_path / "run" / "episodes" / EPISODE
    (directory / EPISODE).mkdir(parents=True)
    (directory / EPISODE / "stale.mcap").write_bytes(b"stale")
    record(writer, directory, [0.1])
    assert sorted(p.name for p in directory.iterdir()) == [f"{EPISODE}.mcap", f"{EPISODE}.yaml", f"{EPISODE}_unclosed_0"]
    assert (directory / f"{EPISODE}_unclosed_0" / "stale.mcap").read_bytes() == b"stale"


def test_recorder_node_writes_through_the_same_methods():
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("rclpy")
    from arena_evaluation.ingestion.episode_writer import EpisodeWriter
    from arena_evaluation.ingestion.recorder import DataRecorderNode

    for name in MOVED:
        assert name not in vars(DataRecorderNode)
        assert getattr(DataRecorderNode, name) is getattr(EpisodeWriter, name)
