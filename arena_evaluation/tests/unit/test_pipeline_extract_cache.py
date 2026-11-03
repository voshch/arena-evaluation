from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import ODOMETRY, _header, _write_bag

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import EpisodeDescriptor

SEC = 1_000_000_000


def _episode(tmp_path: pathlib.Path) -> tuple[ProcessingPipeline, EpisodeDescriptor, pathlib.Path]:
    episode_dir = tmp_path / "bench" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    odom = {"header": _header(1, 0, "odom"), "pose": {"pose": {"position": {"x": 1.0, "y": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {"linear": {"x": 0.5}, "angular": {"z": 0.1}}}}
    bag = _write_bag(episode_dir / "episode_000.mcap", [("/arena/env_0/arena_env_node/jackal/odom", ODOMETRY, odom, 1 * SEC)])
    descriptor = EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="bench", episode_id=0, planner="dwb", stage="s", map="barn_0")
    return ProcessingPipeline(FolderManager(tmp_path), workers=1), descriptor, bag


def test_extract_episode_reextracts_a_cache_written_by_an_older_layout(tmp_path: pathlib.Path) -> None:
    pipeline, episode, _ = _episode(tmp_path)
    topics = pathlib.Path(episode.episode_dir) / "topics"
    (topics / "env_0_jackal").mkdir(parents=True)
    pl.DataFrame({"time_ns": [1], "pos_x": [999.0]}).write_parquet(topics / "env_0_jackal" / "odom.parquet")
    assert not TopicParquetStore.is_current(topics)

    bundles = pipeline.extract_episode(episode)

    assert bundles is not None
    assert bundles["env_0_jackal"].odom.collect()["pos_x"].to_list() == [1.0]
    assert TopicParquetStore.is_current(topics)


def test_extract_episode_reuses_a_cache_written_by_this_layout(tmp_path: pathlib.Path) -> None:
    pipeline, episode, bag = _episode(tmp_path)
    first = pipeline.extract_episode(episode)
    bag.write_bytes(b"not an mcap")

    second = pipeline.extract_episode(episode)

    assert first is not None and second is not None
    assert second["env_0_jackal"].odom.collect().equals(first["env_0_jackal"].odom.collect())
