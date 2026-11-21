"""Stale per-episode outputs are refused or retired, and force reaches the skip guard."""

from __future__ import annotations

import pathlib
import shutil
import time

import polars as pl
import pytest

from arena_evaluation.processing.pipeline import (
    ProcessingPipeline,
)
from arena_evaluation.storage.exceptions import StaleBenchmarkError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.layout import write_episode_stamp
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata


def _metadata(benchmark_id: str, episode_id: int, world: str) -> RunMetadata:
    """The runner metadata a discoverable episode carries."""
    return RunMetadata(
        benchmark_id=benchmark_id,
        episode_id=episode_id,
        planner="nav2",
        local_planner="dwb",
        inter_planner="",
        stage="s0",
        map=world,
        outcome_state=2,
        outcome_info="finished",
        recording_started_at="2026-01-01T00:00:00Z",
        python_version="3.12",
        ros_distro="jazzy",
        modes={},
        params={},
    )


def _descriptor(episode_dir: pathlib.Path, benchmark_id: str) -> EpisodeDescriptor:
    """The descriptor of one episode directory under a benchmark."""
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id=benchmark_id,
        episode_id=0,
        planner="nav2",
        stage="s0",
        map="barn_0",
    )


def _old_fact_file(metrics_dir: pathlib.Path) -> pathlib.Path:
    """A 23-column fact_episodes.parquet of the pre-native layout, without a completion stamp."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "fact_episodes.parquet"
    pl.DataFrame({f"old_{index}": [index] for index in range(23)}).write_parquet(path)
    return path


def _legacy_benchmark(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """A benchmark whose only fact file has the old 23-column schema and no episode stamp."""
    benchmark_dir = root / "bench"
    episode_dir = benchmark_dir / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    MetadataWriter.write(_metadata("bench", 0, "barn_0"), episode_dir / "episode_000.yaml")
    return benchmark_dir, _old_fact_file(episode_dir / "metrics")


def _stamped_episode(
    root: pathlib.Path,
) -> tuple[ProcessingPipeline, EpisodeDescriptor, pathlib.Path]:
    """A pipeline over a benchmark whose only episode is complete and stamped."""
    episode_dir = root / "bench" / "episodes" / "episode_000"
    metrics_dir = episode_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    pl.DataFrame({"run_id": ["bench"], "episode_id": [0], "robot_id": ["env_0_jackal_0"]}).write_parquet(metrics_dir / "fact_episodes.parquet")
    write_episode_stamp(metrics_dir)
    return (
        ProcessingPipeline(FolderManager(root), workers=1),
        _descriptor(episode_dir, "bench"),
        metrics_dir,
    )


def test_process_benchmark_refuses_a_stale_fact_file(tmp_path: pathlib.Path) -> None:
    benchmark_dir, _ = _legacy_benchmark(tmp_path)
    pipeline = ProcessingPipeline(FolderManager(tmp_path), workers=1)

    with pytest.raises(StaleBenchmarkError):
        pipeline.process_benchmark(benchmark_dir.name)


def test_force_process_retires_the_stale_fact_file_and_processes(
    tmp_path: pathlib.Path,
) -> None:
    benchmark_dir, stale = _legacy_benchmark(tmp_path)
    pipeline = ProcessingPipeline(FolderManager(tmp_path), workers=1)

    written = pipeline.process_benchmark(benchmark_dir.name, force_process=True)

    assert written == benchmark_dir
    assert not stale.exists()
    moved = tuple((benchmark_dir / "legacy").rglob("fact_episodes.parquet"))
    assert len(moved) == 1


def test_evaluate_episode_skips_only_a_complete_episode(
    tmp_path: pathlib.Path,
) -> None:
    pipeline, complete, _ = _stamped_episode(tmp_path / "complete")
    assert pipeline.evaluate_episode(complete).rows == []

    incomplete_dir = tmp_path / "incomplete" / "bench" / "episodes" / "episode_001"
    metrics_dir = incomplete_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    pl.DataFrame({"value": [1]}).write_parquet(metrics_dir / "fact_episodes.parquet")
    incomplete = ProcessingPipeline(FolderManager(tmp_path / "incomplete"), workers=1)
    evaluated = incomplete.evaluate_episode(_descriptor(incomplete_dir, "bench"))

    assert evaluated.rows
    assert evaluated.rows[0]["status"] == "no_recording"


def test_process_benchmark_force_process_recomputes_a_complete_episode(
    tmp_path: pathlib.Path,
) -> None:
    shapely = pytest.importorskip("shapely")

    from arena_evaluation.processing.metrics.common.caching import world_cache
    from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
        _ZoneGeometry,
    )
    from arena_evaluation.processing.metrics.ecological.condition_metrics import (
        _wall_cache,
    )
    from test_pipeline_process_episode import WORLD, _episode

    world_cache[WORLD] = [
        _ZoneGeometry(
            name="kitchen",
            polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]),
            max_speed=None,
            quiet=False,
            restricted=False,
        )
    ]
    _wall_cache[WORLD] = None
    folders, descriptor = _episode(tmp_path)
    episode_dir = folders.episode_dir("run", 0)
    shutil.copytree(pathlib.Path(descriptor.episode_dir), episode_dir)
    MetadataWriter.write(_metadata("run", 0, WORLD), episode_dir / "episode_000.yaml")
    pipeline = ProcessingPipeline(folders, workers=1)

    assert pipeline.process_benchmark("run") == tmp_path / "run"
    facts = episode_dir / "metrics" / "fact_episodes.parquet"
    before = facts.stat().st_mtime_ns
    time.sleep(0.05)

    assert pipeline.process_benchmark("run", force_process=True) == tmp_path / "run"
    assert facts.stat().st_mtime_ns > before
