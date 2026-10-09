import pathlib
import shutil
import subprocess
import sys

import polars as pl

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "sample_benchmark" / "episodes" / "episode_000"


def _episode(tmp_path: pathlib.Path) -> pathlib.Path:
    episode = tmp_path / "sample_benchmark" / "episodes" / "episode_000"
    shutil.copytree(FIXTURE, episode)
    return episode


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "arena_evaluation.cli", *args], capture_output=True, text=True, timeout=600, check=False)


def test_extract_run_dir_writes_topic_parquet(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path)
    result = _cli("extract", "--run-dir", str(episode))
    assert result.returncode == 0, result.stdout + result.stderr
    assert any((episode / "topics").rglob("*.parquet"))


def test_process_run_dir_writes_the_episode_metrics(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path)
    result = _cli("process", "--run-dir", str(episode))
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Metrics written to: {episode / 'metrics.parquet'}" in result.stdout
    metrics = pl.read_parquet(episode / "metrics.parquet")
    assert metrics.height == 1
    assert metrics["planner"][0] == "dwb"


def test_run_run_dir_processes_and_reports(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path)
    out = tmp_path / "report"
    result = _cli("run", "--run-dir", str(episode), "--output-dir", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (episode / "metrics.parquet").exists()
    assert "Report generation complete." in result.stdout
    assert any(out.iterdir())
