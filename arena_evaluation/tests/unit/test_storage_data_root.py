"""Unit tests for arena_evaluation.storage.data_root.

Covers $ARENA_DATA_DIR resolution in benchmarks_root() and the newest-by-``created_at``
run lookup in latest_benchmark(), addressed by folder name (08 H1.7/H1.8).
"""

from __future__ import annotations

import pathlib

import yaml

from arena_evaluation.storage.data_root import latest_benchmark


# benchmarks_root


# latest_benchmark


def _write_manifest(
    run_dir: pathlib.Path, *, created_at: str, run_id: str | None = None
) -> pathlib.Path:
    """Create a run folder with the manifest the runner writes."""
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump({"run_id": run_id or run_dir.name, "created_at": created_at}),
        encoding="utf-8",
    )
    return run_dir


def test_latest_benchmark_missing_root(tmp_path: pathlib.Path):
    assert latest_benchmark(tmp_path / "does_not_exist") is None


def test_latest_benchmark_ignores_files_and_stray_folders(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "notes.md").write_text("hello")
    (root / "not_a_run").mkdir()
    assert latest_benchmark(root) is None


def test_latest_benchmark_skips_a_stray_folder_without_manifest(tmp_path: pathlib.Path):
    """Three run folders and one stray folder: the stray one is never the default."""
    root = tmp_path / "benchmarks"
    root.mkdir()
    _write_manifest(root / "run_a", created_at="2026-01-01T00:00:00+00:00")
    newest = _write_manifest(root / "run_b", created_at="2026-02-01T00:00:00+00:00")
    _write_manifest(root / "run_c", created_at="2026-01-15T00:00:00+00:00")
    (root / "stray").mkdir()
    assert latest_benchmark(root) == newest


def test_latest_benchmark_reads_an_unquoted_yaml_timestamp(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    older = root / "older"
    older.mkdir()
    (older / "manifest.yaml").write_text(
        "run_id: older\ncreated_at: 2026-01-01T00:00:00+00:00\n", encoding="utf-8"
    )
    newer = root / "newer"
    newer.mkdir()
    (newer / "manifest.yaml").write_text(
        "run_id: newer\ncreated_at: 2026-03-01T00:00:00+00:00\n", encoding="utf-8"
    )
    assert latest_benchmark(root) == newer


def test_latest_benchmark_ignores_an_unreadable_manifest(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    broken = root / "broken"
    broken.mkdir()
    (broken / "manifest.yaml").write_text(": not a mapping [", encoding="utf-8")
    good = _write_manifest(root / "good", created_at="2026-01-01T00:00:00+00:00")
    assert latest_benchmark(root) == good


def test_latest_benchmark_default_root_uses_benchmarks_root(tmp_path: pathlib.Path):
    fake_root = tmp_path / "benchmarks"
    fake_root.mkdir()
    _write_manifest(fake_root / "run_a", created_at="2026-01-01T00:00:00+00:00")
    newest = _write_manifest(
        fake_root / "run_b", created_at="2026-02-01T00:00:00+00:00"
    )
    assert latest_benchmark(environ={"ARENA_DATA_DIR": str(tmp_path)}) == newest
