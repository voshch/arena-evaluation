"""Unit tests for arena_evaluation.storage.data_root.

Covers $ARENA_DATA_DIR resolution in benchmarks_root() and the
lexicographically-latest benchmark lookup in latest_benchmark().
"""

from __future__ import annotations

import pathlib

import pytest

import arena_evaluation.storage.data_root as data_root_mod
from arena_evaluation.storage.data_root import benchmarks_root, latest_benchmark


# ---------------------------------------------------------------------------
# benchmarks_root
# ---------------------------------------------------------------------------


def test_benchmarks_root_uses_arena_data_dir(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    assert benchmarks_root() == tmp_path / "benchmarks"


def test_benchmarks_root_returns_pathlib_path(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    assert isinstance(benchmarks_root(), pathlib.Path)


def test_benchmarks_root_env_var_trailing_slash(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("ARENA_DATA_DIR", f"{tmp_path}/")
    assert benchmarks_root() == tmp_path / "benchmarks"


def test_benchmarks_root_falls_back_to_package_share(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    share = tmp_path / "share"
    monkeypatch.setattr(
        data_root_mod,
        "get_package_share_directory",
        lambda pkg: str(share / pkg),
    )
    assert benchmarks_root() == share / "arena_evaluation" / "data"


def test_benchmarks_root_empty_env_var_falls_back(monkeypatch, tmp_path: pathlib.Path):
    monkeypatch.setenv("ARENA_DATA_DIR", "")
    share = tmp_path / "share"
    monkeypatch.setattr(
        data_root_mod,
        "get_package_share_directory",
        lambda pkg: str(share / pkg),
    )
    assert benchmarks_root() == share / "arena_evaluation" / "data"


# ---------------------------------------------------------------------------
# latest_benchmark
# ---------------------------------------------------------------------------


def test_latest_benchmark_empty_root(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    assert latest_benchmark(root) is None


def test_latest_benchmark_missing_root(tmp_path: pathlib.Path):
    assert latest_benchmark(tmp_path / "does_not_exist") is None


def test_latest_benchmark_root_is_file(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.write_text("not a directory")
    assert latest_benchmark(root) is None


def test_latest_benchmark_ignores_files(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "notes.md").write_text("hello")
    (root / "not_a_run.txt").write_text("x")
    assert latest_benchmark(root) is None


def test_latest_benchmark_returns_most_recent_lexically(tmp_path: pathlib.Path):
    """'latest' is the lexicographically largest directory name, not mtime."""
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "2026-01-01").mkdir()
    (root / "2026-03-03").mkdir()
    (root / "2026-02-02").mkdir()
    assert latest_benchmark(root) == root / "2026-03-03"


def test_latest_benchmark_single_run(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "run_a").mkdir()
    assert latest_benchmark(root) == root / "run_a"


def test_latest_benchmark_returns_pathlib_path(tmp_path: pathlib.Path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "run_a").mkdir()
    result = latest_benchmark(root)
    assert isinstance(result, pathlib.Path)


def test_latest_benchmark_default_root_uses_benchmarks_root(monkeypatch, tmp_path: pathlib.Path):
    fake_root = tmp_path / "benchmarks"
    fake_root.mkdir()
    (fake_root / "run_a").mkdir()
    (fake_root / "run_b").mkdir()
    monkeypatch.setattr(data_root_mod, "benchmarks_root", lambda: fake_root)
    assert latest_benchmark() == fake_root / "run_b"
