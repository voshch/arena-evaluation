import argparse
import os
import pathlib
import tempfile
import pytest
from unittest import mock

from arena_evaluation.cli import resolve_paths


def test_resolve_paths_literal_exists():
    """Ensure existing path is not modified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        args = argparse.Namespace(run_dir=tmp_path, benchmark_dir=tmp_path)
        resolved = resolve_paths(args)
        assert resolved.run_dir == tmp_path
        assert resolved.benchmark_dir == tmp_path


@mock.patch.dict(os.environ, {}, clear=True)
def test_resolve_paths_recording_id():
    """Ensure run_dir is resolved when specifying only ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        rec_dir = tmp_path / "data" / "recordings" / "20260610-153034"
        rec_dir.mkdir(parents=True)

        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            args = argparse.Namespace(run_dir=pathlib.Path("20260610-153034"))
            resolved = resolve_paths(args)
            assert resolved.run_dir == rec_dir.resolve()


@mock.patch.dict(os.environ, {}, clear=True)
def test_resolve_paths_benchmark_id():
    """Ensure benchmark_dir is resolved when specifying only ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        bench_dir = tmp_path / "data" / "benchmarks" / "my_benchmark"
        bench_dir.mkdir(parents=True)

        with mock.patch("pathlib.Path.cwd", return_value=tmp_path):
            args = argparse.Namespace(benchmark_dir=pathlib.Path("my_benchmark"))
            resolved = resolve_paths(args)
            assert resolved.benchmark_dir == bench_dir.resolve()


def test_resolve_paths_env_var():
    """Ensure ARENA_DATA_DIR env variable is respected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        env_data_dir = tmp_path / "custom_env_data"
        rec_dir = env_data_dir / "recordings" / "20260610-153034"
        rec_dir.mkdir(parents=True)

        with mock.patch.dict(os.environ, {"ARENA_DATA_DIR": str(env_data_dir)}):
            args = argparse.Namespace(run_dir=pathlib.Path("20260610-153034"))
            resolved = resolve_paths(args)
            assert resolved.run_dir == rec_dir.resolve()


def test_resolve_paths_nonexistent():
    """Ensure paths that do not exist are left as is."""
    args = argparse.Namespace(run_dir=pathlib.Path("nonexistent_run"), benchmark_dir=pathlib.Path("nonexistent_bench"))
    resolved = resolve_paths(args)
    assert resolved.run_dir == pathlib.Path("nonexistent_run")
    assert resolved.benchmark_dir == pathlib.Path("nonexistent_bench")
