"""Unit tests for arena_evaluation.storage.folder_manager.FolderManager."""

from __future__ import annotations

import os
import pathlib

import pytest

from arena_evaluation.storage.exceptions import ManifestGenerationError
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata


# helpers


def _make_metadata(
    *,
    benchmark_id: str = "bench",
    planner: str = "teb",
    stage: str = "s1",
    map_name: str = "hospital_1",
    episode_id: int | None = None,
    is_reference: bool = False,
    reference_type: str | None = None,
) -> RunMetadata:
    return RunMetadata(
        benchmark_id=benchmark_id,
        planner=planner,
        map=map_name,
        stage=stage,
        episode_id=episode_id,
        recording_started_at="2026-01-01T00:00:00+00:00",
        python_version="3.12.3",
        ros_distro="jazzy",
        modes={},
        params={},
        is_reference=is_reference,
        reference_type=reference_type,
    )


def _write_episode(
    root: pathlib.Path,
    episode_id: int,
    *,
    benchmark_id: str = "bench",
    name: str | None = None,
    legacy: bool = False,
    valid: bool = True,
    **meta_kwargs,
) -> pathlib.Path:
    """Create data_root/<benchmark_id>/episodes/<name>/ plus a metadata sidecar."""
    name = name or f"episode_{episode_id:03d}"
    ep_dir = root / benchmark_id / "episodes" / name
    ep_dir.mkdir(parents=True, exist_ok=True)
    if valid:
        meta = _make_metadata(benchmark_id=benchmark_id, episode_id=episode_id, **meta_kwargs)
        sidecar = ep_dir / ("metadata.yaml" if legacy else f"{name}.yaml")
        MetadataWriter.write(meta, sidecar)
    return ep_dir


def _make_episode_dir(root: pathlib.Path, benchmark_id: str = "bench") -> pathlib.Path:
    ep_dir = root / benchmark_id / "episodes" / "episode_001"
    ep_dir.mkdir(parents=True, exist_ok=True)
    return ep_dir


# __init__


def test_init_default_root_uses_package_share(tmp_path: pathlib.Path):
    share = tmp_path / "share" / "arena_evaluation"
    fm = FolderManager(environ={}, share_directory=lambda pkg: str(share))
    assert fm.data_root == (share / "data").resolve()
    assert fm.data_root.is_dir()


def test_init_default_root_follows_arena_data_dir(tmp_path: pathlib.Path):
    fm = FolderManager(environ={"ARENA_DATA_DIR": str(tmp_path)})
    assert fm.data_root == (tmp_path / "benchmarks").resolve()
    assert fm.data_root.is_dir()


# episodes_dir / episode_dir


# traversal protection


# discover_episodes


def test_discover_episodes_episodes_dir_is_file(tmp_path: pathlib.Path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "episodes").write_text("not a directory")
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_legacy_metadata_yaml(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 9, legacy=True)
    assert (ep_dir / "metadata.yaml").exists()
    fm = FolderManager(tmp_path)
    assert [e.episode_id for e in fm.discover_episodes("bench")] == [9]


def test_discover_episodes_skips_missing_sidecar(tmp_path: pathlib.Path):
    _write_episode(tmp_path, 1, valid=False)
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_refuses_an_invalid_sidecar_loudly(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 3, valid=False)
    (ep_dir / "episode_003.yaml").write_text("hello: world\n")
    fm = FolderManager(tmp_path)
    with pytest.raises(ManifestGenerationError, match="episode_003.yaml"):
        fm.discover_episodes("bench")


def test_discover_episodes_refuses_an_empty_sidecar_loudly(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 4, valid=False)
    (ep_dir / "episode_004.yaml").write_text("")
    fm = FolderManager(tmp_path)
    with pytest.raises(ManifestGenerationError, match="episode_004.yaml"):
        fm.discover_episodes("bench")


def test_discover_episodes_sorted_and_filtered(tmp_path: pathlib.Path):
    root = tmp_path
    _write_episode(root, 1)
    _write_episode(root, 2, planner="dwb", map_name="map_empty")
    (root / "bench" / "episodes" / "scratch").mkdir()
    (root / "bench" / "episodes" / "episode_099").write_text("file, not dir")
    (root / "bench" / "episodes" / "episode_abc").mkdir()
    _write_episode(root, 10)
    fm = FolderManager(root)
    episodes = fm.discover_episodes("bench")
    assert [e.episode_id for e in episodes] == [1, 2, 10]
    assert episodes[1].planner == "dwb"
    assert episodes[1].map == "map_empty"


# mcap_path_for_episode


def test_mcap_fallback_symlink_escape_raises(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path / "root")
    ep_dir = fm.episode_dir("bench", 1)
    ep_dir.mkdir(parents=True)
    escape = tmp_path / "escape_fallback.mcap"
    escape.write_text("secret")
    os.symlink(escape, ep_dir / "evil.mcap")
    with pytest.raises(ValueError, match="outside data_root"):
        fm.mcap_path_for_episode(ep_dir)


# extracted_topics_path_for_episode
