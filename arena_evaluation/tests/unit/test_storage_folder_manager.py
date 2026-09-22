"""Unit tests for arena_evaluation.storage.folder_manager.FolderManager.

Covers path resolution, traversal protection (_safe_resolve), flat
episodes/ discovery, MCAP path selection, and combined metrics paths.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import arena_evaluation.storage.folder_manager as folder_manager_mod
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_resolves_data_root(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    assert fm.data_root == tmp_path.resolve()
    assert fm.data_root.is_dir()


def test_init_creates_missing_data_root(tmp_path: pathlib.Path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    fm = FolderManager(target)
    assert fm.data_root == target.resolve()
    assert fm.data_root.is_dir()


def test_init_accepts_str_data_root(tmp_path: pathlib.Path):
    fm = FolderManager(str(tmp_path / "nested"))
    assert fm.data_root == (tmp_path / "nested").resolve()
    assert fm.data_root.is_dir()


def test_init_default_root_uses_package_share(monkeypatch, tmp_path: pathlib.Path):
    share = tmp_path / "share" / "arena_evaluation"
    monkeypatch.setattr(
        folder_manager_mod,
        "get_package_share_directory",
        lambda pkg: str(share),
    )
    fm = FolderManager()
    assert fm.data_root == share / "data"
    assert fm.data_root.is_dir()


# ---------------------------------------------------------------------------
# episodes_dir / episode_dir
# ---------------------------------------------------------------------------


def test_episodes_dir(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    assert fm.episodes_dir("bench") == (tmp_path / "bench" / "episodes").resolve()


def test_episode_dir_formatting(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    expected = (tmp_path / "bench" / "episodes" / "episode_007").resolve()
    assert fm.episode_dir("bench", 7) == expected


@pytest.mark.parametrize(
    "episode_id, expected_name",
    [(0, "episode_000"), (42, "episode_042"), (1234, "episode_1234")],
)
def test_episode_dir_zero_padding(tmp_path: pathlib.Path, episode_id: int, expected_name: str):
    fm = FolderManager(tmp_path)
    expected = (tmp_path / "bench" / "episodes" / expected_name).resolve()
    assert fm.episode_dir("bench", episode_id) == expected


def test_episode_dir_missing_path_resolves_without_creating(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    result = fm.episode_dir("bench", 1)
    assert result == (tmp_path / "bench" / "episodes" / "episode_001").resolve()
    assert not result.exists()


def test_episode_dir_returns_pathlib_path(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    assert isinstance(fm.episode_dir("bench", 1), pathlib.Path)


# ---------------------------------------------------------------------------
# traversal protection
# ---------------------------------------------------------------------------


def test_episodes_dir_blocks_parent_traversal(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    with pytest.raises(ValueError, match="outside data_root"):
        fm.episodes_dir("../outside")


def test_episode_dir_blocks_parent_traversal(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    with pytest.raises(ValueError, match="outside data_root"):
        fm.episode_dir("../outside", 1)


def test_episode_dir_blocks_absolute_escape(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    # An absolute benchmark_id that resolves outside data_root must be refused.
    with pytest.raises(ValueError, match="outside data_root"):
        fm.episode_dir(str(tmp_path.parent / "outside"), 1)


def test_episode_dir_absolute_nested_inside_root_is_allowed(tmp_path: pathlib.Path):
    """pathlib replacement semantics: an absolute benchmark_id nested under
    data_root resolves to that nested location, which is still inside."""
    fm = FolderManager(tmp_path)
    expected = (tmp_path / "nested" / "episodes" / "episode_001").resolve()
    assert fm.episode_dir(str(tmp_path / "nested"), 1) == expected


def test_combined_metrics_path_blocks_parent_traversal(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    with pytest.raises(ValueError, match="outside data_root"):
        fm.combined_metrics_path("../outside")


def test_combined_metrics_path_blocks_absolute_escape(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    with pytest.raises(ValueError, match="outside data_root"):
        fm.combined_metrics_path(str(tmp_path.parent / "outside"))


# ---------------------------------------------------------------------------
# discover_episodes
# ---------------------------------------------------------------------------


def test_discover_episodes_no_episodes_dir(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_episodes_dir_is_file(tmp_path: pathlib.Path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "episodes").write_text("not a directory")
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_empty_episodes_dir(tmp_path: pathlib.Path):
    (tmp_path / "bench" / "episodes").mkdir(parents=True)
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_single(tmp_path: pathlib.Path):
    _write_episode(tmp_path, 1)
    fm = FolderManager(tmp_path)
    episodes = fm.discover_episodes("bench")
    assert len(episodes) == 1
    assert episodes[0].episode_id == 1


def test_discover_episodes_descriptor_fields(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 5, planner="dwb", stage="stage_one", map_name="map_empty")
    fm = FolderManager(tmp_path)
    episodes = fm.discover_episodes("bench")
    assert len(episodes) == 1
    descriptor = episodes[0]
    assert isinstance(descriptor, EpisodeDescriptor)
    assert descriptor.episode_dir == str(ep_dir)
    assert descriptor.benchmark_id == "bench"
    assert descriptor.episode_id == 5
    assert descriptor.planner == "dwb"
    assert descriptor.stage == "stage_one"
    assert descriptor.map == "map_empty"
    assert descriptor.is_reference is False
    assert descriptor.reference_type is None


def test_discover_episodes_reference_descriptor(tmp_path: pathlib.Path):
    _write_episode(tmp_path, 7, is_reference=True, reference_type="unobstructed_robot")
    fm = FolderManager(tmp_path)
    descriptor = fm.discover_episodes("bench")[0]
    assert descriptor.is_reference is True
    assert descriptor.reference_type == "unobstructed_robot"


def test_discover_episodes_legacy_metadata_yaml(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 9, legacy=True)
    assert (ep_dir / "metadata.yaml").exists()
    fm = FolderManager(tmp_path)
    assert [e.episode_id for e in fm.discover_episodes("bench")] == [9]


def test_discover_episodes_skips_non_episode_entries(tmp_path: pathlib.Path):
    root = tmp_path
    _write_episode(root, 1)
    (root / "bench" / "episodes" / "scratch").mkdir()
    (root / "bench" / "episodes" / "episode_099").write_text("file, not dir")
    fm = FolderManager(root)
    assert [e.episode_id for e in fm.discover_episodes("bench")] == [1]


def test_discover_episodes_skips_bad_ids(tmp_path: pathlib.Path):
    root = tmp_path
    _write_episode(root, 1)
    (root / "bench" / "episodes" / "episode_abc").mkdir()
    (root / "bench" / "episodes" / "episode_").mkdir()
    (root / "bench" / "episodes" / "episode_1_extra").mkdir()
    fm = FolderManager(root)
    assert [e.episode_id for e in fm.discover_episodes("bench")] == [1]


def test_discover_episodes_skips_missing_sidecar(tmp_path: pathlib.Path):
    _write_episode(tmp_path, 1, valid=False)
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_skips_invalid_sidecar(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 3, valid=False)
    (ep_dir / "episode_003.yaml").write_text("hello: world\n")
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_skips_empty_sidecar(tmp_path: pathlib.Path):
    ep_dir = _write_episode(tmp_path, 4, valid=False)
    (ep_dir / "episode_004.yaml").write_text("")
    fm = FolderManager(tmp_path)
    assert fm.discover_episodes("bench") == []


def test_discover_episodes_lexical_order_not_numeric(tmp_path: pathlib.Path):
    """Discovery sorts by directory NAME, so episode_10 precedes episode_2."""
    _write_episode(tmp_path, 2, name="episode_2")
    _write_episode(tmp_path, 10, name="episode_10")
    fm = FolderManager(tmp_path)
    assert [e.episode_id for e in fm.discover_episodes("bench")] == [10, 2]


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


# ---------------------------------------------------------------------------
# mcap_path_for_episode
# ---------------------------------------------------------------------------


def test_mcap_canonical_preferred(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "episode_001.mcap").write_text("payload")
    (ep_dir / "other.mcap").write_text("payload")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "episode_001.mcap").resolve()


def test_mcap_canonical_zero_size_still_used(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "episode_001.mcap").write_text("")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "episode_001.mcap").resolve()


def test_mcap_fallback_other_name(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "recording.mcap").write_text("payload")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "recording.mcap").resolve()


def test_mcap_fallback_first_lexically(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "b.mcap").write_text("bb")
    (ep_dir / "a.mcap").write_text("aa")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "a.mcap").resolve()


def test_mcap_fallback_skips_zero_size(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "zzz.mcap").write_text("")  # zero size: filtered out
    (ep_dir / "a.mcap").write_text("aa")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "a.mcap").resolve()


def test_mcap_only_zero_size_fallback_returns_canonical(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    (ep_dir / "recording.mcap").write_text("")
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "episode_001.mcap").resolve()


def test_mcap_missing_returns_expected_canonical(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    fm = FolderManager(tmp_path)
    assert fm.mcap_path_for_episode(ep_dir) == (ep_dir / "episode_001.mcap").resolve()


def test_mcap_canonical_symlink_escape_raises(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path / "root")
    ep_dir = fm.episode_dir("bench", 1)
    ep_dir.mkdir(parents=True)
    escape = tmp_path / "escape_canonical.mcap"
    escape.write_text("secret")
    os.symlink(escape, ep_dir / "episode_001.mcap")
    with pytest.raises(ValueError, match="outside data_root"):
        fm.mcap_path_for_episode(ep_dir)


def test_mcap_fallback_symlink_escape_raises(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path / "root")
    ep_dir = fm.episode_dir("bench", 1)
    ep_dir.mkdir(parents=True)
    escape = tmp_path / "escape_fallback.mcap"
    escape.write_text("secret")
    os.symlink(escape, ep_dir / "evil.mcap")
    with pytest.raises(ValueError, match="outside data_root"):
        fm.mcap_path_for_episode(ep_dir)


def test_mcap_broken_symlink_propagates_filenotfound(tmp_path: pathlib.Path):
    """Regression-doc: a dangling .mcap symlink crashes path resolution.

    mcap_path_for_episode stats every glob match (size > 0 filter); a
    broken symlink raises FileNotFoundError before a fallback can be
    returned. Documenting current behavior; likely a source bug.
    """
    ep_dir = _make_episode_dir(tmp_path)
    os.symlink(tmp_path / "missing_target.mcap", ep_dir / "broken.mcap")
    fm = FolderManager(tmp_path)
    with pytest.raises(FileNotFoundError):
        fm.mcap_path_for_episode(ep_dir)


# ---------------------------------------------------------------------------
# extracted_topics_path_for_episode
# ---------------------------------------------------------------------------


def test_extracted_topics_path(tmp_path: pathlib.Path):
    ep_dir = _make_episode_dir(tmp_path)
    fm = FolderManager(tmp_path)
    assert fm.extracted_topics_path_for_episode(ep_dir) == (ep_dir / "topics").resolve()


def test_extracted_topics_path_escape_raises(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path / "root")
    with pytest.raises(ValueError, match="outside data_root"):
        fm.extracted_topics_path_for_episode(tmp_path / "evil" / "episode_001")


# ---------------------------------------------------------------------------
# combined_metrics_path
# ---------------------------------------------------------------------------


def test_combined_metrics_path(tmp_path: pathlib.Path):
    fm = FolderManager(tmp_path)
    assert fm.combined_metrics_path("bench") == (tmp_path / "bench" / "combined_metrics.parquet").resolve()
