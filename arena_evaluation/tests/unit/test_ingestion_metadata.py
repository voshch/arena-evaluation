"""Unit tests for arena_evaluation.ingestion.metadata (IngestionMetadata).

Covers per-episode metadata generation: planner-name splitting, git
workspace fingerprinting, environment-derived fields, and the flat
RunMetadata record persisted to episode_XXX.yaml.
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from arena_evaluation.ingestion.metadata import IngestionMetadata
from arena_evaluation.storage.manifest import MetadataWriter


def _base_kwargs(tmp_path, **overrides) -> dict:
    """Keyword args for a deterministic create_episode_metadata call.

    workspace_dir points at a non-repo tmp dir so the git probes deterministically
    return None/False without extra mocking.
    """
    kwargs = dict(
        benchmark_id="bench_1",
        planner="contest-teb-mpc",
        stage="stage_1",
        map_name="hospital_1",
        episode_id=3,
        robot_model="jackal",
        workspace_dir=str(tmp_path),
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# create_episode_metadata
# ---------------------------------------------------------------------------


def test_create_episode_metadata_populates_flat_fields(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    assert meta.benchmark_id == "bench_1"
    assert meta.planner == "contest-teb-mpc"
    assert meta.stage == "stage_1"
    assert meta.map == "hospital_1"
    assert meta.episode_id == 3
    assert meta.robot_model == ["jackal"]
    assert meta.local_planner == "teb"
    assert meta.inter_planner == "mpc"
    assert meta.arena_git_sha is None  # tmp_path is not a git repo
    assert meta.arena_git_dirty is False
    assert meta.recording_started_at is not None
    datetime.datetime.fromisoformat(meta.recording_started_at)


@pytest.mark.parametrize(
    "planner, expected_local, expected_inter",
    [
        ("contest-teb-mpc", "teb", "mpc"),
        ("org-teb", "org", "teb"),
        ("teb", "teb", "none"),
        ("", "unknown", "unknown"),
    ],
)
def test_create_episode_metadata_splits_planner_names(tmp_path, planner, expected_local, expected_inter):
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path, planner=planner))
    assert meta.local_planner == expected_local
    assert meta.inter_planner == expected_inter


def test_create_episode_metadata_none_planner_is_rejected(tmp_path):
    # Suspected bug in metadata.py: split_planner_name(None) is supported, but
    # the raw `planner` argument is passed straight into RunMetadata, where
    # pydantic rejects None for the non-optional `planner` field.
    # Asserting the actual behavior so a future fix is noticed.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path, planner=None))


def test_create_episode_metadata_explicit_planners_override_split(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(
        **_base_kwargs(tmp_path, planner="contest-teb-mpc"),
        local_planner="my_local",
        inter_planner="my_inter",
    )
    assert meta.local_planner == "my_local"
    assert meta.inter_planner == "my_inter"


def test_create_episode_metadata_partial_planner_override_uses_fallback(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(
        **_base_kwargs(tmp_path, planner="contest-teb-mpc"),
        local_planner="my_local",
    )
    assert meta.local_planner == "my_local"
    assert meta.inter_planner == "mpc"


def test_create_episode_metadata_reference_and_namespace_fields(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(
        **_base_kwargs(tmp_path),
        env_ns_root="/env_0",
        is_reference=True,
        reference_type="golden",
    )
    assert meta.env_ns_root == "/env_0"
    assert meta.is_reference is True
    assert meta.reference_type == "golden"

    plain = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    assert plain.env_ns_root is None
    assert plain.is_reference is False
    assert plain.reference_type is None


def test_create_episode_metadata_episode_context_fields(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(
        **_base_kwargs(tmp_path, episode_id=12),
        suite_name="suite_1",
        contest_name="contest_1",
        episodes_requested=25,
        agent_name="jackal",
        task_generator_episode_id=99,
    )
    assert meta.episode_id == 12
    assert meta.suite_name == "suite_1"
    assert meta.contest_name == "contest_1"
    assert meta.episodes_requested == 25
    assert meta.agent_name == "jackal"
    assert meta.task_generator_episode_id == 99


def test_create_episode_metadata_git_fields_from_workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(IngestionMetadata, "get_git_sha", staticmethod(lambda _ws: "a" * 40))
    monkeypatch.setattr(IngestionMetadata, "is_git_dirty", staticmethod(lambda _ws: True))
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    assert meta.arena_git_sha == "a" * 40
    assert meta.arena_git_dirty is True


def test_create_episode_metadata_python_version_and_ros_distro(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "version", "9.99.0 (test-build)")
    monkeypatch.setenv("ROS_DISTRO", "humble")
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    assert meta.python_version == "9.99.0"
    assert meta.ros_distro == "humble"


def test_create_episode_metadata_ros_distro_unknown_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv("ROS_DISTRO", raising=False)
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    assert meta.ros_distro == "unknown"


# ---------------------------------------------------------------------------
# get_git_sha / is_git_dirty
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_get_git_sha_returns_stripped_head_from_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)

    expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    sha = IngestionMetadata.get_git_sha(str(repo))
    assert sha == expected
    assert len(sha) == 40


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_is_git_dirty_detects_modified_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)

    assert IngestionMetadata.is_git_dirty(str(repo)) is False
    (repo / "f.txt").write_text("y")
    assert IngestionMetadata.is_git_dirty(str(repo)) is True


def test_get_git_sha_returns_none_when_rev_parse_fails(monkeypatch, tmp_path):
    def _raise_called(*_a, **_k):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("arena_evaluation.ingestion.metadata.subprocess.run", _raise_called)
    assert IngestionMetadata.get_git_sha(str(tmp_path)) is None


def test_get_git_sha_returns_none_when_git_missing(monkeypatch, tmp_path):
    def _raise_missing(*_a, **_k):
        raise FileNotFoundError("git")

    monkeypatch.setattr("arena_evaluation.ingestion.metadata.subprocess.run", _raise_missing)
    assert IngestionMetadata.get_git_sha(str(tmp_path)) is None


def test_is_git_dirty_false_on_subprocess_error(monkeypatch, tmp_path):
    def _raise_called(*_a, **_k):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("arena_evaluation.ingestion.metadata.subprocess.run", _raise_called)
    assert IngestionMetadata.is_git_dirty(str(tmp_path)) is False


def test_is_git_dirty_reflects_porcelain_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "arena_evaluation.ingestion.metadata.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=" M modified.txt\n"),
    )
    assert IngestionMetadata.is_git_dirty(str(tmp_path)) is True
    monkeypatch.setattr(
        "arena_evaluation.ingestion.metadata.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=""),
    )
    assert IngestionMetadata.is_git_dirty(str(tmp_path)) is False


# ---------------------------------------------------------------------------
# YAML round-trip (the per-episode episode_XXX.yaml lifecycle)
# ---------------------------------------------------------------------------


def test_episode_metadata_roundtrip_through_metadata_writer(tmp_path):
    meta = IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path))
    dest = tmp_path / "episode_000.yaml"
    MetadataWriter.write(meta, dest)
    loaded = MetadataWriter.read(dest)

    assert loaded.benchmark_id == meta.benchmark_id
    assert loaded.planner == "contest-teb-mpc"
    assert loaded.robot_model == ["jackal"]
    assert loaded.map == "hospital_1"
    assert loaded.stage == "stage_1"
    assert loaded.episode_id == 3
    assert loaded.local_planner == "teb"
    assert loaded.inter_planner == "mpc"
    assert loaded.recording_started_at == meta.recording_started_at
    assert loaded.python_version == meta.python_version
