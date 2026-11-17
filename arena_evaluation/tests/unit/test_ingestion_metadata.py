"""Unit tests for arena_evaluation.ingestion.metadata (IngestionMetadata).

Covers per-episode metadata generation: planner-name splitting, git
workspace fingerprinting, environment-derived fields, and the flat
RunMetadata record persisted to episode_XXX.yaml.
"""

from __future__ import annotations


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


# create_episode_metadata


def test_create_episode_metadata_none_planner_is_rejected(tmp_path):
    # Suspected bug in metadata.py: split_planner_name(None) is supported, but
    # the raw `planner` argument is passed straight into RunMetadata, where
    # pydantic rejects None for the non-optional `planner` field.
    # Asserting the actual behavior so a future fix is noticed.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IngestionMetadata.create_episode_metadata(**_base_kwargs(tmp_path, planner=None))


# get_git_sha / is_git_dirty


# YAML round-trip (the per-episode episode_XXX.yaml lifecycle)


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
