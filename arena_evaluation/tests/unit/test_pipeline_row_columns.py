"""Integration check: the pose-frame column reaches the episode rows, and no crowd label does.

Explicit crowd visibility is a set of per-pedestrian settings (`dim_pedestrians`), not an episode
label (D-21), so the row builder no longer publishes `crowd_visible`, `crowd_condition` or
`crowd_visibility_source`.
"""

from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _status_row
from arena_evaluation.storage.schemas import EpisodeDescriptor

_CROWD_FIELDS = frozenset({"crowd_visible", "crowd_condition", "crowd_visibility_source"})


def _episode(tmp_path: pathlib.Path, manifest: str | None = None) -> EpisodeDescriptor:
    episode_dir = tmp_path / "benchmarks" / "run_a" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    if manifest is not None:
        (tmp_path / "benchmarks" / "run_a" / "manifest.yaml").write_text(manifest, encoding="utf-8")
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id="run_a",
        episode_id=0,
        planner="dwb",
        stage="stage_1",
        map="stage_1_world",
    )


def test_status_row_carries_the_pose_frame_column(tmp_path: pathlib.Path) -> None:
    row = _status_row(_episode(tmp_path, "run_id: run_a\ncli_args: ['human.robots.visible:=false']\n"), None, "robot", "no_recording", "no MCAP")
    assert "pose_frame" in row
    assert row["pose_frame"] is None


def test_status_row_publishes_no_explicit_crowd_label(tmp_path: pathlib.Path) -> None:
    row = _status_row(_episode(tmp_path), None, "robot", "error", "processing failed")
    assert _CROWD_FIELDS.isdisjoint(row)
