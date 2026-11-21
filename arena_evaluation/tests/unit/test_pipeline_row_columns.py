"""Integration check: the pose-frame column reaches the episode rows, and no crowd label does."""

from __future__ import annotations

import pathlib


from arena_evaluation.storage.schemas import EpisodeDescriptor

_CROWD_FIELDS = frozenset(
    {"crowd_visible", "crowd_condition", "crowd_visibility_source"}
)


def _episode(tmp_path: pathlib.Path, manifest: str | None = None) -> EpisodeDescriptor:
    episode_dir = tmp_path / "benchmarks" / "run_a" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    if manifest is not None:
        (tmp_path / "benchmarks" / "run_a" / "manifest.yaml").write_text(
            manifest, encoding="utf-8"
        )
    return EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id="run_a",
        episode_id=0,
        planner="dwb",
        stage="stage_1",
        map="stage_1_world",
    )
