"""Ranking validity of one evaluated episode (08 H1.5)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.storage.registry import INVALID_EPISODE_REASONS, NullReason

__all__ = [
    "ROBOT_TRACKS_TOPIC",
    "episode_validity",
    "invalid_episode",
    "teleport_reason",
]

ROBOT_TRACKS_TOPIC: Final[str] = "robot_tracks"

#: A single-sample ground-truth displacement larger than this is a pose jump, m. A speed gate alone is
#: not usable: TF samples carry latency jitter, so a 0.17-0.8 m step inside 33 ms looks like 5-24 m/s
#: while the trajectory stays continuous. Measured over the 620-episode fig6 set, the largest single
#: step is 0.8 m and a speed-only gate flags 80/617 episodes that never teleport; only resets exceed 1 m.
TELEPORT_STEP_M: Final[float] = 1.0

#: ``arena_runtime_msgs/EpisodeRecord`` terminal states: 2 SUCCESS, 3 FAILED, 4 SKIPPED, 5 FATAL.
RANKABLE_OUTCOME_STATES: Final[frozenset[int]] = frozenset({2, 3})
FATAL_OUTCOME_STATE: Final[int] = 5


def invalid_episode(why: str) -> str:
    """``invalid_episode:<why>`` token, rejected when ``why`` is outside the declared vocabulary."""
    base = why.partition(":")[0]
    if (
        why not in INVALID_EPISODE_REASONS
        and f"{base}:<topic>" not in INVALID_EPISODE_REASONS
    ):
        raise ValueError(
            f"unknown invalid_episode reason {why!r}; declare it in INVALID_EPISODE_REASONS"
        )
    return f"{NullReason.INVALID_EPISODE.value}:{why}"


def teleport_reason(tracks: pl.DataFrame | None) -> str | None:
    """``invalid_episode:pose_teleport`` when consecutive ground-truth samples jump more than a robot can travel."""
    if tracks is None or not {"t_ns", "x", "y"} <= set(tracks.columns):
        return None
    rows = tracks.select("t_ns", "x", "y").drop_nulls().sort("t_ns")
    if len(rows) < 2:
        return None
    steps_m = np.hypot(
        np.diff(rows["x"].to_numpy().astype(np.float64)),
        np.diff(rows["y"].to_numpy().astype(np.float64)),
    )
    return (
        invalid_episode("pose_teleport") if np.any(steps_m > TELEPORT_STEP_M) else None
    )


def episode_validity(
    *,
    robot_tracks: pl.DataFrame | None,
    outcome_state: int | None,
    integrity: Sequence[str],
) -> tuple[bool, str | None]:
    """``(is_valid_for_ranking, invalid_reason)``: every defect found, None when the episode is rankable."""
    defects: dict[str, None] = {}
    if robot_tracks is None or robot_tracks.is_empty():
        defects[invalid_episode(f"missing_stream:{ROBOT_TRACKS_TOPIC}")] = None
    elif (jump := teleport_reason(robot_tracks)) is not None:
        defects[jump] = None
    if outcome_state not in RANKABLE_OUTCOME_STATES:
        why = (
            "simulator_error"
            if outcome_state == FATAL_OUTCOME_STATE
            else "incomplete_recording"
        )
        defects[invalid_episode(why)] = None
    if integrity:
        defects[invalid_episode("incomplete_recording")] = None
    reason = ";".join(defects) or None
    return not reason, reason
