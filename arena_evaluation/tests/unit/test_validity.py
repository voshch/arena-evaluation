"""Ranking validity of one evaluated episode: the defects that exclude it, and none for a clean one.

Every fixture is in memory: no disk I/O, no mocks.
"""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.validity import (
    TELEPORT_STEP_M,
    episode_validity,
    invalid_episode,
    teleport_reason,
)

NS_PER_TICK = 100_000_000
RANKABLE_OUTCOME = 2
FATAL_OUTCOME = 5
SKIPPED_OUTCOME = 4


def _tracks(xs: list[float], ys: list[float] | None = None) -> pl.DataFrame:
    ys = [0.0] * len(xs) if ys is None else ys
    return pl.DataFrame(
        {"t_ns": [i * NS_PER_TICK for i in range(len(xs))], "x": xs, "y": ys}
    )


def test_a_clean_recording_is_rankable_with_no_reason() -> None:
    assert episode_validity(
        robot_tracks=_tracks([0.0, 0.1, 0.2]),
        outcome_state=RANKABLE_OUTCOME,
        integrity=(),
    ) == (True, None)


def test_a_missing_robot_stream_is_refused_with_its_reason() -> None:
    assert episode_validity(
        robot_tracks=None, outcome_state=RANKABLE_OUTCOME, integrity=()
    ) == (
        False,
        invalid_episode("missing_stream:robot_tracks"),
    )


@pytest.mark.parametrize(
    ("outcome_state", "why"),
    [
        (FATAL_OUTCOME, "simulator_error"),
        (SKIPPED_OUTCOME, "incomplete_recording"),
        (None, "incomplete_recording"),
    ],
)
def test_an_outcome_that_is_not_terminal_and_rankable_is_refused(
    outcome_state: int | None, why: str
) -> None:
    assert episode_validity(
        robot_tracks=_tracks([0.0, 0.1]), outcome_state=outcome_state, integrity=()
    ) == (False, invalid_episode(why))


def test_lost_sections_in_the_recording_are_refused() -> None:
    assert episode_validity(
        robot_tracks=_tracks([0.0, 0.1]),
        outcome_state=RANKABLE_OUTCOME,
        integrity=("gap",),
    ) == (
        False,
        invalid_episode("incomplete_recording"),
    )


def test_every_defect_is_listed_once() -> None:
    _, reason = episode_validity(
        robot_tracks=None, outcome_state=None, integrity=("gap",)
    )

    assert reason == ";".join(
        dict.fromkeys(
            [
                invalid_episode("missing_stream:robot_tracks"),
                invalid_episode("incomplete_recording"),
            ]
        )
    )


@pytest.mark.parametrize(
    ("step_m", "teleports"),
    [
        (TELEPORT_STEP_M - 0.01, False),
        (TELEPORT_STEP_M, False),
        (TELEPORT_STEP_M + 0.01, True),
    ],
)
def test_a_pose_jump_is_a_teleport_only_beyond_the_declared_step(
    step_m: float, teleports: bool
) -> None:
    reason = teleport_reason(_tracks([0.0, step_m]))

    assert (reason == invalid_episode("pose_teleport")) is teleports


def test_a_short_track_has_no_teleport() -> None:
    assert teleport_reason(_tracks([0.0])) is None


def test_an_undeclared_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown invalid_episode reason"):
        invalid_episode("not_declared")
