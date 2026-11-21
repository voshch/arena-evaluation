"""Ranking validity of one evaluated episode: the defects that exclude it, and none for a clean one."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.validity import (
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


def test_a_short_track_has_no_teleport() -> None:
    assert teleport_reason(_tracks([0.0])) is None


def test_an_undeclared_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown invalid_episode reason"):
        invalid_episode("not_declared")
