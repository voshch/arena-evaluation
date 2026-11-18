from __future__ import annotations

import pytest

pytest.importorskip("polars")

import polars as pl
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.social.interaction_common import (
    PEDESTRIAN_AGENT_STATES_TOPIC,
    PEDESTRIAN_ARENA_PEDS_TOPIC,
)
from arena_evaluation.processing.pipeline import _with_pedestrian_constants
from arena_evaluation.processing.scoped_evaluator import _blind_from_awareness, _latched

ARENA = pl.DataFrame(
    {
        "ped_id": ["Pedestrian_1", "Pedestrian_1", "Pedestrian_2"],
        "t_ns": [0, 1, 0],
        "radius_m": [None, None, None],
    },
    schema={"ped_id": pl.String, "t_ns": pl.Int64, "radius_m": pl.Float64},
)
ENGINE = pl.DataFrame(
    {
        "ped_id": ["Pedestrian_1", "Pedestrian_1", "Pedestrian_2"],
        "t_ns": [0, 1, 0],
        "radius_m": [0.25, 0.25, 0.4],
        "robot_awareness": [0.0, None, 1.0],
    },
    schema={
        "ped_id": pl.String,
        "t_ns": pl.Int64,
        "radius_m": pl.Float64,
        "robot_awareness": pl.Float32,
    },
)


@pytest.mark.parametrize(
    ("robot_awareness", "expected"),
    [(0.0, True), (0.3, False), (1.0, False), (None, None)],
)
def test_a_pedestrian_is_blind_only_at_zero_awareness(
    robot_awareness: float | None, expected: bool | None
) -> None:
    assert _blind_from_awareness(robot_awareness) is expected


def test_recorded_constants_join_onto_each_pedestrian_row_by_ped_id() -> None:
    joined = _with_pedestrian_constants(
        {PEDESTRIAN_ARENA_PEDS_TOPIC: ARENA, PEDESTRIAN_AGENT_STATES_TOPIC: ENGINE}
    )[PEDESTRIAN_ARENA_PEDS_TOPIC]

    assert joined.get_column("radius_m").to_list() == [0.25, 0.25, 0.4]
    assert joined.get_column("robot_awareness").to_list() == [0.0, 0.0, 1.0]


def test_without_the_engine_table_the_arena_rows_are_unchanged() -> None:
    result = _with_pedestrian_constants({PEDESTRIAN_ARENA_PEDS_TOPIC: ARENA})

    assert_frame_equal(result[PEDESTRIAN_ARENA_PEDS_TOPIC], ARENA)


def test_a_constant_the_track_frame_lacks_is_unknown() -> None:
    assert _latched(ARENA, "robot_awareness") is None
