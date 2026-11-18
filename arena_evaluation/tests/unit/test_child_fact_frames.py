"""Child fact frames: each builder emits exactly the columns its registry table declares, with every element kept."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator
from arena_evaluation.storage.registry import FACT_TABLES, FactTable
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

_IDENTITY = {"run_id": "run", "episode_id": 1, "robot_id": "env_0_jackal_0"}
_NS_PER_S = 1_000_000_000


def _evaluator() -> ScopedEvaluator:
    """The child builders read no evaluator state, so a bare instance serves them."""
    return ScopedEvaluator.__new__(ScopedEvaluator)


def _episode(times_ns: list[int]) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": times_ns}, schema={"t_ns": pl.Int64}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
        topics={},
    )


def _declared(table: FactTable) -> list[str]:
    return FACT_TABLES[table].polars_schema().names()


def test_edge_case_rows_keep_each_list_element() -> None:
    frame = _evaluator()._edge_case_fact_frame(
        **_IDENTITY,
        metrics={"edge_case_steps": ["s0", "s1"], "edge_case_agents": ["a0"]},
    )
    assert frame.columns == _declared(FactTable.EDGE_CASES)
    expected = pl.DataFrame(
        {
            "kind": ["step", "step", "agent"],
            "ordinal": [0, 1, 0],
            "element": ["s0", "s1", "a0"],
        },
        schema={"kind": pl.String, "ordinal": pl.Int64, "element": pl.String},
    )
    assert_frame_equal(frame.select(expected.columns), expected)


def test_formation_rows_keep_each_member_and_absent_agent() -> None:
    frame = _evaluator()._formation_fact_frame(
        **_IDENTITY,
        metrics={
            "edge_case_formation_members": ["m0"],
            "edge_case_formation_absent": ["x0"],
        },
    )
    assert frame.columns == _declared(FactTable.FORMATIONS)
    expected = pl.DataFrame(
        {"role": ["member", "absent"], "element": ["m0", "x0"]},
        schema={"role": pl.String, "element": pl.String},
    )
    assert_frame_equal(frame.select(expected.columns), expected)


def test_phase_outcome_rows_keep_the_token_and_the_met_flag() -> None:
    frame = _evaluator()._phase_outcome_fact_frame(
        **_IDENTITY, metrics={"phase_outcomes": ["met", "failed"]}
    )
    assert frame.columns == _declared(FactTable.PHASE_OUTCOMES)
    expected = pl.DataFrame(
        {"phase_idx": [0, 1], "outcome": ["met", "failed"], "met": [True, False]},
        schema={"phase_idx": pl.Int64, "outcome": pl.String, "met": pl.Boolean},
    )
    assert_frame_equal(frame.select(expected.columns), expected)


def test_judge_disagreement_rows_keep_their_text() -> None:
    frame = _evaluator()._judge_disagreement_fact_frame(
        **_IDENTITY, metrics={"judge_disagreements": ["replay ends early"]}
    )
    assert frame.columns == _declared(FactTable.JUDGE_DISAGREEMENTS)
    expected = pl.DataFrame(
        {"disagreement_idx": [0], "disagreement": ["replay ends early"]},
        schema={"disagreement_idx": pl.Int64, "disagreement": pl.String},
    )
    assert_frame_equal(frame.select(expected.columns), expected)


def test_collision_rows_keep_the_obstacle_identifier_without_a_contact_stream() -> None:
    frame = _evaluator()._collision_fact_frame(
        _episode([0]), **_IDENTITY, metrics={"collision_obstacles": ["wall_3"]}
    )
    assert frame.columns == _declared(FactTable.COLLISIONS)
    expected = pl.DataFrame(
        {
            "collision_idx": [0],
            "obstacle_id": ["wall_3"],
            "obstacle_kind": [None],
            "t_ns": [None],
            "x_m": [None],
            "y_m": [None],
        },
        schema={
            "collision_idx": pl.Int64,
            "obstacle_id": pl.String,
            "obstacle_kind": pl.String,
            "t_ns": pl.Int64,
            "x_m": pl.Float64,
            "y_m": pl.Float64,
        },
    )
    assert_frame_equal(frame.select(expected.columns), expected)


def test_mechanism_rows_record_each_field_at_its_change_points() -> None:
    state = {
        "entity": "door_1",
        "level": "0",
        "kind": "door",
        "progress": [0.0, 0.5],
        "locked": [False, True],
    }
    frame = _evaluator()._mechanism_state_fact_frame(
        _episode([0, _NS_PER_S]), **_IDENTITY, metrics={"mechanism_states": [state]}
    )
    assert frame.columns == _declared(FactTable.MECHANISM_STATES)
    expected = pl.DataFrame(
        {
            "t_ns": [0, 0, _NS_PER_S, _NS_PER_S],
            "field": ["locked", "progress", "locked", "progress"],
            "value": [0.0, 0.0, 1.0, 0.5],
        },
        schema={"t_ns": pl.Int64, "field": pl.String, "value": pl.Float64},
    )
    assert_frame_equal(frame.select(expected.columns), expected)
