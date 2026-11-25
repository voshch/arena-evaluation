from __future__ import annotations

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.metrics.social.interaction_common import (
    PEDESTRIAN_AGENT_STATES_TOPIC,
    PEDESTRIAN_ARENA_PEDS_TOPIC,
)
from arena_evaluation.processing.pipeline import _with_pedestrian_constants
from arena_evaluation.processing.scoped_evaluator import ScopedEvaluator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

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
        "robot_blind": [True, None, False],
    },
    schema={
        "ped_id": pl.String,
        "t_ns": pl.Int64,
        "radius_m": pl.Float64,
        "robot_blind": pl.Boolean,
    },
)


def _pedestrian_blindness(tracks: pl.DataFrame) -> bool | None:
    bundle = AlignedEpisodeBundle(
        episode_id=0,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0],
        goal_pos=[1.0, 0.0],
        topics={},
    )
    slices = ScopedEvaluator._build_pedestrian_slices(
        bundle,
        tracks,
        run_id="run",
        episode_id=0,
        env_id="env_0",
        ep_start_ns=0,
        ep_end_ns=0,
        episode_slice_id="ep",
    )
    return slices[0].identity["is_blind"]


def test_recorded_constants_join_onto_each_pedestrian_row_by_ped_id() -> None:
    joined = _with_pedestrian_constants({PEDESTRIAN_ARENA_PEDS_TOPIC: ARENA, PEDESTRIAN_AGENT_STATES_TOPIC: ENGINE})[PEDESTRIAN_ARENA_PEDS_TOPIC]

    assert joined.get_column("radius_m").to_list() == [0.25, 0.25, 0.4]
    assert joined.get_column("robot_blind").to_list() == [True, True, False]
