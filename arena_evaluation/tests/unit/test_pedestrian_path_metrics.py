"""Pedestrian trajectories: one row per (t_ns, ped_id) in the 1NF pedestrian path series table."""

from __future__ import annotations

import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.performance.pedestrian_path_metrics import (
    PedestrianPathMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _episode(times: list[int], frames: list[tuple[int, list[dict]]] | None = None) -> AlignedEpisodeBundle:
    """Robot samples at the given times, with the pedestrian table of the given frames."""
    data = pl.DataFrame({"t_ns": times}, schema={"t_ns": pl.Int64})
    if frames is None:
        return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])
    peds = flat_peds(frames)
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[],
        goal_pos=[],
        peds=peds,
        topics={"pedestrian_arena_peds": peds},
    )


def _series(times: list[int], frames: list[tuple[int, list[dict]]]) -> pl.DataFrame:
    calculator = PedestrianPathMetricsCalculator(make_robot_params())
    return calculator.series_table(_episode(times, frames), run_id="run", robot_id="jackal_0")


def _at(x: float, y: float) -> dict:
    return {"x": x, "y": y}


def test_the_legacy_list_key_has_no_metric_row_output():
    assert PedestrianPathMetricsCalculator.output_keys() == []
    assert PedestrianPathMetricsCalculator.OUTPUTS == ()
    assert PedestrianPathMetricsCalculator(make_robot_params()).calculate(AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[]), {}) == {}


def test_a_pedestrian_table_without_a_position_column_is_rejected():
    calculator = PedestrianPathMetricsCalculator(make_robot_params())
    tracks = pl.DataFrame(
        {"t_ns": [0], "ped_id": ["a"], "x": [0.0]},
        schema={"t_ns": pl.Int64, "ped_id": pl.String, "x": pl.Float64},
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0]}, schema={"t_ns": pl.Int64}),
        start_pos=[],
        goal_pos=[],
        topics={"pedestrian_arena_peds": tracks},
    )

    with pytest.raises(ValueError, match=r"pedestrian_arena_peds lacks \['y'\]"):
        calculator.series_table(episode)
