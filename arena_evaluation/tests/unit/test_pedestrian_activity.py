"""Pedestrian activity distance: a step pairs one sample's x with the same sample's y."""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.pedestrian_activity import (
    PedestrianActivityCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
PED_ID = "Pedestrian_0"


def _track(xs: Sequence[float | None], ys: Sequence[float | None]) -> pl.DataFrame:
    """One pedestrian's samples at 1 s spacing, a sample may carry a null x or a null y."""
    assert len(xs) == len(ys)
    frame = flat_peds(
        [
            (index * SEC, [{"ped_id": PED_ID, "x": 0.0, "y": 0.0}])
            for index in range(len(xs))
        ]
    )
    return frame.with_columns(
        pl.Series("x", xs, dtype=pl.Float64), pl.Series("y", ys, dtype=pl.Float64)
    )


def _total_distance_m(peds: pl.DataFrame) -> float:
    calculator = PedestrianActivityCalculator(make_robot_params())
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
        num_pedestrians=1,
        topics={"pedestrian_arena_peds": peds},
    )
    total = calculator.calculate(episode, {})["total_distance_m"]
    assert isinstance(total, float)
    return total


def test_a_complete_track_sums_every_step() -> None:
    """With no missing position every consecutive pair is one step."""
    peds = _track([0.0, 3.0, 3.0], [0.0, 0.0, 4.0])

    assert _total_distance_m(peds) == pytest.approx(7.0)


def test_a_step_pairs_the_same_sample_x_and_y() -> None:
    """The null x at sample 2 and the null y at sample 1 must not shift the pairs apart.

    ``x`` and ``y`` each lose one difference, so dropping the null differences one column at a
    time keeps two values on both sides and pairs sample 1's dx with sample 3's dy: 2 * sqrt(2).
    Only sample 4 spans two positions, so the walked distance is sqrt(2).
    """
    peds = _track([0.0, 1.0, None, 4.0, 5.0], [0.0, None, 3.0, 4.0, 5.0])

    assert _total_distance_m(peds) == pytest.approx(math.sqrt(2.0))


def test_a_null_position_never_contributes_a_step() -> None:
    """A step needs both of its endpoints, so the samples around a null position are skipped."""
    peds = _track(
        [0.0, 3.0, None, 6.0, 9.0, 9.0], [0.0, 0.0, 4.0, 8.0, 8.0, None]
    )

    assert _total_distance_m(peds) == pytest.approx(6.0)
