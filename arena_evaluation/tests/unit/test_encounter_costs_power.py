"""Encounter energy reads the recorded robot power in time order and only inside its span."""

import numpy as np
import polars as pl

import pytest

from arena_evaluation.processing.metrics.social.encounter_costs import (
    _psychoacoustic_yielding_cost,
    _robot_excess_energy_wh,
    _window_energy_wh,
)
from arena_evaluation.processing.metrics.social.interaction_common import Trajectory2D
from arena_evaluation.processing.path.theta_star import RouteResult
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

SEC = 1_000_000_000


def _episode_with_power(power_table: pl.DataFrame) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0, 0.0],
        topics={"robot_power": power_table},
    )


def test_a_window_past_the_recorded_power_has_no_energy() -> None:
    times_ns = np.arange(7, dtype=np.int64) * (SEC // 2)
    robot = Trajectory2D(
        times_ns, np.arange(7) * 0.1, np.zeros(7), np.ones(7), np.zeros(7), 0.3
    )
    power = (np.array([0.0, 2.0 * SEC]), np.array([10.0, 10.0]))
    reasons: dict[str, str] = {}
    _robot_excess_energy_wh(robot, robot, [], power, (0, 3 * SEC), 1.0, reasons)
    assert reasons["kcsc_wh_vs_planned_route"] == "no_samples"


def test_window_energy_and_pyc_integrate_over_encounter_samples() -> None:
    times_ns = np.array([0, SEC, 2 * SEC], dtype=np.int64)
    robot = Trajectory2D(
        times_ns, np.array([0.0, 1.0, 2.0]), np.zeros(3), np.ones(3), np.zeros(3), 0.3
    )
    power = (np.array([0.0, float(2 * SEC)]), np.array([36.0, 36.0]))
    wh = _window_energy_wh(robot, power, (0, 2 * SEC))
    assert wh is not None
    assert wh * 3600.0 == pytest.approx(72.0)

    ped_window = Trajectory2D(
        times_ns,
        np.array([0.0, 1.0, 2.0]),
        np.array([0.5, 0.5, 0.5]),
        np.ones(3),
        np.zeros(3),
        0.3,
    )
    ped_route = RouteResult(
        points=np.array([[0.0, 0.0], [2.0, 0.0]]), length_m=2.0, null_reason=None
    )
    reasons: dict[str, str] = {}
    pyc = _psychoacoustic_yielding_cost(
        ped_window,
        ped_route,
        (np.array([0, 2 * SEC], dtype=np.int64), np.array([100.0, 100.0])),
        reasons,
    )
    # dev=0.5 m, log10(100)=2.0, dt=1s twice -> 2.0 m*s
    assert pyc == pytest.approx(2.0)
