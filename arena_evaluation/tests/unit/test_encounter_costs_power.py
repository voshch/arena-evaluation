"""Encounter energy reads the recorded robot power in time order and only inside its span."""

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.social.encounter_costs import (
    _robot_excess_energy_wh,
)
from arena_evaluation.processing.metrics.social.interaction_common import Trajectory2D
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
