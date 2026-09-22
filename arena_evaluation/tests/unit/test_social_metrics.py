import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator
from arena_evaluation.processing.metrics.social.gaze import GazeMetricsCalculator


def test_proxemics_no_peds():
    calc = ProxemicsCalculator(RobotParams(0.2, 0.0, 10.0))
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)

    results = calc.calculate(episode, {})
    assert results["time_in_personal_space"] is None
    assert results["total_time_in_personal_space"] is None
    assert results["avg_velocity_in_personal_space"] is None


def test_gaze_no_peds():
    calc = GazeMetricsCalculator(RobotParams(0.2, 0.0, 10.0))
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)

    results = calc.calculate(episode, {})
    assert results["time_looking_at_pedestrians"] is None
    assert results["total_time_looking_at_pedestrians"] is None
    assert results["time_looked_at_by_pedestrians"] is None
    assert results["total_time_looked_at_by_pedestrians"] is None
