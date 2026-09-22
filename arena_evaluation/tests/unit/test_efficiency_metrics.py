import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator


def test_path_efficiency():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])
    calc = PathEfficiencyCalculator(RobotParams(0.2, 0.0, 10.0))

    # Euclidean dist is 5.0. Path length is 10.0. Efficiency should be 0.5.
    results = calc.calculate(episode, {"path_length": 10.0})
    assert np.isclose(results["path_efficiency"], 0.5)


def test_path_efficiency_clamp():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0], goal_pos=[3.0, 4.0])
    calc = PathEfficiencyCalculator(RobotParams(0.2, 0.0, 10.0))

    # Euclidean dist is 5.0. Path length is 4.0 (impossible practically). Efficiency should be clamped to 1.0.
    results = calc.calculate(episode, {"path_length": 4.0})
    assert np.isclose(results["path_efficiency"], 1.0)
