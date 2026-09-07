import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.path_metrics import PathMetricsCalculator


def test_path_length_straight():
    # Straight line along X axis
    odom_df = pl.DataFrame({"time_ns": [1, 2, 3, 4], "pos_x": [0.0, 1.0, 2.0, 3.0], "pos_y": [0.0, 0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0, 0.0]})

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathMetricsCalculator(params)

    results = calc.calculate(episode, {})

    assert np.isclose(results["path_length"], 3.0)
    assert np.isclose(results["curvature_mean"], 0.0)
    assert np.isclose(results["roughness_mean"], 0.0)
    assert np.isclose(results["angle_over_length"], 0.0)


def test_path_length_stationary():
    # Stationary robot
    odom_df = pl.DataFrame({"time_ns": [1, 2, 3, 4], "pos_x": [0.0, 0.0, 0.0, 0.0], "pos_y": [0.0, 0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0, 0.0]})

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathMetricsCalculator(params)

    results = calc.calculate(episode, {})

    assert np.isclose(results["path_length"], 0.0)
    assert np.isclose(results["angle_over_length"], 0.0)


def test_path_length_with_nans():
    # Straight line along X axis with some nulls/NaNs
    odom_df = pl.DataFrame({"time_ns": [1, 2, 3, 4], "pos_x": [0.0, None, 2.0, 3.0], "pos_y": [0.0, 0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0, 0.0]})

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathMetricsCalculator(params)

    results = calc.calculate(episode, {})

    # The second row (index 1) has a None/null pos_x, so it is filtered out.
    # The remaining points are at X = 0.0, 2.0, 3.0.
    # The total path length should be 2.0 + 1.0 = 3.0 (no NaNs should propagate).
    assert results["path_length"] is not None
    assert not np.isnan(results["path_length"])
    assert np.isclose(results["path_length"], 3.0)
