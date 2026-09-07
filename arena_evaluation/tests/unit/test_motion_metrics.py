import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.motion_metrics import MotionMetricsCalculator


def test_motion_constant_velocity():
    odom_df = pl.DataFrame({"time_ns": [1000000000, 2000000000, 3000000000, 4000000000], "vel_linear": [1.0, 1.0, 1.0, 1.0], "vel_angular": [0.0, 0.0, 0.0, 0.0]})

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = MotionMetricsCalculator(params)

    results = calc.calculate(episode, {})

    assert np.isclose(results["velocity_mean"], 1.0)
    assert np.isclose(results["velocity_max"], 1.0)
    assert np.isclose(results["acceleration_mean"], 0.0)
    assert np.isclose(results["jerk_mean"], 0.0)


def test_motion_linear_acceleration():
    odom_df = pl.DataFrame(
        {
            "time_ns": [1000000000, 2000000000, 3000000000, 4000000000],
            "vel_linear": [0.0, 1.0, 2.0, 3.0],  # accel is 1.0 per step
            "vel_angular": [0.0, 0.0, 0.0, 0.0],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = MotionMetricsCalculator(params)

    results = calc.calculate(episode, {})

    assert np.isclose(results["velocity_mean"], 1.5)
    assert np.isclose(results["velocity_max"], 3.0)
    assert np.isclose(results["acceleration_mean"], 1.0)
    assert np.isclose(results["jerk_mean"], 0.0)
