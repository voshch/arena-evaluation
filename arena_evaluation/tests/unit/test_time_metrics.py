import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.time_metrics import TimeMetricsCalculator


def test_time_to_goal_and_idling():
    odom_df = pl.DataFrame(
        {
            "time_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=odom_df, start_pos=[], goal_pos=[])
    calc = TimeMetricsCalculator(RobotParams(0.2, 0.0, 10.0))

    prior = {"velocity": [0.5, 0.0, 0.0, 0.5]}
    results = calc.calculate(episode, prior)

    assert np.isclose(results["time_to_goal"], 3.0)  # 4s - 1s = 3s
    assert results["time_diff"] == 3_000_000_000
    assert len(results["time"]) == 4

    # dt between [2s,3s] and [3s,4s]?
    # velocities: [0.5, 0.0, 0.0, 0.5]
    # mask: [False, True, True, False]
    # dt: [1s, 1s, 1s]
    # mask[:-1]: [False, True, True]
    # idling_time = sum of dt where mask is True = 1.0 + 1.0 = 2.0s
    assert np.isclose(results["idling_time"], 2.0)
