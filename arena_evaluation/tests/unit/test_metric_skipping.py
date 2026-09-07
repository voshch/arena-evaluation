import pytest
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.registry import MetricRegistry


def test_registry_skipping_with_available_topics():
    params = RobotParams(0.25, 0.0, 30.0)
    registry = MetricRegistry(params)

    df_odom_only = pl.DataFrame({"time_ns": [1000, 2000, 3000], "pos_x": [0.0, 1.0, 2.0], "pos_y": [0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0], "vel_linear": [1.0, 1.0, 1.0], "vel_angular": [0.0, 0.0, 0.0]})

    episode = AlignedEpisodeBundle(episode_id=1, data=df_odom_only, start_pos=[0.0, 0.0, 0.0], goal_pos=[2.0, 0.0, 0.0])

    available_topics = {"odom"}
    results = registry.run(episode, pedsim_available=True, available_topics=available_topics)

    assert results["path_length"] > 0
    assert results["time_to_goal"] == 2e-6  # 2000 ns
    assert results["velocity_mean"] == 1.0

    assert results["collision_amount"] is None
    assert results["collisions"] is None
    assert results["result"] is None
    assert results["success"] is None
    assert results["num_pedestrians"] is None
    assert results["total_time_in_personal_space"] is None
    assert results["total_time_looking_at_pedestrians"] is None


def test_registry_skipping_with_all_topics():
    params = RobotParams(0.25, 0.0, 30.0)
    registry = MetricRegistry(params)

    df_all = pl.DataFrame(
        {
            "time_ns": [1000, 2000, 3000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "vel_linear": [1.0, 1.0, 1.0],
            "vel_angular": [0.0, 0.0, 0.0],
            "scan_ranges": [[10.0] * 10, [10.0] * 10, [10.0] * 10],
            "scan_min": [0.0, 0.0, 0.0],
            "peds_positions": [[], [], []],
            "peds_headings": [[], [], []],
            "num_pedestrians": [0, 0, 0],
            "collision_event": [0, 0, 0],
            "collision_monitor_state_action": ["none", "none", "none"],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df_all, start_pos=[0.0, 0.0, 0.0], goal_pos=[2.0, 0.0, 0.0])

    available_topics = {"odom", "scan", "peds", "collision_events", "collision_monitor_state"}
    results = registry.run(episode, pedsim_available=True, available_topics=available_topics)

    assert results["path_length"] > 0
    assert results["collision_amount"] == 0
    assert results["result"] == "GOAL_REACHED"
    assert results["success"] is True
    assert results["num_pedestrians"] == 0
    assert results["total_time_looking_at_pedestrians"] == 0.0
