from collections.abc import Mapping

import polars as pl
import pytest
from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params


def _robot_slice(episode: AlignedEpisodeBundle) -> ScopeSlice:
    """The ROBOT pass's slice of the whole episode, the shape scoped_evaluator hands it."""
    return ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot",
        t_start_ns=1000,
        t_end_ns=3000,
        identity={},
        window_bundle=episode,
        full_episode=episode,
    )


def _run_skipping(
    registry: MetricRegistry,
    episode: AlignedEpisodeBundle,
    available_topics: set[str],
) -> tuple[dict[str, object], Mapping[str, object]]:
    """The EPISODE and ROBOT results merged, and the ROBOT pass's child rows."""
    episode_outcome = registry.run_scope(MetricScope.EPISODE, episode, available_topics=available_topics, pedsim_available=True)
    robot_outcome = registry.run_scope(
        MetricScope.ROBOT,
        episode,
        available_topics=available_topics,
        pedsim_available=True,
        scope_slice=_robot_slice(episode),
    )
    return {**episode_outcome.results, **robot_outcome.results}, robot_outcome.children


def test_registry_skipping_with_available_topics():
    params = make_robot_params(robot_radius=0.25, laser_min_range=0.0, laser_max_range=30.0)
    registry = MetricRegistry(params)

    df_odom_only = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 1.0, 2.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
            "vel_linear": [1.0, 1.0, 1.0],
            "vel_angular": [0.0, 0.0, 0.0],
        }
    )

    episode = AlignedEpisodeBundle(episode_id=1, data=df_odom_only, start_pos=[0.0, 0.0, 0.0], goal_pos=[2.0, 0.0, 0.0])
    results, children = _run_skipping(registry, episode, {"robot_tracks"})

    assert results["path_length_m"] > 0
    # The sample at 2000 ns is exactly 1 m from the goal, which is within the 1 m tolerance.
    assert results["time_to_goal"] == pytest.approx(1e-6)  # 1000 ns after the first sample
    assert results["mean_velocity_mps"] == 1.0

    assert results["collision_amount"] is None
    assert children.get("collisions") is None
    assert results["success"] is None
    assert results["num_pedestrians"] is None
    assert results["total_time_in_personal_space"] is None
    assert results["total_time_looking_at_pedestrians"] is None


def test_registry_skipping_with_all_topics():
    params = make_robot_params(robot_radius=0.25, laser_min_range=0.0, laser_max_range=30.0)
    registry = MetricRegistry(params)

    df_all = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "pos_x": [0.0, 1.0, 2.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 1.0, 2.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
            "vel_linear": [1.0, 1.0, 1.0],
            "vel_angular": [0.0, 0.0, 0.0],
            "collision_event": [0, 0, 0],
            "collision_monitor_state_action": ["none", "none", "none"],
        }
    )
    peds = pl.DataFrame(
        {
            "t_ns": [1000, 2000, 3000],
            "env_id": ["env_0"] * 3,
            "ped_id": ["ped_0"] * 3,
            "ped_name": ["ped_0"] * 3,
            "x": [10.0, 10.0, 10.0],
            "y": [10.0, 10.0, 10.0],
            "z": [0.0] * 3,
            "yaw": [0.0] * 3,
            "vx": [0.0] * 3,
            "vy": [0.0] * 3,
            "vz": [0.0] * 3,
            "speed_mps": [0.0] * 3,
            "radius_m": [0.35] * 3,
            "desired_speed": [1.0] * 3,
            "policy": [""] * 3,
            "agent_type": [""] * 3,
            "animation_state": [0] * 3,
        }
    )

    collision_events = pl.DataFrame({"t_ns": [1000, 2000, 3000], "event_count": [0, 0, 0]})
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=df_all,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[2.0, 0.0, 0.0],
        topics={"pedestrian_tracks": peds, "collision_events": collision_events},
        outcome_state=2,
        outcome_info="",
    )
    results, _ = _run_skipping(registry, episode, {"robot_odom", "robot_tracks", "pedestrian_tracks", "collision_events", "collision_monitor_state"})

    assert results["path_length_m"] > 0
    assert results["collision_amount"] == 0
    assert results["success"] is True
    assert results["num_pedestrians"] == 0
    assert results["total_time_looking_at_pedestrians"] == 0.0
