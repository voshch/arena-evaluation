from __future__ import annotations

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.metrics.social.gaze import GazeMetricsCalculator
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params


def test_collision_adversarial_overlapping_hitboxes() -> None:
    # Overlapping hitboxes causing extreme or corrupt collision counts (NaN, Inf, massive ints)
    events = pl.DataFrame(
        {
            "t_ns": list(range(6)),
            "event_count": [0.0, float("nan"), 1000.0, float("inf"), -1.0, 0.0],
            "wall_count": [0.0] * 6,
            "static_count": [0.0] * 6,
            "ped_count": [0.0] * 6,
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 1.0, 0.0],
        topics={"collision_events": events},
    )
    params = make_robot_params(robot_radius=0.2)
    calc = CollisionMetricsCalculator(params)
    prior = {"time_to_goal_s": 10.0}
    results = calc.calculate(episode, prior)

    # Should robustly count valid consecutive groups
    assert "collision_amount" in results
    assert not np.isnan(results["collision_amount"])
    assert not np.isinf(results["collision_amount"])


def test_efficiency_adversarial_zero_length() -> None:
    # Mathematically zero-length paths and near-zero floats
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0, 0.0], goal_pos=[0.0, 0.0, 0.0])
    params = make_robot_params(robot_radius=0.2)
    calc = PathEfficiencyCalculator(params)

    prior = {"path_length_m": 0.0, "theta_star_length": 5.0}
    results = calc.calculate(episode, prior)
    assert results["path_efficiency"] is None

    # Near-zero (subnormal floats) that might explode during division
    prior = {"path_length_m": 1e-320, "theta_star_length": 5.0}
    results = calc.calculate(episode, prior)
    assert results["path_efficiency"] is None

    # An unavailable geodesic reference is null, not a fabricated 0.0
    prior = {"path_length_m": 10.0}
    results = calc.calculate(episode, prior)
    assert results["path_efficiency"] is None


def test_pedestrian_disturbance_adversarial_chaotic() -> None:
    # Infinite velocities, massive jumps, and NaNs
    t_ns = [0, 1, 2, 3, 4, 5, 6]
    peds = pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * 7,
            "ped_id": ["ped_0"] * 7,
            "ped_name": ["ped_0"] * 7,
            "x": [0.0, 1.0, float("inf"), -float("inf"), float("nan"), 1.0, 2.0],
            "y": [0.0, 1.0, float("inf"), 1e300, float("nan"), 1.0, 2.0],
            "z": [0.0] * 7,
            "yaw": [0.0] * 7,
            "vx": [0.0] * 7,
            "vy": [0.0] * 7,
            "vz": [0.0] * 7,
            "speed_mps": [0.0] * 7,
            "radius_m": [0.35] * 7,
            "desired_speed": [1.0] * 7,
            "policy": [""] * 7,
            "agent_type": [""] * 7,
            "animation_state": [0] * 7,
        }
    )
    data = pl.DataFrame({"t_ns": t_ns})
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], topics={"pedestrian_tracks": peds})
    params = make_robot_params(robot_radius=0.2)
    calc = PedestrianDisturbanceCalculator(params)
    prior = {}

    # This should not raise ValueError or TypeError, and should not return NaN/Inf metrics
    results = calc.calculate(episode, prior)
    assert results["ped_path_deflection_m"] is None or np.isfinite(results["ped_path_deflection_m"])
    assert not np.isnan(results["ped_velocity_delay_ratio"])
    assert results["ped_path_deflection_m"] is None or not np.isinf(results["ped_path_deflection_m"])


def test_gaze_adversarial_chaotic() -> None:
    t_ns = [0, 1_000_000_000, 2_000_000_000]
    tf_df = pl.DataFrame({"t_ns": t_ns, "pos_x_gt": [0.0, 0.0, 0.0], "pos_y_gt": [0.0, 0.0, 0.0], "yaw_gt": [0.0, 0.0, 0.0]})
    peds = flat_peds(
        [
            (t_ns[0], [{"ped_id": "ped_0", "x": float("nan"), "y": float("nan"), "yaw": float("nan")}]),
            (t_ns[1], [{"ped_id": "ped_0", "x": float("inf"), "y": float("inf"), "yaw": float("inf")}]),
            (t_ns[2], [{"ped_id": "ped_0", "x": 1.0, "y": 1.0, "yaw": 0.0}]),
        ]
    )
    episode = AlignedEpisodeBundle(episode_id=1, data=tf_df, start_pos=[], goal_pos=[], topics={"pedestrian_tracks": peds})
    params = make_robot_params(robot_radius=0.2)
    calc = GazeMetricsCalculator(params)

    # Must run without crashing
    results = calc.calculate(episode, {})
    assert "total_time_looking_at_pedestrians" in results


def test_proxemics_adversarial_chaotic() -> None:
    t_ns = [0, 1_000_000_000, 2_000_000_000]
    df = pl.DataFrame(
        {
            "t_ns": t_ns,
            "pos_x": [0.0, 0.0, 0.0],
            "pos_y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pos_x_gt": [0.0, 0.0, 0.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
        }
    )
    peds = pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * 3,
            "ped_id": ["ped_0"] * 3,
            "ped_name": ["ped_0"] * 3,
            "x": [0.0, float("inf"), float("nan")],
            "y": [0.0, float("inf"), float("nan")],
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
    robot_tracks = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "pose_source": ["tf_gt"] * 3,
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=df,
        start_pos=[],
        goal_pos=[],
        topics={"pedestrian_tracks": peds, "robot_tracks": robot_tracks},
    )
    episode.num_pedestrians = 1
    params = make_robot_params(robot_radius=0.2)
    calc = ProxemicsCalculator(params)

    prior = {"velocity": [1.0, float("inf"), float("nan")]}
    results = calc.calculate(episode, prior)

    avg_velocity = results["avg_velocity_in_personal_space"]
    # The only in-personal-space sample is the first: its speed has no previous step, so the
    # removed zero fill leaves the mean undefined (NULL), never a fabricated 0.0 or NaN.
    assert avg_velocity is None or not np.isnan(avg_velocity)
    assert avg_velocity is None or not np.isinf(avg_velocity)
