import pytest
import numpy as np
import polars as pl
from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.processing.metrics.social.gaze import GazeMetricsCalculator
from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator


def test_collision_adversarial_overlapping_hitboxes():
    # Overlapping hitboxes causing extreme or corrupt collision counts (NaN, Inf, massive ints)
    df = pl.DataFrame({"collision_event": [0.0, float('nan'), 1000.0, float('inf'), -1.0, 0.0], "action_type": [0, 0, 0, 0, 0, 0]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[0.0, 0.0, 0.0], goal_pos=[1.0, 1.0, 0.0])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = CollisionMetricsCalculator(params)
    prior = {"time_to_goal": 10.0}
    results = calc.calculate(episode, prior)

    # Should robustly count valid consecutive groups
    assert "collision_amount" in results
    assert not np.isnan(results["collision_amount"])
    assert not np.isinf(results["collision_amount"])


def test_efficiency_adversarial_zero_length():
    # Mathematically zero-length paths and near-zero floats
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0, 0.0], goal_pos=[0.0, 0.0, 0.0])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PathEfficiencyCalculator(params)

    prior = {"path_length": 0.0}
    results = calc.calculate(episode, prior)
    assert results["path_efficiency"] == 0.0

    # Near-zero (subnormal floats) that might explode during division
    prior = {"path_length": 1e-320}
    episode.start_pos = [0.0, 0.0, 0.0]
    episode.goal_pos = [1.0, 1.0, 0.0]
    results = calc.calculate(episode, prior)
    assert not np.isnan(results["path_efficiency"])

    episode.start_pos = [float('inf'), float('inf'), 0.0]
    episode.goal_pos = [float('nan'), float('nan'), 0.0]
    prior = {"path_length": 10.0}
    results = calc.calculate(episode, prior)
    assert not np.isnan(results["path_efficiency"])


def test_pedestrian_disturbance_adversarial_chaotic():
    # Infinite velocities, massive jumps, and NaNs
    df = pl.DataFrame({"time_ns": [0, 1, 2, 3, 4, 5, 6], "peds_positions": [[(0.0, 0.0)], [(1.0, 1.0)], [(float('inf'), float('inf'))], [(-float('inf'), 1e300)], [(float('nan'), float('nan'))], [(1.0, 1.0)], [(2.0, 2.0)]]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = PedestrianDisturbanceCalculator(params)
    prior = {}

    # This should not raise ValueError or TypeError, and should not return NaN/Inf metrics
    results = calc.calculate(episode, prior)
    assert not np.isnan(results["ped_path_deflection_m"])
    assert not np.isnan(results["ped_velocity_delay_ratio"])
    assert not np.isnan(results["ped_round_trips_completed"])
    assert not np.isinf(results["ped_path_deflection_m"])


def test_gaze_adversarial_chaotic():
    df = pl.DataFrame(
        {"time_ns": [0, 1000000000, 2000000000], "pos_x": [0.0, 0.0, 0.0], "pos_y": [0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0], "peds_positions": [[(float('nan'), float('nan'))], [(float('inf'), float('inf'))], [(1.0, 1.0)]], "peds_headings": [[float('nan')], [float('inf')], [0.0]], "num_pedestrians": [1, 1, 1]}
    )
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    params = RobotParams(0.2, 0.0, 10.0)
    calc = GazeMetricsCalculator(params)

    # Must run without crashing
    results = calc.calculate(episode, {})
    assert "time_looking_at_pedestrians" in results


def test_proxemics_adversarial_chaotic():
    df = pl.DataFrame({"time_ns": [0, 1000000000, 2000000000], "pos_x": [0.0, 0.0, 0.0], "pos_y": [0.0, 0.0, 0.0], "yaw": [0.0, 0.0, 0.0], "peds_positions": [[(0.0, 0.0)], [(float('inf'), float('inf'))], [(float('nan'), float('nan'))]], "num_pedestrians": [1, 1, 1]})
    episode = AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[])
    episode.num_pedestrians = 1
    params = RobotParams(0.2, 0.0, 10.0)
    calc = ProxemicsCalculator(params)

    prior = {"velocity": [1.0, float('inf'), float('nan')]}
    results = calc.calculate(episode, prior)

    assert not np.isnan(results["avg_velocity_in_personal_space"])
    assert not np.isinf(results["avg_velocity_in_personal_space"])
