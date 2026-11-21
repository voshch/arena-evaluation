from __future__ import annotations

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.performance.efficiency_metrics import (
    PathEfficiencyCalculator,
)
from arena_evaluation.processing.metrics.social.gaze import GazeMetricsCalculator
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import (
    PedestrianDisturbanceCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params


def test_efficiency_adversarial_zero_length() -> None:
    # Mathematically zero-length paths and near-zero floats
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.0, 0.0, 0.0],
    )
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
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=[],
        goal_pos=[],
        topics={"pedestrian_arena_peds": peds},
    )
    params = make_robot_params(robot_radius=0.2)
    calc = PedestrianDisturbanceCalculator(params)
    prior = {}

    # This should not raise ValueError or TypeError, and should not return NaN/Inf metrics
    results = calc.calculate(episode, prior)
    assert results["ped_path_deflection_m"] is None or np.isfinite(
        results["ped_path_deflection_m"]
    )
    assert not np.isnan(results["ped_velocity_delay_ratio"])
    assert results["ped_path_deflection_m"] is None or not np.isinf(
        results["ped_path_deflection_m"]
    )


def test_gaze_adversarial_chaotic() -> None:
    t_ns = [0, 1_000_000_000, 2_000_000_000]
    tf_df = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )
    peds = flat_peds(
        [
            (
                t_ns[0],
                [
                    {
                        "ped_id": "ped_0",
                        "x": float("nan"),
                        "y": float("nan"),
                        "yaw": float("nan"),
                    }
                ],
            ),
            (
                t_ns[1],
                [
                    {
                        "ped_id": "ped_0",
                        "x": float("inf"),
                        "y": float("inf"),
                        "yaw": float("inf"),
                    }
                ],
            ),
            (t_ns[2], [{"ped_id": "ped_0", "x": 1.0, "y": 1.0, "yaw": 0.0}]),
        ]
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=tf_df,
        start_pos=[],
        goal_pos=[],
        topics={"pedestrian_arena_peds": peds},
    )
    params = make_robot_params(robot_radius=0.2)
    calc = GazeMetricsCalculator(params)

    # Must run without crashing
    results = calc.calculate(episode, {})
    assert "total_time_looking_at_pedestrians" in results
