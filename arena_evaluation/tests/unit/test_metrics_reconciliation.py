from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.holistic.holistic_metrics import HolisticMetricsCalculator
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import ProxemicsExtendedCalculator
from arena_evaluation.processing.path.theta_star import StaticGrid, plan_route
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from flat_peds import flat_peds

SEC = 1_000_000_000
PED_RADIUS_M = 0.3


def _peds(frames: Sequence[tuple[int, Sequence[Mapping[str, object]]]]) -> pl.DataFrame:
    """Flat pedestrian rows with each pedestrian's recorded body radius."""
    return flat_peds(frames).with_columns(pl.lit(PED_RADIUS_M, dtype=pl.Float64).alias("radius_m"))


def test_geometric_theta_star_open_space() -> None:
    """Theta* in an empty grid must yield exact Euclidean line-of-sight path."""
    grid = StaticGrid(origin_xy=(0.0, 0.0), resolution_m=0.05, lethal=np.zeros((100, 100), dtype=bool))

    res = plan_route(grid, (1.0, 1.0), (4.0, 4.0))
    assert res.null_reason is None
    assert abs(res.length_m - np.hypot(3.0, 3.0)) < 1e-3
    assert len(res.points) == 2  # Straight 2-point segment (start, goal)


def test_geometric_theta_star_obstacle_detour() -> None:
    """Theta* around a center wall must navigate around corners with line-of-sight segments."""
    lethal = np.zeros((100, 100), dtype=bool)
    # Put a vertical wall in the middle x=50, y from 20 to 80
    lethal[20:80, 48:52] = True
    start = (1.0, 2.5)  # left of wall
    goal = (4.0, 2.5)  # right of wall

    res = plan_route(StaticGrid(origin_xy=(0.0, 0.0), resolution_m=0.05, lethal=lethal), start, goal)

    assert res.null_reason is None
    assert res.length_m > np.hypot(3.0, 0.0)  # Path must be longer than straight line
    assert len(res.points) >= 3  # Start, corner waypoint(s), goal


def test_spl_and_path_efficiency() -> None:
    """SPL and Path Efficiency must correctly evaluate against Theta* reference length."""
    params = RobotParams(robot_radius=0.2)
    coll_calc = CollisionMetricsCalculator(params)
    eff_calc = PathEfficiencyCalculator(params)

    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0, 0.0],
        outcome_state=2,
        topics={"collision_events": pl.DataFrame({"t_ns": [0, 1, 2], "event_count": [0, 0, 0]})},
    )

    # 1. Perfect shortest path on success. The episode SPL is supplied by the evaluator and carried on the contact row.
    prior = {"time_to_goal_s": 10.0, "path_length_m": 10.0, "theta_star_length": 10.0, "spl": 1.0}
    res_coll = coll_calc.calculate(episode, prior)
    res_eff = eff_calc.calculate(episode, prior)
    assert res_coll["success"] is True
    assert res_eff["path_efficiency"] == 1.0

    # 2. Detour path (20m actual vs 10m optimal)
    prior_detour = {"time_to_goal_s": 20.0, "path_length_m": 20.0, "theta_star_length": 10.0, "spl": 0.5}
    res_coll_detour = coll_calc.calculate(episode, prior_detour)
    res_eff_detour = eff_calc.calculate(episode, prior_detour)
    assert res_coll_detour["success"] is True
    assert abs(res_eff_detour["path_efficiency"] - 0.5) < 1e-4

    # 3. Collision failure (SPL must be 0.0)
    episode_fail = AlignedEpisodeBundle(
        episode_id=2,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0, 0.0],
        outcome_state=3,
        topics={"collision_events": pl.DataFrame({"t_ns": [0, 1, 2, 3, 4, 5, 6, 7], "event_count": [0, 1, 0, 1, 0, 1, 0, 1]})},
    )
    prior_fail = {"time_to_goal_s": 10.0, "path_length_m": 10.0, "theta_star_length": 10.0, "spl": 0.0}
    res_coll_fail = coll_calc.calculate(episode_fail, prior_fail)
    assert res_coll_fail["success"] is False


def test_psii_penetration_depth_monotonicity() -> None:
    """Deep penetration into personal space must yield strictly higher PSII penalty than shallow skimming."""
    params = RobotParams(robot_radius=0.3)  # r_robot = 0.3
    calc = ProxemicsExtendedCalculator(params)

    # Robot at origin (0, 0)
    # Ped 1: Deep penetration at (0.4, 0.0) -> d_center = 0.4, d_eff = 0.4 - (0.3 + 0.3) = -0.2 -> penetration depth = 1.2 - 0 = 1.2
    # Ped 2: Shallow skimming at (1.5, 0.0) -> d_center = 1.5, d_eff = 1.5 - 0.6 = 0.9 -> penetration depth = 1.2 - 0.9 = 0.3

    t_ns = [0, 1_000_000_000]
    tf_df = pl.DataFrame({"t_ns": t_ns, "pos_x_gt": [0.0, 0.0], "pos_y_gt": [0.0, 0.0], "yaw_gt": [0.0, 0.0]})
    robot_tracks = pl.DataFrame({"t_ns": t_ns, "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "pose_source": ["tf_gt", "tf_gt"]})

    # Case A: Deep
    df_deep = _peds([(int(t), [{"ped_id": "ped_0", "x": 0.4, "y": 0.0}]) for t in t_ns])
    ep_deep = AlignedEpisodeBundle(
        episode_id=1,
        data=tf_df,
        topics={
            "robot_tracks": robot_tracks,
            "pedestrian_tracks": df_deep,
        },
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0, 0.0],
    )

    # Case B: Shallow
    df_shallow = _peds([(int(t), [{"ped_id": "ped_0", "x": 1.5, "y": 0.0}]) for t in t_ns])
    ep_shallow = AlignedEpisodeBundle(
        episode_id=2,
        data=tf_df,
        topics={
            "robot_tracks": robot_tracks,
            "pedestrian_tracks": df_shallow,
        },
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0, 0.0],
    )

    res_deep = calc.calculate(ep_deep, {})
    res_shallow = calc.calculate(ep_shallow, {})

    psii_deep = res_deep["personal_space_intrusion_integral"]
    psii_shallow = res_shallow["personal_space_intrusion_integral"]

    assert psii_deep is not None and psii_shallow is not None
    assert psii_deep > psii_shallow, f"Deep intrusion {psii_deep} should be strictly greater than shallow {psii_shallow}"


def test_holistic_effective_cot() -> None:
    """Effective Cost of Transport (E-CoT) must correctly penalize baseline CoT for collisions and PSII."""
    params = RobotParams(robot_radius=0.3)
    calc = HolisticMetricsCalculator(params)

    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[])

    # No collisions, no PSII
    prior_clean = {
        "specific_cost_of_transport": 1.5,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 0.0,
        "time_to_goal": 10.0,
    }
    res_clean = calc.calculate(episode, prior_clean)
    assert res_clean["e_cot"] == 1.5

    # 1 collision (+2.0 penalty) + PSII of 2.0s over 10s (+0.1 penalty) -> multiplier 3.1
    prior_penalized = {
        "specific_cost_of_transport": 1.5,
        "collision_amount": 1,
        "personal_space_intrusion_integral": 2.0,
        "time_to_goal": 10.0,
    }
    res_pen = calc.calculate(episode, prior_penalized)
    expected_e_cot = 1.5 * (1.0 + 2.0 * 1 + 0.5 * (2.0 / 10.0))  # 1.5 * 3.1 = 4.65
    assert abs(res_pen["e_cot"] - expected_e_cot) < 1e-4

