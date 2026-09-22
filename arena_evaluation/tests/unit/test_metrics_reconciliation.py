import numpy as np
import polars as pl
import pytest

from arena_evaluation.storage.schemas import RobotParams, AlignedEpisodeBundle
from arena_evaluation.processing.path.theta_star import GeometricThetaStar, compute_theta_star_path
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.metrics.naturalness.trajectory import TrajectoryMetricsCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import ProxemicsExtendedCalculator
from arena_evaluation.processing.metrics.holistic.holistic_metrics import HolisticMetricsCalculator


def test_geometric_theta_star_open_space():
    """Theta* in an empty grid must yield exact Euclidean line-of-sight path."""
    grid = np.zeros((100, 100), dtype=bool)
    start = (1.0, 1.0)
    goal = (4.0, 4.0)

    res = compute_theta_star_path(grid, start, goal, resolution=0.05, origin=(0.0, 0.0), robot_radius=0.1)
    assert res.success
    expected_dist = np.hypot(3.0, 3.0)
    assert abs(res.geodesic_length - expected_dist) < 1e-3
    assert len(res.path_x) == 2  # Straight 2-point segment (start, goal)


def test_geometric_theta_star_obstacle_detour():
    """Theta* around a center wall must navigate around corners with line-of-sight segments."""
    grid = np.zeros((100, 100), dtype=bool)
    # Put a vertical wall in the middle x=50, y from 20 to 80
    grid[20:80, 48:52] = True
    start = (1.0, 2.5)  # left of wall
    goal = (4.0, 2.5)  # right of wall

    solver = GeometricThetaStar(grid, resolution=0.05, origin=(0.0, 0.0), robot_radius=0.1)
    pts, length = solver.solve(start, goal)

    assert length > np.hypot(3.0, 0.0)  # Path must be longer than straight line
    assert len(pts) >= 3  # Start, corner waypoint(s), goal


def test_spl_and_path_efficiency():
    """SPL and Path Efficiency must correctly evaluate against Theta* reference length."""
    params = RobotParams(0.2, 0.0, 10.0)
    coll_calc = CollisionMetricsCalculator(params)
    eff_calc = PathEfficiencyCalculator(params)

    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"collision_event": [0, 0, 0]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0, 0.0],
    )

    # 1. Perfect shortest path on success
    prior = {"time_to_goal": 10.0, "path_length": 10.0, "theta_star_length": 10.0}
    res_coll = coll_calc.calculate(episode, prior)
    res_eff = eff_calc.calculate(episode, prior)
    assert res_coll["spl"] == 1.0
    assert res_eff["path_efficiency"] == 1.0

    # 2. Detour path (20m actual vs 10m optimal)
    prior_detour = {"time_to_goal": 20.0, "path_length": 20.0, "theta_star_length": 10.0}
    res_coll_detour = coll_calc.calculate(episode, prior_detour)
    res_eff_detour = eff_calc.calculate(episode, prior_detour)
    assert abs(res_coll_detour["spl"] - 0.5) < 1e-4
    assert abs(res_eff_detour["path_efficiency"] - 0.5) < 1e-4

    # 3. Collision failure (SPL must be 0.0)
    episode_fail = AlignedEpisodeBundle(
        episode_id=2,
        data=pl.DataFrame({"collision_event": [0, 1, 0, 1, 0, 1, 0, 1]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0, 0.0],
    )
    res_coll_fail = coll_calc.calculate(episode_fail, prior)
    assert res_coll_fail["success"] is False
    assert res_coll_fail["spl"] == 0.0


def test_psii_penetration_depth_monotonicity():
    """Deep penetration into personal space must yield strictly higher PSII penalty than shallow skimming."""
    params = RobotParams(0.3, 0.0, 10.0)  # r_robot = 0.3
    calc = ProxemicsExtendedCalculator(params)

    # Robot at origin (0, 0)
    # Ped 1: Deep penetration at (0.4, 0.0) -> d_center = 0.4, d_eff = 0.4 - (0.3 + 0.3) = -0.2 -> penetration depth = 1.2 - 0 = 1.2
    # Ped 2: Shallow skimming at (1.5, 0.0) -> d_center = 1.5, d_eff = 1.5 - 0.6 = 0.9 -> penetration depth = 1.2 - 0.9 = 0.3

    # Case A: Deep
    df_deep = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "peds_positions": ["[ [0.4, 0.0] ]", "[ [0.4, 0.0] ]"],
            "num_pedestrians": [1, 1],
        }
    )
    tf_df = pl.DataFrame({"time_ns": [0, 1_000_000_000], "pos_x_gt": [0.0, 0.0], "pos_y_gt": [0.0, 0.0], "yaw_gt": [0.0, 0.0]})
    ep_deep = AlignedEpisodeBundle(
        episode_id=1,
        data=tf_df,
        topics={
            "tf_gt": tf_df,
            "peds": df_deep,
        },
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0, 0.0],
    )

    # Case B: Shallow
    df_shallow = pl.DataFrame(
        {
            "time_ns": [0, 1_000_000_000],
            "peds_positions": ["[ [1.5, 0.0] ]", "[ [1.5, 0.0] ]"],
            "num_pedestrians": [1, 1],
        }
    )
    ep_shallow = AlignedEpisodeBundle(
        episode_id=2,
        data=tf_df,
        topics={
            "tf_gt": tf_df,
            "peds": df_shallow,
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


def test_holistic_effective_cot():
    """Effective Cost of Transport (E-CoT) must correctly penalize baseline CoT for collisions and PSII."""
    params = RobotParams(0.3, 0.0, 10.0)
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


def test_mutual_accommodation_reconciliation():
    """MutualAccommodationCalculator must correctly compute MAR, PFI, robot detour, and relative throughput."""
    from arena_evaluation.processing.metrics.social.mutual_accommodation import MutualAccommodationCalculator

    # Dynamic run: robot took a 1.0m lateral detour around ped
    dynamic_row = {
        "path": [[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]],
        "path_length": 10.2,
        "time_to_goal": 20.0,
        "energy_total_wh": 5.0,
        "pedestrian_path": [[[5.0, -0.5], [5.0, 0.0]]],
    }

    # Reference robot: straight line, took 10.0s, 3.0Wh
    ref_robot = {
        "path": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
        "path_length": 10.0,
        "time_to_goal": 10.0,
        "energy_total_wh": 3.0,
    }

    # Reference ped: unhindered straight line
    ref_ped = {
        "pedestrian_path": [[[5.0, 0.0], [5.0, 0.0]]],
    }

    reconciled = MutualAccommodationCalculator.reconcile_stage_references(
        dynamic_row=dynamic_row,
        ref_robot_row=ref_robot,
        ref_ped_row=ref_ped,
    )

    assert reconciled["relative_throughput"] == 0.5  # 10.0 / 20.0
    assert reconciled["kcsc_wh"] == 2.0  # 5.0 - 3.0
    assert reconciled["robot_path_deviation_m"] == 1.0  # 1.0m perpendicular offset
    assert reconciled["pfi"] is not None
    assert reconciled["mar"] is not None
