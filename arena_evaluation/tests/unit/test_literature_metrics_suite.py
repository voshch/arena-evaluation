"""
All-Encompassing Literature Metrics Test Suite for Arena Evaluation 3.0.

Verifies mathematical definitions and scientific specifications from foundational literature:
- Hall (1966) & Gao & Huang (2022): Concentric Proxemic Zones & Monotonic PSII
- Trautman & Krause (2010), Mavrogiannis et al. (2019/2022): Mutual Accommodation & DTW PFI
- Anderson et al. (2018): Success Weighted by Path Length (SPL) & Path Efficiency
- Helbing & Molnar (1995): Anisotropic Social Forces & Heading Potentials
- Bhattacharya et al. (2012): Any-Angle Geometric Theta* and Topological Winding Loops
- Selek / Albers / Master Spec: Multi-domain Energy, CoT, E-CoT, and Acoustic Personal Exposure
- Continental Traffic Rules: Corridor Passing Rule Compliance
"""

import numpy as np
import polars as pl
import pytest

from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams
from arena_evaluation.processing.path.theta_star import compute_theta_star_path
from arena_evaluation.processing.metrics.naturalness.trajectory import TrajectoryMetricsCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import ProxemicsExtendedCalculator
from arena_evaluation.processing.metrics.social.social_forces import SocialForcesCalculator
from arena_evaluation.processing.metrics.social.mutual_accommodation import MutualAccommodationCalculator
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.metrics.performance.efficiency_metrics import PathEfficiencyCalculator
from arena_evaluation.processing.metrics.performance.motion_metrics import MotionMetricsCalculator
from arena_evaluation.processing.metrics.ecological.compliance_metrics import ComplianceMetricsCalculator
from arena_evaluation.processing.metrics.holistic.holistic_metrics import HolisticMetricsCalculator

SEC = 1_000_000_000


# ==============================================================================
# 1. NATURALNESS & GEOMETRIC BASELINES (Theta*, ADE, FDE, MHD, Topology, PI)
# ==============================================================================


def test_literature_theta_star_any_angle_optimality():
    """Bhattacharya et al. (2012): Theta* must find Euclidean straight line in free space."""
    grid = np.zeros((50, 50), dtype=np.uint8)
    res = compute_theta_star_path(grid, resolution=0.1, origin=(0.0, 0.0, 0.0), start_pos=(1.0, 1.0), goal_pos=(4.0, 5.0), robot_radius=0.0)
    assert res.success
    expected_len = np.hypot(3.0, 4.0)  # 5.0m
    assert abs(res.geodesic_length - expected_len) < 1e-3
    assert len(res.path_x) == 2  # Exactly start and goal (any-angle shortcutting)


def test_literature_path_irregularity_straight_vs_weaving():
    """Morales et al. / Stulp et al.: Path Irregularity must be 0.0 for straight line, >0 for weaving."""
    calc = TrajectoryMetricsCalculator(RobotParams(robot_radius=0.25))

    # Case A: Straight line trajectory
    df_straight = pl.DataFrame(
        {
            "time_ns": [0, SEC, 2 * SEC],
            "pos_x_gt": [0.0, 5.0, 10.0],
            "pos_y_gt": [0.0, 0.0, 0.0],
            "yaw_gt": [0.0, 0.0, 0.0],
        }
    )
    ep_straight = AlignedEpisodeBundle(1, df_straight, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], topics={"tf_gt": df_straight})
    res_straight = calc.calculate(ep_straight, {})
    assert abs(res_straight["path_irregularity"]) < 1e-4

    # Case B: Weaving trajectory (zigzagging yaw changes)
    df_weave = pl.DataFrame(
        {
            "time_ns": [0, SEC, 2 * SEC, 3 * SEC, 4 * SEC],
            "pos_x_gt": [0.0, 2.5, 5.0, 7.5, 10.0],
            "pos_y_gt": [0.0, 1.0, 0.0, 1.0, 0.0],
            "yaw_gt": [0.38, -0.38, 0.38, -0.38, 0.0],
        }
    )
    ep_weave = AlignedEpisodeBundle(2, df_weave, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], topics={"tf_gt": df_weave})
    res_weave = calc.calculate(ep_weave, {})
    assert res_weave["path_irregularity"] > 0.1


def test_literature_topological_complexity_winding_loops():
    """Mavrogiannis et al. (2019): Winding number W = 1/(2*pi) * |sum Delta theta_rel|."""
    calc = TrajectoryMetricsCalculator(RobotParams(robot_radius=0.25))

    # Robot circles a stationary pedestrian at (5.0, 5.0) in a full 2*pi loop (radius 2.0m)
    angles = np.linspace(0, 2 * np.pi, 50)
    rx = 5.0 + 2.0 * np.cos(angles)
    ry = 5.0 + 2.0 * np.sin(angles)
    t_ns = np.linspace(0, 10 * SEC, 50, dtype=int)

    df_tf = pl.DataFrame({"time_ns": t_ns, "pos_x_gt": rx, "pos_y_gt": ry, "yaw_gt": angles})
    df_peds = pl.DataFrame({"time_ns": t_ns, "peds_positions": ["[ [5.0, 5.0] ]"] * 50, "num_pedestrians": [1] * 50})

    ep = AlignedEpisodeBundle(1, df_tf, [7.0, 5.0, 0.0], [7.0, 5.0, 0.0], topics={"tf_gt": df_tf, "peds": df_peds})
    res = calc.calculate(ep, {})

    assert res["topological_complexity"] is not None
    assert abs(res["topological_complexity"] - 1.0) < 0.05


# ==============================================================================
# 2. SOCIAL PROXEMICS, FORCES & MUTUAL ACCOMMODATION (Hall, Helbing, Trautman)
# ==============================================================================


def test_literature_proxemics_four_zones_and_monotonic_psii():
    """Hall (1966) & Gao & Huang (2022): Clearance classifies zones; PSII increases monotonically with intrusion depth."""
    calc = ProxemicsExtendedCalculator(RobotParams(robot_radius=0.35))

    tf_df = pl.DataFrame({"time_ns": [0, SEC], "pos_x_gt": [0.0, 0.0], "pos_y_gt": [0.0, 0.0], "yaw_gt": [0.0, 0.0]})

    # Case A: Intimate Intrusion
    peds_intimate = pl.DataFrame({"time_ns": [0, SEC], "peds_positions": ["[ [0.8, 0.0] ]", "[ [0.8, 0.0] ]"]})
    ep_a = AlignedEpisodeBundle(1, tf_df, [0, 0, 0], [0, 0, 0], topics={"tf_gt": tf_df, "peds": peds_intimate})
    res_a = calc.calculate(ep_a, {})

    # Case B: Personal Intrusion
    peds_personal = pl.DataFrame({"time_ns": [0, SEC], "peds_positions": ["[ [1.45, 0.0] ]", "[ [1.45, 0.0] ]"]})
    ep_b = AlignedEpisodeBundle(2, tf_df, [0, 0, 0], [0, 0, 0], topics={"tf_gt": tf_df, "peds": peds_personal})
    res_b = calc.calculate(ep_b, {})

    # Case C: Free / Public
    peds_free = pl.DataFrame({"time_ns": [0, SEC], "peds_positions": ["[ [4.0, 0.0] ]", "[ [4.0, 0.0] ]"]})
    ep_c = AlignedEpisodeBundle(3, tf_df, [0, 0, 0], [0, 0, 0], topics={"tf_gt": tf_df, "peds": peds_free})
    res_c = calc.calculate(ep_c, {})

    assert res_a["time_in_intimate_zone"] > 0.9
    assert res_b["time_in_personal_zone"] > 0.9
    assert res_c["time_in_intimate_zone"] == 0.0 and res_c["time_in_personal_zone"] == 0.0

    assert res_a["personal_space_intrusion_integral"] > res_b["personal_space_intrusion_integral"] > 0.0
    assert res_c["personal_space_intrusion_integral"] == 0.0


def test_literature_social_forces_anisotropic_potential():
    """Helbing & Molnar (1995): Pedestrian heading induces elliptical potential (higher in front, lower on sides)."""
    calc = SocialForcesCalculator(RobotParams(robot_radius=0.35))

    tf_front = pl.DataFrame({"time_ns": [0, SEC], "pos_x_gt": [1.0, 1.0], "pos_y_gt": [0.0, 0.0], "yaw_gt": [np.pi, np.pi]})
    tf_side = pl.DataFrame({"time_ns": [0, SEC], "pos_x_gt": [0.0, 0.0], "pos_y_gt": [1.0, 1.0], "yaw_gt": [-np.pi / 2, -np.pi / 2]})

    peds_df = pl.DataFrame(
        {
            "time_ns": [0, SEC],
            "peds_positions": ["[ [0.0, 0.0] ]", "[ [0.0, 0.0] ]"],
            "peds_headings": ["[ 0.0 ]", "[ 0.0 ]"],
        }
    )

    ep_front = AlignedEpisodeBundle(1, tf_front, [0, 0, 0], [0, 0, 0], topics={"tf_gt": tf_front, "peds": peds_df})
    ep_side = AlignedEpisodeBundle(2, tf_side, [0, 0, 0], [0, 0, 0], topics={"tf_gt": tf_side, "peds": peds_df})

    res_front = calc.calculate(ep_front, {})
    res_side = calc.calculate(ep_side, {})

    assert res_front["ci_mean"] > res_side["ci_mean"]


def test_literature_mutual_accommodation_ratio_burden_sharing():
    """Mavrogiannis et al. (2019/2022) & Master Spec: MAR = Delta d_ped / Delta d_robot."""
    # Reference robot & ped: straight 10m lines
    xs = np.linspace(0, 10, 10)
    ref_robot = {
        "path": [[float(x), 0.0] for x in xs],
        "path_length": 10.0,
        "time_to_goal": 10.0,
        "energy_total_wh": 3.0,
    }
    ref_ped = {
        "pedestrian_path": [[[float(x), 0.0] for x in xs]],
    }

    # Case 1: Aggressive robot stays straight (dev = 0.05m), forces ped to detour by 1.0m -> MAR >= 5.0
    agg_dynamic = {
        "path": [[float(x), 0.05] for x in xs],
        "path_length": 10.0,
        "time_to_goal": 10.0,
        "energy_total_wh": 3.0,
        "pedestrian_path": [[[float(x), 1.0] for x in xs]],
    }
    res_agg = MutualAccommodationCalculator.reconcile_stage_references(agg_dynamic, ref_robot, ref_ped)
    assert res_agg["mar"] is not None and res_agg["mar"] >= 5.0

    # Case 2: Cooperative robot yields by 0.5m, ped detours by 0.5m -> MAR approx 1.0
    coop_dynamic = {
        "path": [[float(x), 0.5] for x in xs],
        "path_length": 10.1,
        "time_to_goal": 12.0,
        "energy_total_wh": 3.5,
        "pedestrian_path": [[[float(x), 0.5] for x in xs]],
    }
    res_coop = MutualAccommodationCalculator.reconcile_stage_references(coop_dynamic, ref_robot, ref_ped)
    assert res_coop["mar"] is not None and abs(res_coop["mar"] - 1.0) < 0.2


# ==============================================================================
# 3. PERFORMANCE & EFFICIENCY (Anderson et al. SPL, Jerk, Path Efficiency)
# ==============================================================================


def test_literature_spl_and_path_efficiency():
    """Anderson et al. (2018): SPL = Success * (L_optimal / max(L_actual, L_optimal))."""
    calc_coll = CollisionMetricsCalculator(RobotParams(0.25))
    calc_eff = PathEfficiencyCalculator(RobotParams(0.25))

    ep_success = AlignedEpisodeBundle(1, pl.DataFrame(), [0, 0, 0], [10, 0, 0])

    # Case 1: Optimal traversal
    prior_opt = {"path_length": 10.0, "theta_star_length": 10.0, "time_to_goal": 10.0, "collision_amount": 0}
    res_opt_coll = calc_coll.calculate(ep_success, prior_opt)
    res_opt_eff = calc_eff.calculate(ep_success, prior_opt)
    assert res_opt_coll["spl"] == 1.0
    assert res_opt_eff["path_efficiency"] == 1.0

    # Case 2: Inefficient detour
    prior_detour = {"path_length": 20.0, "theta_star_length": 10.0, "time_to_goal": 20.0, "collision_amount": 0}
    res_detour_coll = calc_coll.calculate(ep_success, prior_detour)
    res_detour_eff = calc_eff.calculate(ep_success, prior_detour)
    assert res_detour_coll["spl"] == 0.5
    assert res_detour_eff["path_efficiency"] == 0.5

    # Case 3: Collision / Failure (Success = False)
    ep_fail = AlignedEpisodeBundle(1, pl.DataFrame(), [0, 0, 0], [10, 0, 0], outcome_info="collision")
    prior_fail = {"path_length": 10.0, "theta_star_length": 10.0, "time_to_goal": None}
    res_fail_coll = calc_coll.calculate(ep_fail, prior_fail)
    assert res_fail_coll["spl"] == 0.0


def test_literature_kinematic_jerk_smoothness():
    """Flash & Hogan / Kyriakopoulos: Jerk measures derivative of acceleration."""
    calc = MotionMetricsCalculator(RobotParams(0.25))

    t = np.linspace(0, 5, 50)
    t_ns = (t * SEC).astype(int)

    df_smooth = pl.DataFrame(
        {
            "time_ns": t_ns,
            "pos_x_gt": 0.25 * t**2,
            "pos_y_gt": np.zeros_like(t),
            "yaw_gt": np.zeros_like(t),
        }
    )
    ep_smooth = AlignedEpisodeBundle(1, df_smooth, [0, 0, 0], [10, 0, 0], topics={"tf_gt": df_smooth})
    res_smooth = calc.calculate(ep_smooth, {})

    assert res_smooth["jerk_mean"] is not None
    assert res_smooth["jerk_mean"] >= 0.0


# ==============================================================================
# 4. ECOLOGICAL, PASSING RULES & HOLISTIC (CoT, E-CoT, Right-Hand Passing)
# ==============================================================================


def test_literature_corridor_passing_rule_compliance():
    """Continental Traffic Rule: Head-on encounters require passing to the right of oncoming pedestrians."""
    calc = ComplianceMetricsCalculator(RobotParams(0.35))

    t_ns = np.array([0, SEC, 2 * SEC, 3 * SEC])
    ped_pos = ["[ [8.0, 0.0] ]", "[ [6.0, 0.0] ]", "[ [4.0, 0.0] ]", "[ [2.0, 0.0] ]"]
    peds_df = pl.DataFrame({"time_ns": t_ns, "peds_positions": ped_pos})

    # Case A: Robot moves in +X direction at y = -0.5 (to the right of oncoming ped -> Compliant)
    tf_right = pl.DataFrame({"time_ns": t_ns, "pos_x_gt": [1.0, 3.0, 5.0, 7.0], "pos_y_gt": [-0.5, -0.5, -0.5, -0.5], "yaw_gt": [0.0, 0.0, 0.0, 0.0]})
    ep_right = AlignedEpisodeBundle(1, tf_right, [0, 0, 0], [10, 0, 0], topics={"tf_gt": tf_right, "peds": peds_df})
    comp_right = calc._compute_passing_compliance(ep_right)
    assert comp_right == 1.0

    # Case B: Robot moves in +X direction at y = +0.5 (to the left of oncoming ped -> Non-compliant)
    tf_left = pl.DataFrame({"time_ns": t_ns, "pos_x_gt": [1.0, 3.0, 5.0, 7.0], "pos_y_gt": [0.5, 0.5, 0.5, 0.5], "yaw_gt": [0.0, 0.0, 0.0, 0.0]})
    ep_left = AlignedEpisodeBundle(2, tf_left, [0, 0, 0], [10, 0, 0], topics={"tf_gt": tf_left, "peds": peds_df})
    comp_left = calc._compute_passing_compliance(ep_left)
    assert comp_left == 0.0


def test_literature_holistic_effective_cost_of_transport():
    """Gao & Huang (2022) & Master Spec: E-CoT = CoT * (1.0 + 2.0*N_coll + 0.5*PSII/T)."""
    calc = HolisticMetricsCalculator(RobotParams(0.35, 0.0, 20.0))
    ep = AlignedEpisodeBundle(1, pl.DataFrame(), [0, 0, 0], [10, 0, 0])

    # Unimpeded run
    prior_clean = {"specific_cost_of_transport": 2.0, "collision_amount": 0, "personal_space_intrusion_integral": 0.0, "time_to_goal": 20.0}
    res_clean = calc.calculate(ep, prior_clean)
    assert res_clean["e_cot"] == 2.0

    # Penalized run
    prior_pen = {"specific_cost_of_transport": 2.0, "collision_amount": 1, "personal_space_intrusion_integral": 4.0, "time_to_goal": 20.0}
    res_pen = calc.calculate(ep, prior_pen)
    assert abs(res_pen["e_cot"] - 6.2) < 1e-4
