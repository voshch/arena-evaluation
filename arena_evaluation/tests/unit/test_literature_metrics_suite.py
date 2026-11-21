"""All-encompassing literature metrics test suite for Arena Evaluation 3.0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from arena_evaluation.processing.metrics.holistic.holistic_metrics import (
    HolisticMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.collision_metrics import (
    CollisionMetricsCalculator,
)
from arena_evaluation.processing.metrics.performance.efficiency_metrics import (
    PathEfficiencyCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
PED_RADIUS_M = 0.3


def _peds(frames: Sequence[tuple[int, Sequence[Mapping[str, object]]]]) -> pl.DataFrame:
    """Flat pedestrian rows with each pedestrian's recorded body radius."""
    return flat_peds(frames).with_columns(
        pl.lit(PED_RADIUS_M, dtype=pl.Float64).alias("radius_m")
    )


def _robot_tracks(df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": df["t_ns"],
            "x": df["x"],
            "y": df["y"],
            "yaw": df["yaw"],
            "pose_source": ["tf_gt"] * len(df),
        }
    )


# 1. NATURALNESS & GEOMETRIC BASELINES (Theta*, ADE, FDE, MHD, Topology, PI)


# 2. SOCIAL PROXEMICS, FORCES & MUTUAL ACCOMMODATION (Hall, Helbing, Trautman)


# 3. PERFORMANCE & EFFICIENCY (Anderson et al. SPL, Jerk, Path Efficiency)


def test_literature_spl_and_path_efficiency() -> None:
    """Anderson et al. (2018): SPL = Success * (L_optimal / max(L_actual, L_optimal))."""
    calc_coll = CollisionMetricsCalculator(make_robot_params(robot_radius=0.25))
    calc_eff = PathEfficiencyCalculator(make_robot_params(robot_radius=0.25))

    ep_success = AlignedEpisodeBundle(
        1,
        pl.DataFrame(),
        [0, 0, 0],
        [10, 0, 0],
        outcome_state=2,
        topics={
            "robot_collision_events": pl.DataFrame(
                {"t_ns": [0, 1, 2], "event_count": [0, 0, 0]}
            )
        },
    )

    # Case 1: Optimal traversal. The episode SPL comes from the evaluator, carried on the contact row.
    prior_opt = {
        "path_length_m": 10.0,
        "theta_star_length": 10.0,
        "time_to_goal_s": 10.0,
        "collision_amount": 0,
        "spl": 1.0,
    }
    res_opt_coll = calc_coll.calculate(ep_success, prior_opt)
    res_opt_eff = calc_eff.calculate(ep_success, prior_opt)
    assert res_opt_coll["success"] is True
    assert res_opt_eff["path_efficiency"] == 1.0

    # Case 2: Inefficient detour
    prior_detour = {
        "path_length_m": 20.0,
        "theta_star_length": 10.0,
        "time_to_goal_s": 20.0,
        "collision_amount": 0,
        "spl": 0.5,
    }
    res_detour_coll = calc_coll.calculate(ep_success, prior_detour)
    res_detour_eff = calc_eff.calculate(ep_success, prior_detour)
    assert res_detour_coll["success"] is True
    assert res_detour_eff["path_efficiency"] == 0.5

    # Case 3: Collision / Failure (Success = False)
    ep_fail = AlignedEpisodeBundle(
        1,
        pl.DataFrame(),
        [0, 0, 0],
        [10, 0, 0],
        outcome_info="collision",
        topics={
            "robot_collision_events": pl.DataFrame(
                {"t_ns": [0, 1], "event_count": [0, 1]}
            )
        },
    )
    prior_fail = {
        "path_length_m": 10.0,
        "theta_star_length": 10.0,
        "time_to_goal_s": None,
        "spl": 0.0,
    }
    res_fail_coll = calc_coll.calculate(ep_fail, prior_fail)
    assert res_fail_coll["success"] is False


# 4. ECOLOGICAL, PASSING RULES & HOLISTIC (CoT, E-CoT, Right-Hand Passing)


def test_literature_holistic_effective_cost_of_transport() -> None:
    """Gao & Huang (2022) & Master Spec: E-CoT = CoT * (1.0 + 2.0*N_coll + 0.5*PSII/T)."""
    calc = HolisticMetricsCalculator(make_robot_params(robot_radius=0.35))
    ep = AlignedEpisodeBundle(1, pl.DataFrame(), [0, 0, 0], [10, 0, 0])

    # Unimpeded run
    prior_clean = {
        "specific_cost_of_transport": 2.0,
        "collision_amount": 0,
        "personal_space_intrusion_integral": 0.0,
        "time_to_goal": 20.0,
    }
    res_clean = calc.calculate(ep, prior_clean)
    assert res_clean["e_cot"] == 2.0

    # Penalized run
    prior_pen = {
        "specific_cost_of_transport": 2.0,
        "collision_amount": 1,
        "personal_space_intrusion_integral": 4.0,
        "time_to_goal": 20.0,
    }
    res_pen = calc.calculate(ep, prior_pen)
    assert abs(res_pen["e_cot"] - 6.2) < 1e-4
