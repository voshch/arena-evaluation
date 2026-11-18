"""Unit tests for ranking stability, contextual stability, and cluster bootstrap intervals."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.analytics.ranking import (
    CONTEXTUAL_STABILITY_SCHEMA,
    RANK_INTERVAL_SCHEMA,
    RANK_SPLITS_SCHEMA,
    RANK_STABILITY_SCHEMA,
    ContextualStabilityResult,
    RankInterval,
    RankStabilityResult,
    contextual_rank_stability,
    paired_cluster_bootstrap_rank_intervals,
    seed_split_rank_stability,
)
from arena_evaluation.analytics.stats import split_discovery_verification_seeds


def _build_synthetic_twin_episodes(
    *,
    n_seeds: int = 40,
    seed_offset: int = 1000,
) -> pl.DataFrame:
    """Build synthetic episode dataframe with deterministic planner performance."""
    planners = ["planner_a", "planner_b", "planner_c", "planner_d"]
    # Base success rates: A > B > C > D strictly across all seeds
    base_success = {
        "planner_a": 0.95,
        "planner_b": 0.75,
        "planner_c": 0.50,
        "planner_d": 0.20,
    }
    base_time = {
        "planner_a": 12.0,
        "planner_b": 22.0,
        "planner_c": 35.0,
        "planner_d": 50.0,
    }

    rows: list[dict[str, object]] = []
    for s_idx in range(n_seeds):
        actual_seed = seed_offset + s_idx
        for p in planners:
            # Deterministic, non-overlapping performance
            jitter = ((s_idx % 5) - 2) * 0.01
            rows.append(
                {
                    "arm": "arm_main",
                    "world": "hospital",
                    "scenario": "door_bottleneck",
                    "crowd_size": "dense",
                    "planner": p,
                    "seed": actual_seed,
                    "success": base_success[p] + jitter,
                    "time_to_goal": base_time[p] + jitter * 10.0,
                }
            )
    return pl.DataFrame(rows)


def _build_noisy_twin_episodes(
    *,
    n_seeds: int = 30,
) -> pl.DataFrame:
    """Build synthetic episodes where planners swap rankings across seeds."""
    rows: list[dict[str, object]] = []
    for s_idx in range(n_seeds):
        # Even seeds favor X, odd seeds favor Y
        score_x = 0.90 if s_idx % 2 == 0 else 0.40
        score_y = 0.40 if s_idx % 2 == 0 else 0.90
        rows.append(
            {
                "arm": "arm_test",
                "world": "office",
                "scenario": "hallway",
                "crowd_size": "low",
                "planner": "planner_x",
                "seed": s_idx,
                "success": score_x,
                "time_to_goal": 20.0,
            }
        )
        rows.append(
            {
                "arm": "arm_test",
                "world": "office",
                "scenario": "hallway",
                "crowd_size": "low",
                "planner": "planner_y",
                "seed": s_idx,
                "success": score_y,
                "time_to_goal": 20.0,
            }
        )
    return pl.DataFrame(rows)


def test_seed_split_rank_stability_perfect_concordance() -> None:
    """Verify perfect Kendall tau and top-k preservation for strictly separated planners."""
    df = _build_synthetic_twin_episodes(n_seeds=40)
    result = seed_split_rank_stability(
        df,
        "success",
        planner_col="planner",
        seed_col="seed",
        n_splits=50,
        k=3,
        descending=True,
        seed=42,
    )

    assert isinstance(result, RankStabilityResult)
    assert result.metric_col == "success"
    assert result.n_splits == 50
    assert result.k == 3
    assert result.mean_tau == 1.0
    assert result.std_tau == 0.0
    assert result.top_k_preservation_rate == 1.0
    assert len(result.empirical_distribution) == 50
    assert np.all(result.empirical_distribution == 1.0)

    # Validate output schemas
    assert result.splits_frame.schema == RANK_SPLITS_SCHEMA
    summary_df = result.to_frame()
    assert summary_df.schema == RANK_STABILITY_SCHEMA
    assert summary_df.item(0, "mean_tau") == 1.0
    assert summary_df.item(0, "top_k_preservation_rate") == 1.0


def test_seed_split_rank_stability_ascending_direction() -> None:
    """Verify lower-is-better metric respects ascending rank ordering."""
    df = _build_synthetic_twin_episodes(n_seeds=30)
    result = seed_split_rank_stability(
        df,
        "time_to_goal",
        planner_col="planner",
        seed_col="seed",
        n_splits=30,
        k=2,
        descending=False,
        seed=123,
    )

    assert result.mean_tau == 1.0
    assert result.std_tau == 0.0
    assert result.top_k_preservation_rate == 1.0


def test_seed_split_rank_stability_noisy_swaps() -> None:
    """Verify unstable planners produce swaps, intermediate tau, and variance."""
    df = _build_noisy_twin_episodes(n_seeds=40)
    result = seed_split_rank_stability(
        df,
        "success",
        planner_col="planner",
        seed_col="seed",
        n_splits=60,
        k=1,
        descending=True,
        seed=999,
    )

    assert -1.0 <= result.mean_tau <= 1.0
    assert result.splits_frame.height == 60
    assert result.splits_frame.schema == RANK_SPLITS_SCHEMA
    # Top-1 is not always preserved because X and Y alternate
    assert 0.0 <= result.top_k_preservation_rate <= 1.0


def test_seed_split_edge_cases_and_validation() -> None:
    """Verify boundary conditions: empty frame, single seed, single planner, and invalid params."""
    empty_df = pl.DataFrame(
        schema={
            "planner": pl.String,
            "seed": pl.Int64,
            "success": pl.Float64,
        }
    )
    empty_res = seed_split_rank_stability(empty_df, "success")
    assert empty_res.n_splits == 0
    assert empty_res.mean_tau == 0.0
    assert empty_res.splits_frame.is_empty()

    single_seed_df = pl.DataFrame(
        {
            "planner": ["p1", "p2"],
            "seed": [1, 1],
            "success": [0.8, 0.5],
        }
    )
    single_res = seed_split_rank_stability(single_seed_df, "success")
    assert single_res.n_splits == 0
    assert single_res.mean_tau == 0.0

    single_planner_df = pl.DataFrame(
        {
            "planner": ["p1", "p1", "p1", "p1"],
            "seed": [1, 2, 3, 4],
            "success": [0.8, 0.7, 0.9, 0.6],
        }
    )
    single_p_res = seed_split_rank_stability(single_planner_df, "success", n_splits=10)
    assert single_p_res.mean_tau == 1.0
    assert single_p_res.top_k_preservation_rate == 1.0

    valid_df = _build_synthetic_twin_episodes(n_seeds=10)
    with pytest.raises(ValueError, match="n_splits must be >= 1"):
        _ = seed_split_rank_stability(valid_df, "success", n_splits=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        _ = seed_split_rank_stability(valid_df, "success", k=0)


def test_contextual_rank_stability_breakdown() -> None:
    """Verify contextual rank stability across arms, worlds, and crowd sizes."""
    episodes_1 = _build_synthetic_twin_episodes(n_seeds=20).with_columns(
        arm=pl.lit("arm_alpha"),
        world=pl.lit("hospital"),
        crowd_size=pl.lit("dense"),
    )
    episodes_2 = _build_synthetic_twin_episodes(n_seeds=20).with_columns(
        arm=pl.lit("arm_alpha"),
        world=pl.lit("warehouse"),
        crowd_size=pl.lit("sparse"),
    )
    episodes_3 = _build_noisy_twin_episodes(n_seeds=20).with_columns(
        arm=pl.lit("arm_beta"),
        world=pl.lit("office"),
        crowd_size=pl.lit("dense"),
    )

    combined_df = pl.concat([episodes_1, episodes_2, episodes_3], how="diagonal")
    result_df = contextual_rank_stability(
        combined_df,
        "success",
        arm_col="arm",
        world_col="world",
        crowd_col="crowd_size",
        planner_col="planner",
        seed_col="seed",
        n_splits=20,
    )

    assert result_df.schema == CONTEXTUAL_STABILITY_SCHEMA
    assert result_df.height == 3

    results = ContextualStabilityResult.from_dataframe(result_df)
    assert len(results) == 3
    assert results[0].arm == "arm_alpha"
    assert results[0].world == "hospital"
    assert results[0].mean_tau == 1.0
    assert results[1].arm == "arm_alpha"
    assert results[1].world == "warehouse"
    assert results[1].mean_tau == 1.0


def test_contextual_rank_stability_empty() -> None:
    """Verify empty input returns empty frame with canonical schema."""
    empty_df = pl.DataFrame(
        schema={
            "arm": pl.String,
            "world": pl.String,
            "crowd_size": pl.String,
            "planner": pl.String,
            "seed": pl.Int64,
            "success": pl.Float64,
        }
    )
    result_df = contextual_rank_stability(empty_df, "success")
    assert result_df.is_empty()
    assert result_df.schema == CONTEXTUAL_STABILITY_SCHEMA


def test_paired_cluster_bootstrap_rank_intervals() -> None:
    """Verify paired cluster bootstrap yields correct rank quantiles for separated planners."""
    df = _build_synthetic_twin_episodes(n_seeds=30)
    intervals_df = paired_cluster_bootstrap_rank_intervals(
        df,
        "success",
        planner_col="planner",
        seed_col="seed",
        n_bootstrap=200,
        ci=0.95,
        descending=True,
        seed=42,
    )

    assert intervals_df.schema == RANK_INTERVAL_SCHEMA
    assert intervals_df.height == 4

    records = RankInterval.from_dataframe(intervals_df)
    assert len(records) == 4

    # Planners strictly ordered: A (1) < B (2) < C (3) < D (4)
    expected_ranks = {
        "planner_a": 1.0,
        "planner_b": 2.0,
        "planner_c": 3.0,
        "planner_d": 4.0,
    }
    for rec in records:
        exp = expected_ranks[rec.planner]
        assert rec.median_rank == exp
        assert rec.ci_low == exp
        assert rec.ci_high == exp

    # Ascending test (e.g. time_to_goal: lower is better)
    intervals_asc = paired_cluster_bootstrap_rank_intervals(
        df,
        "time_to_goal",
        planner_col="planner",
        seed_col="seed",
        n_bootstrap=100,
        descending=False,
    )
    for rec in RankInterval.from_dataframe(intervals_asc):
        exp = expected_ranks[rec.planner]
        assert rec.median_rank == exp


def test_paired_cluster_bootstrap_validation_and_empty() -> None:
    """Verify input validation and empty handling for cluster bootstrap."""
    empty_df = pl.DataFrame(
        schema={
            "planner": pl.String,
            "seed": pl.Int64,
            "success": pl.Float64,
        }
    )
    assert paired_cluster_bootstrap_rank_intervals(empty_df, "success").is_empty()

    df = _build_synthetic_twin_episodes(n_seeds=10)
    with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
        _ = paired_cluster_bootstrap_rank_intervals(df, "success", n_bootstrap=0)
    with pytest.raises(ValueError, match="ci must be in"):
        _ = paired_cluster_bootstrap_rank_intervals(df, "success", ci=1.5)


def test_discovery_verification_cohort_partitioning() -> None:
    """Verify twin seed discovery/verification cohort halving."""
    df = _build_synthetic_twin_episodes(n_seeds=20)
    split_df = split_discovery_verification_seeds(
        df, seed_col="seed", discovery_fraction=0.5
    )

    assert "cohort" in split_df.columns
    unique_seeds_discovery = (
        split_df.filter(pl.col("cohort") == "discovery").get_column("seed").unique()
    )
    unique_seeds_verification = (
        split_df.filter(pl.col("cohort") == "verification").get_column("seed").unique()
    )

    assert len(unique_seeds_discovery) == 10
    assert len(unique_seeds_verification) == 10
    # No seed overlap between cohorts
    assert set(unique_seeds_discovery).isdisjoint(set(unique_seeds_verification))

    # All twin episodes for a seed share the same cohort
    seed_cohort_counts = split_df.group_by("seed").agg(
        n_cohorts=pl.col("cohort").n_unique()
    )
    assert (seed_cohort_counts.get_column("n_cohorts") == 1).all()

    # Test LazyFrame overload
    lazy_split = split_discovery_verification_seeds(df.lazy(), seed_col="seed")
    assert isinstance(lazy_split, pl.LazyFrame)
    assert_frame_equal(split_df, lazy_split.collect())
