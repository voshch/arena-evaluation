"""Unit tests for the named analysis procedures: pairing, selection, attribution, triage."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from synthetic_benchmark import SyntheticTruth, write_synthetic_benchmark

from arena_evaluation.analytics.analyses import run_procedure
from arena_evaluation.analytics.analyses.procedures import (
    ATTRIBUTION_SHARES_SCHEMA,
    LEAD_MEASURES_SCHEMA,
    RANKING_STABILITY_SCHEMA,
)
from arena_evaluation.analytics.catalog import BenchmarkCatalog


def _catalog(tmp_path: pathlib.Path, *, include_swap: bool = False) -> tuple[BenchmarkCatalog, SyntheticTruth]:
    run = tmp_path / "benchmarks" / ("syn_swap" if include_swap else "syn_noswap")
    truth = write_synthetic_benchmark(run, include_swap=include_swap)
    return BenchmarkCatalog.open(run), truth


def test_compare_reports_a_null_pairing_key_instead_of_silent_nulls(
    tmp_path: pathlib.Path,
) -> None:
    """An unrecorded crowd condition cannot form a pair; compare says so."""
    catalog, truth = _catalog(tmp_path)
    run = tmp_path / "benchmarks" / "syn_noswap"
    pedestrians = pl.read_parquet(run / "dim_pedestrians.parquet")
    pedestrians.with_columns(pl.lit(None, dtype=pl.Boolean).alias("robot_blind")).write_parquet(run / "dim_pedestrians.parquet")
    unrecorded = BenchmarkCatalog.open(run)
    assert unrecorded.get_table("v_episode_summary").select(pl.col("crowd_condition").null_count()).collect().item() == 128

    result = run_procedure(
        "compare",
        unrecorded,
        {
            "planner_a": truth.planted_diff_a,
            "planner_b": truth.planted_diff_b,
            "metric": truth.planted_diff_metric,
            "n_boot": 100,
        },
    )
    row = result.summary.row(0, named=True)
    assert row["n_pairs"] == 0
    assert row["reason"] == "no_complete_pair"
    assert "crowd_condition" in str(row["hint"])
    assert result.excluded_rows["crowd_condition:not_recorded"] == 64
    assert result.frame.is_empty()
    assert catalog is not unrecorded


def test_compare_pairs_across_two_crowd_sizes(tmp_path: pathlib.Path) -> None:
    """A cell with two crowd sizes yields twice the pairs, not one pooled cell."""
    catalog, truth = _catalog(tmp_path)
    params = {
        "planner_a": truth.planted_diff_a,
        "planner_b": truth.planted_diff_b,
        "metric": truth.planted_diff_metric,
        "n_boot": 100,
    }
    single = run_procedure("compare", catalog, params)
    assert "crowd_size" in single.summary.row(0, named=True)["pairing_keys"]

    run = tmp_path / "benchmarks" / "syn_noswap"
    episodes = pl.read_parquet(run / "dim_episodes.parquet")
    unrecorded = episodes.with_columns(pl.when(pl.col("seed") % 2 == 0).then(pl.lit(None, dtype=pl.Int64)).otherwise(pl.col("crowd_size")).alias("crowd_size"))
    unrecorded.write_parquet(run / "dim_episodes.parquet")
    halved = run_procedure("compare", BenchmarkCatalog.open(run), params)

    assert halved.excluded_rows["crowd_size:not_recorded"] == 32
    assert halved.summary.row(0, named=True)["n_pairs"] == (single.summary.row(0, named=True)["n_pairs"] // 2)


def test_lead_measures_selects_on_discovery_and_counts_verification_pairs(
    tmp_path: pathlib.Path,
) -> None:
    """Selection happens on discovery seeds; separated pairs are counted on verification."""
    catalog, _ = _catalog(tmp_path)
    result = run_procedure("lead_measures", catalog, {"n_boot": 100})

    assert result.frame.schema == LEAD_MEASURES_SCHEMA
    assert not any(isinstance(dtype, (pl.List, pl.Struct)) for dtype in result.frame.schema.values())
    row = result.frame.filter(pl.col("metric") == "time_to_goal").row(0, named=True)
    assert row["selected"] is True
    assert row["discovery_separation"] > row["discovery_seed_noise"]
    assert row["verification_separated_pairs"] >= 1
    assert row["verification_pairs_total"] == 6
    assert "planner_a[" in row["planner_intervals"]

    summary = result.summary.row(0, named=True)
    assert summary["discovery_seeds"] and summary["verification_seeds"]
    assert set(summary["discovery_seeds"]).isdisjoint(set(summary["verification_seeds"]))
    assert summary["discovery_seeds_csv"] == ",".join(str(seed) for seed in summary["discovery_seeds"])
    assert summary["n_selected_measures"] >= 1


def test_ranking_stability_returns_one_declared_table_for_any_world_count(
    tmp_path: pathlib.Path,
) -> None:
    """The frame schema does not depend on how many worlds the run contains."""
    catalog, _ = _catalog(tmp_path, include_swap=True)
    many = run_procedure(
        "ranking_stability",
        catalog,
        {"metric": "time_to_goal", "n_boot": 100, "n_splits": 20},
    )
    assert many.frame.schema == RANKING_STABILITY_SCHEMA
    assert {"contextual", "rank_interval"} <= set(many.frame["table"].to_list())
    summary = many.summary.row(0, named=True)
    assert summary["n_swaps"] >= 1
    assert summary["n_seed_splits"] == 20
    assert summary["kendalls_w"] is not None

    run = tmp_path / "benchmarks" / "syn_swap"
    episodes = pl.read_parquet(run / "dim_episodes.parquet")
    episodes.with_columns(pl.lit("world_1").alias("world")).write_parquet(run / "dim_episodes.parquet")
    single_world = run_procedure(
        "ranking_stability",
        BenchmarkCatalog.open(run),
        {"metric": "time_to_goal", "n_boot": 100, "n_splits": 20},
    )
    assert single_world.frame.schema == RANKING_STABILITY_SCHEMA
    assert single_world.summary.row(0, named=True)["kendalls_w"] is None


def test_visibility_effect_bootstrap_weighs_rows_by_their_drawn_seed(
    tmp_path: pathlib.Path,
) -> None:
    """Every seed replicate keeps the planted swap delta: a seed-count weight, never a per-row rescan."""
    catalog, _ = _catalog(tmp_path, include_swap=True)
    result = run_procedure("visibility_effect", catalog, {"metric": "time_to_goal", "n_boot": 100})

    row = result.summary.row(0, named=True)
    assert row["condition_swaps"] == 5
    assert row["seed_half_swaps"] == 0
    assert row["swap_delta"] == 5
    assert (row["ci_low"], row["ci_high"]) == (5.0, 5.0)
    assert row["effect_detected"] is True
    # No replicate delta reaches 0, so p = 1 / (n_boot + 1) exactly.
    assert row["p_value"] == pytest.approx(1.0 / 101.0)


def test_attribution_shares_use_pedestrian_contacts_only(
    tmp_path: pathlib.Path,
) -> None:
    """A wall contact cannot close in, so it never enters the closing share."""
    catalog, truth = _catalog(tmp_path)
    run = tmp_path / "benchmarks" / "syn_noswap"
    contacts = pl.read_parquet(run / "episodes" / "episode_000" / "metrics" / "fact_contacts.parquet")
    wall = pl.DataFrame(
        [
            {
                "run_id": "synthetic_truth",
                "episode_id": 0,
                "robot_id": "robot_0",
                "contact_id": "wall_0",
                "collider_kind": "wall",
                "v_close_robot_mps": 0.0,
                "v_close_target_mps": 1.0,
                "relative_impact_speed_mps": 1.0,
                "freeze_clearance_m": 0.1,
            }
        ],
        schema=contacts.schema,
    )
    pl.concat([contacts, wall], how="vertical_relaxed").write_parquet(run / "episodes" / "episode_000" / "metrics" / "fact_contacts.parquet")

    result = run_procedure("attribution_shares", BenchmarkCatalog.open(run), {"n_boot": 100})
    assert result.frame.schema == ATTRIBUTION_SHARES_SCHEMA
    planner_a = result.frame.filter(pl.col("planner") == "planner_a")
    robot_sum = float(planner_a["robot_closing_sum"].sum())
    target_sum = float(planner_a["target_closing_sum"].sum())
    assert robot_sum / (robot_sum + target_sum) == pytest.approx(truth.closer_shares["planner_a"]["robot_closing_share"])
    assert result.excluded_rows["non_pedestrian_contacts"] == 1
    assert set(result.summary["metric"].unique().to_list()) == {"robot_closing_share"}
    assert result.summary.schema["n_clusters"] == pl.Int64
    # The cluster unit is the seed, not the episode.
    assert result.summary["n_clusters"].max() <= 8


def test_triage_breaks_missingness_down_by_metric_and_reason(
    tmp_path: pathlib.Path,
) -> None:
    """Triage names the metric that is missing and why, per planner and world."""
    catalog, _ = _catalog(tmp_path)
    result = run_procedure("triage", catalog, {"metrics": ["ci_max", "time_to_goal"]})

    kinds = set(result.frame["row_kind"].unique().to_list())
    assert {"group", "metric_null"} <= kinds
    missing = result.frame.filter((pl.col("row_kind") == "metric_null") & (pl.col("metric") == "ci_max"))
    assert missing.height > 0
    assert set(missing["reason"].unique().to_list()) <= {
        "not_recorded",
        "no_samples",
    }
    assert result.summary.row(0, named=True)["metrics"] == 2
