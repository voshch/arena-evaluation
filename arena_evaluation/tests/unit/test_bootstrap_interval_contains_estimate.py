"""Reported point estimates lie inside their own bootstrap intervals, including degenerate bootstraps whose replicates differ only by rounding."""

import polars as pl
from arena_evaluation.analytics.stats import cluster_bootstrap_ci, paired_treatment_diff


def _pair_frame(values_by_seed: list[list[float]]) -> pl.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for seed, values in enumerate(values_by_seed):
        for pair_id, value in enumerate(values):
            rows.append(
                {"planner": "A", "seed": seed, "pair_id": pair_id, "metric": value}
            )
            rows.append(
                {"planner": "B", "seed": seed, "pair_id": pair_id, "metric": 0.0}
            )
    return pl.DataFrame(rows)


def test_paired_point_estimate_inside_interval_for_identical_seeds() -> None:
    values_by_seed = [[5.535, 9.955, 7.927]] * 5
    res = paired_treatment_diff(
        _pair_frame(values_by_seed),
        "metric",
        treatment_col="planner",
        treatment_a="A",
        treatment_b="B",
        pair_cols=("pair_id", "seed"),
        cluster_col="seed",
        n_boot=2000,
    )
    assert res.ci_low <= res.mean_diff <= res.ci_high


def test_paired_point_estimate_inside_interval_for_unbalanced_seeds() -> None:
    values_by_seed = [[3.407] * n for n in (3, 6, 3, 6)]
    res = paired_treatment_diff(
        _pair_frame(values_by_seed),
        "metric",
        treatment_col="planner",
        treatment_a="A",
        treatment_b="B",
        pair_cols=("pair_id", "seed"),
        cluster_col="seed",
        n_boot=2000,
    )
    assert res.ci_low <= res.mean_diff <= res.ci_high


def test_cluster_point_estimate_inside_interval_for_unbalanced_seeds() -> None:
    values_by_seed = [[3.32] * n for n in (2, 3, 2, 1, 2, 6, 5, 3)]
    row = cluster_bootstrap_ci(
        _pair_frame(values_by_seed).filter(pl.col("planner") == "A"),
        "metric",
        group_cols=("planner",),
        cluster_col="seed",
        n_boot=2000,
    ).row(0, named=True)
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]
