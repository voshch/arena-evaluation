"""One cluster-bootstrap p-value: both entry points report the same shifted-null test."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.analytics.stats import (
    cluster_bootstrap_p_value,
    paired_treatment_diff,
)
from arena_evaluation.experiments.stats import cluster_bootstrap

_N_BOOT = 500


def _paired_frame(seed_count: int = 8) -> pl.DataFrame:
    """A balanced two-arm frame whose paired differences are exact integers per seed."""
    return pl.DataFrame(
        [
            {
                "world": "w",
                "scenario": 0,
                "crowd_size": "dense",
                "seed": seed,
                "planner": arm,
                "success": value,
            }
            for seed in range(seed_count)
            for arm, value in (("alpha", 3.0 + seed % 3), ("beta", 1.0))
        ]
    )


def test_both_entry_points_report_the_same_p_value_on_one_fixture() -> None:
    frame = _paired_frame()
    analytics = paired_treatment_diff(
        frame,
        "success",
        treatment_a="alpha",
        treatment_b="beta",
        n_boot=_N_BOOT,
        random_seed=42,
    )
    assert analytics.mean_diff is not None

    diffs = (
        frame.pivot(
            on="planner",
            index=["world", "scenario", "crowd_size", "seed"],
            values="success",
        )
        .with_columns(diff=pl.col("alpha") - pl.col("beta"))
        .sort("seed")["diff"]
        .to_list()
    )
    experiments = cluster_bootstrap(
        [np.array([diff]) for diff in diffs],
        n_boot=_N_BOOT,
        p_seed=42,
        rng=np.random.default_rng(0),
    )
    shared = cluster_bootstrap_p_value(
        diffs,
        [1.0] * len(diffs),
        observed=analytics.mean_diff,
        n_boot=_N_BOOT,
        random_seed=42,
    )

    assert analytics.p_value == experiments.p == shared


def test_p_value_is_calibrated_under_the_null_by_sign_permutation() -> None:
    """Flipping the arm labels of every pair keeps H0 true, so the p-values stay near-uniform."""
    rng = np.random.default_rng(7)
    observed_diffs = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5])
    null_p_values = []
    for _ in range(200):
        flipped = observed_diffs * rng.choice([-1.0, 1.0], size=observed_diffs.size)
        null_p_values.append(
            cluster_bootstrap_p_value(
                flipped,
                np.ones_like(flipped),
                observed=float(flipped.mean()),
                n_boot=200,
                random_seed=3,
            )
        )
    null_p = np.array(null_p_values)

    assert 0.35 <= float(null_p.mean()) <= 0.65
    assert float(np.mean(null_p < 0.05)) <= 0.2


def test_p_value_rejects_a_shift_that_the_null_cannot_produce() -> None:
    shifted = 1.0 + np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15])
    assert (
        cluster_bootstrap_p_value(
            shifted,
            np.ones_like(shifted),
            observed=float(shifted.mean()),
            n_boot=_N_BOOT,
            random_seed=42,
        )
        < 0.01
    )


def test_a_single_cluster_cannot_reject_null() -> None:
    """One cluster resamples to itself and cannot reject H0, so p == 1.0."""
    assert cluster_bootstrap_p_value([5.0], [3.0], observed=1.0, n_boot=10) == 1.0
    assert cluster_bootstrap_p_value([0.0], [3.0], observed=0.0, n_boot=10) == 1.0


def test_paired_treatment_diff_pairs_by_crowd_size_by_default() -> None:
    """Default pair_cols includes crowd_size so distinct crowd sizes at one seed remain separate pairs."""
    rows = [
        {
            "world": "w",
            "scenario": 0,
            "crowd_size": crowd,
            "seed": seed,
            "planner": arm,
            "success": value,
        }
        for crowd in ("sparse", "dense")
        for seed in range(4)
        for arm, value in (("alpha", 2.0), ("beta", 1.0))
    ]
    res = paired_treatment_diff(
        pl.DataFrame(rows),
        "success",
        treatment_a="alpha",
        treatment_b="beta",
        n_boot=50,
        random_seed=42,
    )
    assert res.n_pairs == 8
    assert res.n_clusters == 4
    assert res.mean_diff == pytest.approx(1.0)


def test_bootstrap_p_value_rejects_a_non_positive_draw_count() -> None:
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        cluster_bootstrap_p_value([1.0, 2.0], [1.0, 1.0], observed=1.5, n_boot=0)
