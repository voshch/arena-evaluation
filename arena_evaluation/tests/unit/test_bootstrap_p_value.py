"""One cluster-bootstrap p-value: both entry points report the same shifted-null test."""

from __future__ import annotations

import numpy as np
import polars as pl

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


