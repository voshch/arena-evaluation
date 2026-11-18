from __future__ import annotations

import math

import numpy as np
import polars as pl

from arena_evaluation.experiments.stats import (
    cluster_bootstrap,
    paired_differences,
    summarize_family,
)


def test_bootstrap_without_finite_pairs_is_undefined() -> None:
    for clusters in ([], [np.array([float("nan")])]):
        res = cluster_bootstrap(clusters, n_boot=10)
        assert math.isnan(res.estimate) and math.isnan(res.ci_low) and math.isnan(res.p)
        assert (res.n_scenarios, res.n_pairs) == (0, 0)


def test_bootstrap_keeps_the_estimate_when_every_draw_is_undefined() -> None:
    calls: list[int] = []

    def statistic(drawn: list[np.ndarray]) -> float:
        calls.append(len(drawn))
        return 1.0 if len(calls) == 1 else float("nan")

    res = cluster_bootstrap([np.ones(2), np.ones(3)], statistic, n_boot=10)
    assert res.estimate == 1.0
    assert math.isnan(res.ci_low) and math.isnan(res.ci_high) and math.isnan(res.p)
    assert (res.n_scenarios, res.n_pairs) == (2, 5)


def test_paired_differences_without_both_arms_keeps_the_key_schema() -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb"],
            "scenario": [0],
            "seed": [0],
            "arm": ["contact"],
            "stalls": [1.0],
        }
    )
    diffs = paired_differences(
        df,
        cell=["planner"],
        metric="stalls",
        arm="arm",
        treatment="contact",
        control="locomotion",
    )
    assert diffs.height == 0
    assert dict(diffs.schema) == {
        "planner": pl.String,
        "scenario": pl.Int64,
        "seed": pl.Int64,
        "diff": pl.Float64,
    }
    assert summarize_family(diffs, cell=["planner"], n_boot=10).height == 0
