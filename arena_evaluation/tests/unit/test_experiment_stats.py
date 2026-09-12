from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.experiments.stats import cluster_bootstrap, holm, paired_differences, summarize_family


def test_holm_matches_the_textbook_example() -> None:
    # m = 4: 0.01*4, 0.02*3, 0.03*2, 0.04*1, made monotone
    assert holm([0.03, 0.01, 0.04, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])


def test_holm_skips_nan_and_caps_at_one() -> None:
    out = holm([0.5, float("nan"), 0.6])
    assert out[0] == pytest.approx(1.0) and out[2] == pytest.approx(1.0)
    assert np.isnan(out[1])


def test_bootstrap_excludes_zero_for_a_clear_shift() -> None:
    rng = np.random.default_rng(1)
    clusters = [rng.normal(1.0, 0.3, size=3) for _ in range(15)]
    res = cluster_bootstrap(clusters, n_boot=2000, rng=np.random.default_rng(2))
    assert res.ci_low > 0 and res.p < 0.01
    assert res.n_scenarios == 15 and res.n_pairs == 45


def test_bootstrap_of_a_null_contains_zero() -> None:
    clusters = [np.array([-1.0, 1.0, 0.0])] * 12
    res = cluster_bootstrap(clusters, n_boot=500)
    assert res.estimate == 0.0 and res.ci_low <= 0.0 <= res.ci_high and res.p == 1.0


def test_bootstrap_resamples_whole_scenarios() -> None:
    # one scenario carries all the signal: a row-level bootstrap would be far tighter
    clusters = [np.array([10.0, 10.0, 10.0])] + [np.zeros(3)] * 9
    res = cluster_bootstrap(clusters, n_boot=4000, rng=np.random.default_rng(0))
    assert res.estimate == pytest.approx(1.0)
    assert res.ci_low == 0.0  # a draw without the loud scenario is common with 10 clusters


def test_p_below_alpha_iff_interval_excludes_zero() -> None:
    rng = np.random.default_rng(3)
    for shift in (0.0, 0.2, 0.5, 1.0):
        res = cluster_bootstrap([rng.normal(shift, 1.0, size=3) for _ in range(12)], n_boot=1000, rng=rng)
        assert (res.p < 0.05) == (res.ci_low > 0 or res.ci_high < 0)


def test_paired_differences_and_family_summary() -> None:
    rows = []
    for planner in ("dwb", "mppi"):
        for scenario in range(6):
            for seed in range(3):
                rows.append({"planner": planner, "scenario": scenario, "seed": seed, "arm": "contact", "stalls": 3.0 if planner == "dwb" else 1.0})
                rows.append({"planner": planner, "scenario": scenario, "seed": seed, "arm": "locomotion", "stalls": 1.0})
    rows.append({"planner": "dwb", "scenario": 0, "seed": 9, "arm": "contact", "stalls": 5.0})  # unpaired: drops out
    df = pl.DataFrame(rows)
    diffs = paired_differences(df, cell=["planner"], metric="stalls", arm="arm", treatment="contact", control="locomotion")
    assert diffs.height == 36
    summary = summarize_family(diffs, cell=["planner"], n_boot=500).sort("planner")
    assert summary["estimate"].to_list() == pytest.approx([2.0, 0.0])
    assert summary["excludes_zero"].to_list() == [True, False]
    assert summary["p_holm"][0] <= 2 * summary["p"][0]
