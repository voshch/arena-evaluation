"""Pin tests for analytics.stats edge cases and previously uncovered functions."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.analytics.stats import (
    CLUSTER_BOOTSTRAP_SCHEMA,
    KENDALL_RANKINGS_SCHEMA,
    RANK_SWAP_MATRIX_SCHEMA,
    cluster_bootstrap_ci,
    draw_cluster_indices,
    kendalls_w,
    paired_treatment_diff,
    rank_swap_matrix,
    split_discovery_verification_seeds,
)


def test_draw_cluster_indices_validation() -> None:
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="n_clusters must be >= 1"):
        draw_cluster_indices(0, 10, rng=rng)
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        draw_cluster_indices(3, 0, rng=rng)


def test_cluster_bootstrap_ci_validation_empty_ungrouped_and_single_cluster() -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb"] * 4 + ["teb"] * 2,
            "seed": [1, 2, 3, 4, 1, 1],
            "val": [1.0, 2.0, 3.0, 6.0, 5.0, 7.0],
        }
    )
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        cluster_bootstrap_ci(df, "val", n_boot=0)
    with pytest.raises(ValueError, match="ci_level must be in"):
        cluster_bootstrap_ci(df, "val", ci_level=1.0)

    empty = cluster_bootstrap_ci(df.head(0), "val", group_cols=("planner",))
    assert empty.is_empty()
    assert empty.schema == pl.Schema({"planner": pl.String, **CLUSTER_BOOTSTRAP_SCHEMA})

    grouped = cluster_bootstrap_ci(
        df, "val", group_cols=("planner",), n_boot=200, ci_level=0.90, random_seed=42
    ).sort("planner")
    dwb, teb = grouped.to_dicts()
    # dwb: 4 clusters [1, 2, 3, 6], mean=3.0; bootstrap quantiles/SE pinned at seed=42, n_boot=200
    assert dwb["n_obs"] == 4
    assert dwb["n_clusters"] == 4
    assert dwb["mean"] == 3.0
    assert dwb["std_error"] == pytest.approx(0.9283696941115234)
    assert dwb["ci_low"] == pytest.approx(1.75)
    assert dwb["ci_high"] == pytest.approx(4.762499999999996)
    assert dwb["basic_low"] == pytest.approx(1.2375000000000043)
    assert dwb["basic_high"] == pytest.approx(4.25)

    # teb: 1 cluster [5, 7], mean=6.0, zero spread, degenerate CI == mean
    assert teb["n_obs"] == 2
    assert teb["n_clusters"] == 1
    assert teb["mean"] == 6.0
    assert teb["std_error"] == 0.0
    assert teb["ci_low"] == 6.0
    assert teb["ci_high"] == 6.0
    assert teb["basic_low"] == 6.0
    assert teb["basic_high"] == 6.0

    ungrouped = cluster_bootstrap_ci(
        df, "val", group_cols=(), n_boot=200, ci_level=0.90, random_seed=42
    )
    row = ungrouped.to_dicts()[0]
    # ungrouped: 6 obs across 4 clusters, weighted mean (1+2+3+6+5+7)/6 = 4.0
    assert row["n_obs"] == 6
    assert row["n_clusters"] == 4
    assert row["mean"] == 4.0
    assert row["std_error"] == pytest.approx(0.6135771886481514)
    assert row["ci_low"] == pytest.approx(3.0)
    assert row["ci_high"] == pytest.approx(5.0)


def test_paired_treatment_diff_edge_cases() -> None:
    df = pl.DataFrame(
        {
            "world": ["w", "w", "w", "w"],
            "scenario": [0, 0, 1, 1],
            "crowd_size": ["s", "s", "s", "s"],
            "seed": [1, 1, 2, 2],
            "planner": ["a", "b", "a", "b"],
            "val": [3.0, 1.0, 3.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        paired_treatment_diff(df, "val", treatment_a="a", treatment_b="b", n_boot=0)
    with pytest.raises(ValueError, match="ci_level must be in"):
        paired_treatment_diff(df, "val", treatment_a="a", treatment_b="b", ci_level=0.0)
    with pytest.raises(ValueError, match="pair_cols must contain"):
        paired_treatment_diff(df, "val", treatment_a="a", treatment_b="b", pair_cols=())

    empty_res = paired_treatment_diff(
        df.filter(pl.col("planner") == "a"), "val", treatment_a="a", treatment_b="b"
    )
    assert empty_res.n_pairs == 0
    assert empty_res.n_clusters == 0
    assert empty_res.mean_diff is None
    assert empty_res.summary_frame.height == 1

    # A constant difference has no standardised effect size: dz is NULL with its reason (L3-034)
    const_res = paired_treatment_diff(
        df, "val", treatment_a="a", treatment_b="b", n_boot=100, random_seed=42
    )
    assert const_res.mean_diff == 2.0
    assert const_res.std_diff == 0.0
    assert const_res.cohens_dz is None
    assert (
        const_res.summary_frame.item(0, "cohens_dz_null_reason")
        == "std_diff:zero_denominator"
    )
    assert const_res.p_value == pytest.approx(1.0 / 101.0)

    # Constant zero diff: no spread either, so dz stays NULL instead of a fabricated 0.0
    zero_df = df.with_columns(pl.lit(1.0).alias("val"))
    zero_res = paired_treatment_diff(
        zero_df, "val", treatment_a="a", treatment_b="b", n_boot=100, random_seed=42
    )
    assert zero_res.cohens_dz is None

    # Single cluster: ci_low == ci_high == mean_diff, p_value == 1.0
    one_cluster = paired_treatment_diff(
        df.filter(pl.col("seed") == 1),
        "val",
        treatment_a="a",
        treatment_b="b",
        n_boot=100,
    )
    assert one_cluster.n_clusters == 1
    assert one_cluster.ci_low == 2.0
    assert one_cluster.ci_high == 2.0
    assert one_cluster.p_value == 1.0

    # Missing cluster_col in pair_cols falls back to one cluster per pair
    no_cluster = paired_treatment_diff(
        df,
        "val",
        treatment_a="a",
        treatment_b="b",
        cluster_col="missing",
        n_boot=100,
        random_seed=42,
    )
    assert no_cluster.n_clusters == 2
    assert no_cluster.p_value == pytest.approx(1.0 / 101.0)


def test_rank_swap_matrix_pin() -> None:
    df = pl.DataFrame(
        {
            "crowd_condition": ["blind"] * 3 + ["visible"] * 3 + ["mixed"] * 3,
            "planner": ["p1", "p2", "p3"] * 3,
            "score": [1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 1.0, 1.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="direction must be"):
        rank_swap_matrix(df, "score", direction="invalid")

    empty = rank_swap_matrix(df.head(0), "score")
    assert empty.is_empty()
    assert empty.schema == RANK_SWAP_MATRIX_SCHEMA

    single_regime = rank_swap_matrix(
        df.filter(pl.col("crowd_condition") == "blind"), "score"
    )
    assert single_regime.is_empty()
    assert single_regime.schema == RANK_SWAP_MATRIX_SCHEMA

    # Fewer than 2 common entities -> total_pairs=0, kendall_tau=1.0
    one_entity = rank_swap_matrix(df.filter(pl.col("planner") == "p1"), "score")
    assert one_entity.height == 3
    assert one_entity["total_pairs"].to_list() == [0, 0, 0]
    assert one_entity["kendall_tau"].to_list() == [1.0, 1.0, 1.0]

    # Exact pair comparisons across 3 planners:
    # (blind, mixed): 0 discordant, 2 concordant -> tau = 2/3
    # (blind, visible): 1 discordant, 2 concordant -> swap_rate = 1/3, tau = 1/3
    # (mixed, visible): 0 discordant, 2 concordant -> tau = 2/3
    res = rank_swap_matrix(df, "score", direction="asc").sort(["regime_a", "regime_b"])
    rows = res.to_dicts()
    assert [(r["regime_a"], r["regime_b"]) for r in rows] == [
        ("blind", "mixed"),
        ("blind", "visible"),
        ("mixed", "visible"),
    ]
    assert [r["swap_count"] for r in rows] == [0, 1, 0]
    assert [r["swap_rate"] for r in rows] == pytest.approx([0.0, 1.0 / 3.0, 0.0])
    assert [r["kendall_tau"] for r in rows] == pytest.approx(
        [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0]
    )


def test_kendalls_w_pin_all_df_branches() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        kendalls_w(
            pl.DataFrame({"crowd_condition": [], "planner": [], "score": []}),
            "score",
            direction="bad",
        )

    empty_kw = kendalls_w(
        pl.DataFrame(
            schema={
                "crowd_condition": pl.String,
                "planner": pl.String,
                "score": pl.Float64,
            }
        ),
        "score",
    )
    assert (empty_kw.n_judges, empty_kw.n_entities, empty_kw.w, empty_kw.p_value) == (
        0,
        0,
        None,
        None,
    )
    assert empty_kw.rankings_frame.schema == KENDALL_RANKINGS_SCHEMA

    # A single judge cannot be concordant with itself: W is undefined (L3-039)
    one_judge = kendalls_w(
        pl.DataFrame(
            {
                "crowd_condition": ["j1", "j1"],
                "planner": ["p1", "p2"],
                "score": [1.0, 2.0],
            }
        ),
        "score",
    )
    assert one_judge.n_judges == 1
    assert one_judge.n_entities == 2
    assert one_judge.w is None
    assert one_judge.p_value is None
    assert "at least two of each" in one_judge.note

    # Single entity -> m=2, k=1 -> concordance is undefined as well
    one_entity_kw = kendalls_w(
        pl.DataFrame(
            {
                "crowd_condition": ["j1", "j2"],
                "planner": ["p1", "p1"],
                "score": [1.0, 2.0],
            }
        ),
        "score",
    )
    assert (
        one_entity_kw.n_judges,
        one_entity_kw.n_entities,
        one_entity_kw.w,
        one_entity_kw.p_value,
    ) == (
        2,
        1,
        None,
        None,
    )

    # df == 1 (k=2, m=2, perfect agreement): W=1.0, chi2=2.0, p=erfc(1.0)
    kw2 = kendalls_w(
        pl.DataFrame(
            {
                "crowd_condition": ["j1", "j1", "j2", "j2"],
                "planner": ["p1", "p2", "p1", "p2"],
                "score": [1.0, 2.0, 1.0, 2.0],
            }
        ),
        "score",
        direction="asc",
    )
    assert kw2.df == 1
    assert kw2.w == pytest.approx(1.0)
    assert kw2.friedman_chi2 == pytest.approx(2.0)
    assert kw2.p_value == pytest.approx(math.erfc(1.0))

    # df == 2 (k=3, m=2): S=6, denom=96 -> W=0.75, chi2=3.0, p=exp(-1.5)
    kw3 = kendalls_w(
        pl.DataFrame(
            {
                "crowd_condition": ["j1"] * 3 + ["j2"] * 3,
                "planner": ["p1", "p2", "p3"] * 2,
                "score": [1.0, 2.0, 3.0, 1.0, 3.0, 2.0],
            }
        ),
        "score",
        direction="asc",
    )
    assert kw3.df == 2
    assert kw3.w == pytest.approx(0.75)
    assert kw3.friedman_chi2 == pytest.approx(3.0)
    assert kw3.p_value == pytest.approx(math.exp(-1.5))

    # df == 3 (k=4, m=3, one tie in j2): S=37.5, denom=522 -> W=25/29, chi2=225/29, Wilson-Hilferty p
    kw4 = kendalls_w(
        pl.DataFrame(
            {
                "crowd_condition": ["j1"] * 4 + ["j2"] * 4 + ["j3"] * 4,
                "planner": ["p1", "p2", "p3", "p4"] * 3,
                "score": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0, 2.0, 1.0, 3.0, 4.0],
            }
        ),
        "score",
        direction="asc",
    )
    assert kw4.df == 3
    assert kw4.w == pytest.approx(25.0 / 29.0)
    assert kw4.friedman_chi2 == pytest.approx(225.0 / 29.0)
    assert kw4.p_value == pytest.approx(0.05036691073802673)


def test_split_discovery_verification_seeds_validation() -> None:
    df = pl.DataFrame({"seed": [1, 2, 3, None]})
    with pytest.raises(ValueError, match="discovery_fraction must be in"):
        split_discovery_verification_seeds(df, discovery_fraction=1.0)
    out = split_discovery_verification_seeds(df, discovery_fraction=0.5)
    assert out["cohort"].to_list() == ["discovery", "discovery", "verification", None]


def test_kendalls_w_all_tied_single_judge_returns_p_one() -> None:
    df = pl.DataFrame(
        {
            "planner": ["a", "b", "c"],
            "crowd_condition": ["only", "only", "only"],
            "metric": [1.0, 1.0, 1.0],
        }
    )
    res = kendalls_w(df, "metric")
    assert res.friedman_chi2 == pytest.approx(0.0)
    assert res.p_value is None
