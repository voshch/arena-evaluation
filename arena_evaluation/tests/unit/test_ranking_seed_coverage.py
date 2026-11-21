"""Seed coverage of the ranking products: a planner absent from a seed is reported, never scored 0.0 (L3-032)."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.analytics.ranking import (
    RANK_INTERVAL_SCHEMA,
    RANK_SPLITS_SCHEMA,
    RANK_STABILITY_SCHEMA,
    paired_cluster_bootstrap_rank_intervals,
    seed_split_rank_stability,
)

_COLUMNS = {
    "arm": pl.String,
    "world": pl.String,
    "scenario": pl.Int64,
    "crowd_size": pl.String,
    "planner": pl.String,
    "seed": pl.Int64,
    "success": pl.Float64,
    "time_to_goal": pl.Float64,
}


def _row(planner: str, seed: int, success: float, time_to_goal: float) -> dict[str, object]:
    return {
        "arm": "arm_main",
        "world": "hospital",
        "scenario": 0,
        "crowd_size": "dense",
        "planner": planner,
        "seed": seed,
        "success": success,
        "time_to_goal": time_to_goal,
    }


def _partial_coverage() -> pl.DataFrame:
    """Two planners on six shared seeds plus one planner evaluated on a single seed."""
    rows: list[dict[str, object]] = []
    for seed in range(1, 7):
        rows.append(_row("planner_a", seed, 0.9, 10.0))
        rows.append(_row("planner_b", seed, 0.5, 20.0))
    # A very good but sparsely evaluated planner: a fabricated 0.0 elsewhere would distort every split.
    rows.append(_row("planner_partial", 1, 1.0, 1.0))
    return pl.DataFrame(rows, schema=_COLUMNS)


def test_seed_split_stability_drops_a_half_without_observations() -> None:
    result = seed_split_rank_stability(_partial_coverage(), "success", n_splits=40, k=1, seed=7)

    # planner_partial has data on one seed only, so no split observes it in both halves.
    assert result.planners_ranked == 2
    assert result.seeds_ranked == 6
    assert result.partially_evaluated == ("planner_partial",)
    # The two complete planners never swap: agreement stays perfect instead of being dragged down by a
    # planner that would have scored 0.0 in every half it was missing from.
    assert result.mean_tau == pytest.approx(1.0)
    assert result.top_k_preservation_rate == pytest.approx(1.0)
    assert "planner_partial" in result.restriction
    assert result.splits_frame.schema == RANK_SPLITS_SCHEMA
    assert set(result.splits_frame.get_column("planners_ranked").to_list()) == {2}

    summary = result.to_frame()
    assert summary.schema == RANK_STABILITY_SCHEMA
    assert summary.item(0, "planners_ranked") == 2
    assert summary.item(0, "partially_evaluated") == "planner_partial"


def test_rank_intervals_leave_the_unrankable_planner_null() -> None:
    intervals = paired_cluster_bootstrap_rank_intervals(_partial_coverage(), "success", n_bootstrap=100, seed=11)

    assert intervals.schema == RANK_INTERVAL_SCHEMA
    by_planner = {
        row["planner"]: (
            row["median_rank"],
            row["ci_low"],
            row["ci_high"],
            row["seeds_ranked"],
        )
        for row in intervals.to_dicts()
    }
    # No fabricated best rank 1: the complete planners occupy ranks 1 and 2, the sparse one is NULL
    # while still reporting how many seeds it covers.
    assert by_planner == {
        "planner_a": (1.0, 1.0, 1.0, 6),
        "planner_b": (2.0, 2.0, 2.0, 6),
        "planner_partial": (None, None, None, 1),
    }
    # NULL ranks sort last, so a report does not present an unranked planner as the winner.
    assert intervals.get_column("planner").to_list()[-1] == "planner_partial"
