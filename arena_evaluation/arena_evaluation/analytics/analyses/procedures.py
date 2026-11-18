"""Pure statistical analysis procedures built on BenchmarkCatalog and analytics.stats.

Each procedure returns one declared output table (``frame``) and one summary row per subject
(``summary``), so a consumer's column access never depends on what the run happens to contain.
Every procedure filters episodes and contacts the same way the catalogue does: references are dropped
unless asked for, ``is_valid_for_ranking`` is required, and a value that was never recorded stays NULL
with its reason counted in ``excluded_rows`` instead of becoming a zero.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import (
    MetricDirections,
    null_reason_counts,
    resolve_metric_directions,
)
from arena_evaluation.analytics.evidence import AnalysisResult, EvidenceStore
from arena_evaluation.analytics.ranking import (
    CONTEXTUAL_STABILITY_SCHEMA,
    contextual_rank_stability,
    paired_cluster_bootstrap_rank_intervals,
    seed_split_rank_stability,
)
from arena_evaluation.analytics.stats import (
    cluster_bootstrap_ci,
    contain_estimate,
    kendalls_w,
    paired_treatment_diff,
    rank_swap_matrix,
    split_discovery_verification_seeds,
)

__all__ = [
    "ATTRIBUTION_SHARES_SCHEMA",
    "LEAD_MEASURES_SCHEMA",
    "RANKING_STABILITY_SCHEMA",
    "TRIAGE_SCHEMA",
    "VISIBILITY_EFFECT_SCHEMA",
    "AttributionSharesParams",
    "CompareParams",
    "LeadMeasuresParams",
    "ProcedureSpec",
    "RankingStabilityParams",
    "TriageParams",
    "VisibilityEffectParams",
    "get_procedure",
    "list_procedures",
    "run_procedure",
]

#: Blocking keys a paired comparison must agree on: seeds are shared by arm, and world, scenario,
#: crowd size and crowd condition are the cell the difference is taken in (H3, ledger L3-09).
_PAIRING_KEYS: Final[tuple[str, ...]] = (
    "seed",
    "world",
    "scenario",
    "crowd_size",
    "crowd_condition",
)
_NULL_KEY_REASON: Final[str] = "not_recorded"
_REFERENCE_FILTER: Final[str] = "references"

TRIAGE_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "planner": pl.String,
        "world": pl.String,
        "crowd_condition": pl.String,
        "row_kind": pl.String,
        "metric": pl.String,
        "reason": pl.String,
        "episodes": pl.Int64,
        "total_episodes": pl.Int64,
        "valid_episodes": pl.Int64,
        "excluded_episodes": pl.Int64,
    }
)

RANKING_STABILITY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "table": pl.String,
        "arm": pl.String,
        "world": pl.String,
        "crowd_size": pl.String,
        "planner": pl.String,
        "metric": pl.String,
        "mean_tau": pl.Float64,
        "std_tau": pl.Float64,
        "top_k_preservation_rate": pl.Float64,
        "median_rank": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "seeds_ranked": pl.Int64,
    }
)

LEAD_MEASURES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "metric": pl.String,
        "discovery_separation": pl.Float64,
        "discovery_seed_noise": pl.Float64,
        "selected": pl.Boolean,
        "verification_separated_pairs": pl.Int64,
        "verification_pairs_total": pl.Int64,
        "planner_intervals": pl.String,
    }
)

VISIBILITY_EFFECT_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "metric": pl.String,
        "condition_swaps": pl.Int64,
        "seed_half_swaps": pl.Int64,
        "swap_delta": pl.Int64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "p_value": pl.Float64,
        "effect_detected": pl.Boolean,
        "n_boot": pl.Int64,
        "seed_half_split": pl.String,
    }
)

ATTRIBUTION_SHARES_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "planner": pl.String,
        "crowd_condition": pl.String,
        "n_contacts": pl.Int64,
        "robot_closing_sum": pl.Float64,
        "target_closing_sum": pl.Float64,
        "robot_closing_share": pl.Float64,
        "target_closing_share": pl.Float64,
        "mean_contact_robot_share": pl.Float64,
        "frozen_robot_share": pl.Float64,
        "freeze_clearance_mean_m": pl.Float64,
        "probst_R_robot_pooled": pl.Float64,
        "probst_R_actor2_pooled": pl.Float64,
        "probst_E_robot_pooled": pl.Float64,
        "probst_E_actor2_pooled": pl.Float64,
    }
)


class TriageParams(BaseModel):
    """Parameters for episode usability and missingness triage."""

    include_references: bool = False
    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Per-metric missingness to break down; empty takes every metric the metric dictionary "
            "publishes at episode or robot scope."
        ),
    )


class CompareParams(BaseModel):
    """Parameters for paired planner comparison on one metric."""

    planner_a: str
    planner_b: str
    metric: str = "time_to_goal"
    cluster_col: str = "seed"
    n_boot: int = Field(default=500, ge=50)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 42
    include_references: bool = False


class RankingStabilityParams(BaseModel):
    """Parameters for planner ranking repeatability across worlds, conditions, and seed halves."""

    metric: str = "time_to_goal"
    k: int = Field(default=3, ge=1)
    n_splits: int = Field(default=100, ge=1)
    n_boot: int = Field(default=500, ge=50)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 42
    include_references: bool = False


class VisibilityEffectParams(BaseModel):
    """Parameters for the H1 visibility effect: rank swaps across crowd conditions vs across seed halves."""

    metric: str = "time_to_goal"
    n_boot: int = Field(default=500, ge=50)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 42
    include_references: bool = False


class LeadMeasuresParams(BaseModel):
    """Parameters for discovery/verification seed-split lead measure selection."""

    metrics: list[str] = Field(
        default_factory=lambda: ["time_to_goal", "energy_total_wh", "path_efficiency"]
    )
    n_boot: int = Field(default=500, ge=50)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 42
    include_references: bool = False


class AttributionSharesParams(BaseModel):
    """Parameters for summed-parts pedestrian contact attribution shares."""

    n_boot: int = Field(default=500, ge=50)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    seed: int = 42
    include_references: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class ProcedureSpec:
    """Registered named analysis procedure."""

    name: str
    description: str
    params_model: type[BaseModel]
    fn: Callable[[BenchmarkCatalog, BaseModel], AnalysisResult]


# -- shared helpers ------------------------------------------------------------------------------


def _episodes(
    catalog: BenchmarkCatalog, *, include_references: bool
) -> tuple[pl.DataFrame, dict[str, int]]:
    """The catalogue's episode row per robot, references excluded on request; returns the frame and the exclusions."""
    frame = catalog.get_table("v_episode_summary").collect()
    excluded: dict[str, int] = {}
    if not include_references:
        before = frame.height
        frame = frame.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))
        if before > frame.height:
            excluded[_REFERENCE_FILTER] = before - frame.height
    return frame, excluded


def _valid_episodes(
    catalog: BenchmarkCatalog, *, include_references: bool
) -> tuple[pl.DataFrame, dict[str, int]]:
    """The usable episode rows: non-reference (unless asked) and flagged valid for ranking."""
    frame, excluded = _episodes(catalog, include_references=include_references)
    valid = frame.filter(
        pl.col("is_valid_for_ranking").fill_null(False).cast(pl.Boolean)
    )
    if frame.height > valid.height:
        excluded["invalid_for_ranking"] = frame.height - valid.height
    return valid, excluded


def _direction(catalog: BenchmarkCatalog, metric: str) -> MetricDirections:
    """Resolve one metric's better side, reporting where it came from on the result."""
    return resolve_metric_directions(catalog, (metric,))


def _descending(directions: MetricDirections, metric: str) -> bool:
    """True when a higher value of ``metric`` is the better one."""
    return directions.by_metric[metric] == "higher"


def _direction_text(directions: MetricDirections, metric: str) -> str:
    """The ``stats`` direction string (``desc`` for a higher-is-better metric)."""
    return "desc" if _descending(directions, metric) else "asc"


def _seed_half_frame(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """Split the sorted seeds into two deterministic halves and tag every row with its half."""
    seeds = tuple(int(s) for s in frame["seed"].drop_nulls().unique().sort())
    mid = len(seeds) // 2
    first, second = seeds[:mid], seeds[mid:]
    tagged = frame.with_columns(
        pl.when(pl.col("seed").is_in(list(first)))
        .then(pl.lit("h1"))
        .otherwise(pl.lit("h2"))
        .alias("__seed_half")
    )
    return tagged, first, second


def _swap_count(
    frame: pl.DataFrame,
    metric: str,
    *,
    regime_col: str,
    regime_a: object,
    regime_b: object,
    direction: str,
    entity_col: str = "planner",
) -> int:
    """Ranks entities in two regimes and returns how many pairs the two rankings swap."""
    matrix = rank_swap_matrix(
        frame,
        metric,
        entity_col=entity_col,
        regime_col=regime_col,
        direction=direction,
    )
    if matrix.is_empty():
        return 0
    # The swap matrix keys its rows by the regime pair it compares, so look the pair up either way.
    matched = matrix.filter(
        (
            (pl.col("regime_a").cast(pl.String) == str(regime_a))
            & (pl.col("regime_b").cast(pl.String) == str(regime_b))
        )
        | (
            (pl.col("regime_a").cast(pl.String) == str(regime_b))
            & (pl.col("regime_b").cast(pl.String) == str(regime_a))
        )
    )
    if matched.is_empty():
        return 0
    return int(matched["swap_count"].sum())


def _planner_means(frame: pl.DataFrame, metric: str) -> dict[str, float]:
    """Mean ``metric`` per planner over the rows that carry it."""
    grouped = (
        frame.filter(pl.col(metric).is_not_null())
        .group_by("planner")
        .agg(pl.col(metric).cast(pl.Float64).mean().alias("value"))
        .sort("planner")
    )
    return {
        str(name): float(value)
        for name, value in grouped.iter_rows()  # agent-contracts: ignore metadata dict boundary
        if value is not None
    }


def _planner_intervals(
    frame: pl.DataFrame,
    metric: str,
    *,
    n_boot: int,
    ci_level: float,
    seed: int,
    cluster_col: str = "seed",
) -> dict[str, tuple[float, float]]:
    """Per-planner seed-clustered interval of the metric mean: the seed spread D-12 asks for."""
    usable = frame.filter(
        pl.col(metric).is_not_null() & pl.col(cluster_col).is_not_null()
    )
    if usable.is_empty():
        return {}
    estimates = cluster_bootstrap_ci(
        usable,
        metric,
        group_cols=("planner",),
        cluster_col=cluster_col,
        n_boot=n_boot,
        ci_level=ci_level,
        random_seed=seed,
    )
    intervals: dict[str, tuple[float, float]] = {}
    # agent-contracts: ignore AC004 - one dict per planner at the string-interval boundary
    for row in estimates.iter_rows(named=True):
        low, high = row["ci_low"], row["ci_high"]
        if low is None or high is None:
            continue
        intervals[str(row["planner"])] = (float(low), float(high))
    return intervals


def _separated_pairs(
    intervals: Mapping[str, tuple[float, float]], *, higher_is_better: bool
) -> int:
    """Count planner pairs whose seed intervals do not overlap in the better direction."""
    names = sorted(intervals)
    separated = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            low_left, high_left = intervals[left]
            low_right, high_right = intervals[right]
            if higher_is_better:
                separated += int(low_left > high_right or low_right > high_left)
            else:
                separated += int(high_left < low_right or high_right < low_left)
    return separated


def _format_intervals(intervals: Mapping[str, tuple[float, float]]) -> str:
    """Render per-planner intervals as one flat ``name[low,high]`` CSV string."""
    return ";".join(
        f"{name}[{low:.6g},{high:.6g}]"
        for name, (low, high) in sorted(intervals.items())
    )


def _dictionary_metrics(catalog: BenchmarkCatalog) -> list[str]:
    """Every episode- or robot-scope metric the published dictionary lists, used for the missingness sweep."""
    published = (
        catalog.metric_dictionary()
        .filter(pl.col("scope").is_in(["episodes", "robots"]))
        .select("metric")
        .unique()
        .sort("metric")
        .collect()
    )
    return [str(name) for name in published.get_column("metric")]


def _metric_null_rows(
    usable: pl.DataFrame,
    metric: str,
    group_cols: list[str],
    reason_columns: list[str],
) -> pl.DataFrame:
    """One ``metric_null`` row per group and reason of the metric's NULL cells, plus the unreasoned rest."""
    nulls = usable.filter(pl.col(metric).is_null())
    if nulls.is_empty():
        return pl.DataFrame(schema=TRIAGE_SCHEMA)
    reason_text = pl.concat_str(
        [pl.col(name).fill_null("") for name in reason_columns], separator=";"
    )
    tokens = (
        nulls.select([*group_cols, reason_text.alias("__tokens")])
        .with_columns(pl.col("__tokens").str.split(";").alias("reason"))
        .explode("reason", empty_as_null=True)
        .filter(pl.col("reason").str.starts_with(f"{metric}:"))
        .with_columns(pl.col("reason").str.slice(len(metric) + 1).alias("reason"))
        .group_by([*group_cols, "reason"])
        .agg(pl.len().cast(pl.Int64).alias("episodes"))
    )
    attributed = tokens.group_by(group_cols).agg(
        pl.col("episodes").sum().alias("__attributed")
    )
    remainder = (
        nulls.group_by(group_cols)
        .agg(pl.len().cast(pl.Int64).alias("null_episodes"))
        .join(attributed, on=group_cols, how="left")
        .with_columns(
            (pl.col("null_episodes") - pl.col("__attributed").fill_null(0)).alias(
                "episodes"
            )
        )
        .filter(pl.col("episodes") > 0)
        .with_columns(pl.lit(_NULL_KEY_REASON, dtype=pl.String).alias("reason"))
    )
    return (
        pl.concat(
            [
                tokens.select([*group_cols, "reason", "episodes"]),
                remainder.select([*group_cols, "reason", "episodes"]),
            ],
            how="vertical_relaxed",
        )
        .with_columns(
            pl.lit("metric_null", dtype=pl.String).alias("row_kind"),
            pl.lit(metric, dtype=pl.String).alias("metric"),
            pl.lit(None, dtype=pl.Int64).alias("total_episodes"),
            pl.lit(None, dtype=pl.Int64).alias("valid_episodes"),
            pl.lit(None, dtype=pl.Int64).alias("excluded_episodes"),
        )
        .select(list(TRIAGE_SCHEMA.names()))
        .sort(group_cols + ["reason"])
    )


def _pairing_keys(frame: pl.DataFrame) -> list[str]:
    """The blocking keys this frame actually carries, in the declared order."""
    return [key for key in _PAIRING_KEYS if key in frame.columns]


# -- triage -------------------------------------------------------------------------------------


def _run_triage(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, TriageParams)
    usable, excluded = _valid_episodes(
        catalog, include_references=params.include_references
    )
    metrics = params.metrics or _dictionary_metrics(catalog)
    group_cols = ["planner", "world", "crowd_condition"]
    grouped = usable.group_by(group_cols)
    counts = grouped.agg(
        pl.len().cast(pl.Int64).alias("total_episodes"),
        pl.col("is_valid_for_ranking")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("valid_episodes"),
        (~pl.col("is_valid_for_ranking").fill_null(False))
        .sum()
        .cast(pl.Int64)
        .alias("excluded_episodes"),
    )

    # One flat frame per row class, concatenated in Polars: a columnar result is never walked in Python.
    parts: list[pl.DataFrame] = [
        counts.sort(group_cols)
        .with_columns(
            pl.lit("group", dtype=pl.String).alias("row_kind"),
            pl.lit(None, dtype=pl.String).alias("metric"),
            pl.lit(None, dtype=pl.String).alias("reason"),
            pl.col("valid_episodes").alias("episodes"),
        )
        .select(list(TRIAGE_SCHEMA.names()))
    ]

    # An invalid episode's own reason, per group: the "why is this run smaller" answer.
    invalid_reasons = (
        usable.filter(pl.col("invalid_reason").is_not_null())
        .select([*group_cols, "invalid_reason"])
        .with_columns(pl.col("invalid_reason").str.split(";").alias("reason"))
        .explode("reason", empty_as_null=True)
        .filter(pl.col("reason").is_not_null())
        .group_by([*group_cols, "reason"])
        .agg(pl.len().cast(pl.Int64).alias("episodes"))
    )
    parts.append(
        invalid_reasons.with_columns(
            pl.lit("invalid_reason", dtype=pl.String).alias("row_kind"),
            pl.lit(None, dtype=pl.String).alias("metric"),
            pl.lit(None, dtype=pl.Int64).alias("total_episodes"),
            pl.lit(None, dtype=pl.Int64).alias("valid_episodes"),
            pl.lit(None, dtype=pl.Int64).alias("excluded_episodes"),
        )
        .select(list(TRIAGE_SCHEMA.names()))
        .sort(group_cols + ["reason"])
    )

    # Each metric's NULL cells by the reason the row carries, per group.
    reason_columns = [name for name in usable.columns if name.endswith("_null_reason")]
    parts.extend(
        part
        for metric in metrics
        if not (
            part := _metric_null_rows(usable, metric, group_cols, reason_columns)
        ).is_empty()
    )

    frame = pl.concat(
        [part for part in parts if not part.is_empty()], how="vertical_relaxed"
    ).select([pl.col(name).cast(dtype) for name, dtype in TRIAGE_SCHEMA.items()])
    summary = pl.DataFrame(
        [
            {
                "total_episodes": usable.height,
                "valid_episodes": int(
                    usable.filter(
                        pl.col("is_valid_for_ranking").fill_null(False)
                    ).height
                ),
                "excluded_episodes": excluded.get("invalid_for_ranking", 0),
                "metrics": len(metrics),
            }
        ]
    )
    return AnalysisResult(
        frame=frame,
        summary=summary,
        excluded_rows=excluded,
        snapshot_id=catalog.snapshot_id,
    )


# -- compare ------------------------------------------------------------------------------------


def _run_compare(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, CompareParams)
    usable, excluded = _valid_episodes(
        catalog, include_references=params.include_references
    )
    if params.metric not in usable.columns:
        return AnalysisResult(
            frame=pl.DataFrame(),
            summary=pl.DataFrame(
                [
                    {
                        "reason": f"unknown metric {params.metric!r}",
                        "hint": "pick a metric the metric dictionary publishes",
                    }
                ]
            ),
            excluded_rows={f"{params.metric}:unknown": usable.height},
            snapshot_id=catalog.snapshot_id,
        )

    sub = usable.filter(pl.col("planner").is_in([params.planner_a, params.planner_b]))
    excluded.update(
        null_reason_counts(
            sub,
            [params.metric],
            [name for name in sub.columns if name.endswith("_null_reason")],
        )
    )
    valid = sub.filter(pl.col(params.metric).is_not_null())
    pairing = _pairing_keys(valid)
    if not pairing:
        return AnalysisResult(
            frame=pl.DataFrame(),
            summary=pl.DataFrame(
                [
                    {
                        "planner_a": params.planner_a,
                        "planner_b": params.planner_b,
                        "metric": params.metric,
                        "n_pairs": 0,
                        "reason": "no_pairing_key",
                        "hint": (
                            "the episode view carries none of "
                            f"{list(_PAIRING_KEYS)}; pair on at least one blocking column"
                        ),
                    }
                ]
            ),
            excluded_rows={f"{params.metric}:no_pairing_key": valid.height},
            snapshot_id=catalog.snapshot_id,
        )

    # A pairing key the benchmark never recorded cannot form a pair; say so instead of returning NULL.
    key_complete = valid
    null_keys: list[str] = []
    for key in pairing:
        missing = valid.filter(pl.col(key).is_null())
        if missing.height:
            null_keys.append(key)
            excluded[f"{key}:{_NULL_KEY_REASON}"] = missing.height
            key_complete = key_complete.filter(pl.col(key).is_not_null())

    if valid.height == 0:
        return AnalysisResult(
            frame=pl.DataFrame(),
            summary=pl.DataFrame(
                [
                    {
                        "planner_a": params.planner_a,
                        "planner_b": params.planner_b,
                        "metric": params.metric,
                        "n_pairs": 0,
                        "reason": "all_null",
                        "hint": f"{params.metric} is NULL for both planners; see excluded_rows",
                    }
                ]
            ),
            excluded_rows=excluded or {f"{params.metric}:all_null": sub.height},
            snapshot_id=catalog.snapshot_id,
        )

    cluster_col = params.cluster_col if params.cluster_col in valid.columns else "seed"
    res = paired_treatment_diff(
        key_complete,
        params.metric,
        treatment_col="planner",
        treatment_a=params.planner_a,
        treatment_b=params.planner_b,
        pair_cols=pairing,
        cluster_col=cluster_col,
        n_boot=params.n_boot,
        ci_level=params.ci_level,
        random_seed=params.seed,
    )
    hint = ""
    if res.n_pairs == 0:
        hint = "no complete pair: " + (
            f"the pairing keys {null_keys} are unrecorded"
            if null_keys
            else f"the two planners share no cell on {pairing}"
        )
    elif null_keys:
        hint = f"{len(null_keys)} pairing key(s) had unrecorded rows: {null_keys}"
    summary = pl.DataFrame(
        [
            {
                "planner_a": params.planner_a,
                "planner_b": params.planner_b,
                "metric": params.metric,
                "mean_diff": res.mean_diff,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
                "n_pairs": res.n_pairs,
                "n_clusters": res.n_clusters,
                "cohens_dz": res.cohens_dz,
                "pairing_keys": ",".join(pairing),
                "reason": "no_complete_pair" if res.n_pairs == 0 else None,
                "hint": hint or None,
            }
        ]
    )
    seeds = tuple(int(s) for s in valid["seed"].drop_nulls().unique().sort())
    frame = (
        res.pairs_frame.with_columns(
            pl.lit(params.planner_a).alias("planner_a"),
            pl.lit(params.planner_b).alias("planner_b"),
            pl.lit(params.metric).alias("metric"),
        ).select(
            [
                "planner_a",
                "planner_b",
                "metric",
                "value_a",
                "value_b",
                "diff",
            ]
        )
        if res.n_pairs
        else pl.DataFrame(
            schema={
                "planner_a": pl.String,
                "planner_b": pl.String,
                "metric": pl.String,
                "value_a": pl.Float64,
                "value_b": pl.Float64,
                "diff": pl.Float64,
            }
        )
    )
    return AnalysisResult(
        frame=frame,
        summary=summary,
        excluded_rows=excluded,
        seeds_used=seeds,
        diagnostics=(),
        snapshot_id=catalog.snapshot_id,
    )


# -- ranking stability --------------------------------------------------------------------------


def _run_ranking_stability(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, RankingStabilityParams)
    usable, excluded = _valid_episodes(
        catalog, include_references=params.include_references
    )
    frame = usable.filter(
        pl.col(params.metric).is_not_null() & pl.col("planner").is_not_null()
    )
    if frame.is_empty():
        return AnalysisResult(
            frame=pl.DataFrame(schema=RANKING_STABILITY_SCHEMA),
            summary=pl.DataFrame(
                [
                    {
                        "metric": params.metric,
                        "n_swaps": 0,
                        "reason": "no rows",
                    }
                ]
            ),
            excluded_rows=excluded,
            snapshot_id=catalog.snapshot_id,
        )
    directions = _direction(catalog, params.metric)
    direction = _direction_text(directions, params.metric)
    higher = _descending(directions, params.metric)

    # Agreement across crowd conditions: the pair-swap count the H3 contract reports.
    conditions = tuple(
        str(c) for c in frame["crowd_condition"].drop_nulls().unique().sort()
    )
    n_swaps = (
        _swap_count(
            frame,
            params.metric,
            regime_col="crowd_condition",
            regime_a=conditions[0],
            regime_b=conditions[-1],
            direction=direction,
        )
        if len(conditions) >= 2
        else 0
    )
    worlds = tuple(str(w) for w in frame["world"].drop_nulls().unique().sort())
    w_result = (
        kendalls_w(
            frame,
            params.metric,
            entity_col="planner",
            judge_col="world",
            direction=direction,
        )
        if len(worlds) >= 2
        else None
    )

    seed_split = seed_split_rank_stability(
        frame,
        params.metric,
        planner_col="planner",
        seed_col="seed",
        n_splits=params.n_splits,
        k=params.k,
        descending=higher,
        seed=params.seed,
    )
    intervals = paired_cluster_bootstrap_rank_intervals(
        frame,
        params.metric,
        planner_col="planner",
        seed_col="seed",
        n_bootstrap=params.n_boot,
        ci=params.ci_level,
        descending=higher,
        seed=params.seed,
    )
    contextual = (
        contextual_rank_stability(
            frame,
            params.metric,
            arm_col="stage",
            world_col="world",
            crowd_col="crowd_size",
            planner_col="planner",
            seed_col="seed",
            n_splits=params.n_splits,
            descending=higher,
            k=params.k,
            seed=params.seed,
        )
        if {"world", "crowd_size"} <= set(frame.columns)
        else pl.DataFrame(schema=CONTEXTUAL_STABILITY_SCHEMA)
    )

    contextual_rows = contextual.with_columns(
        pl.lit("contextual", dtype=pl.String).alias("table"),
        pl.lit(params.metric, dtype=pl.String).alias("metric"),
        pl.lit(None, dtype=pl.String).alias("planner"),
        pl.lit(None, dtype=pl.Float64).alias("median_rank"),
        pl.lit(None, dtype=pl.Float64).alias("ci_low"),
        pl.lit(None, dtype=pl.Float64).alias("ci_high"),
        pl.lit(None, dtype=pl.Int64).alias("seeds_ranked"),
    )
    interval_rows = intervals.with_columns(
        pl.lit("rank_interval", dtype=pl.String).alias("table"),
        pl.lit(params.metric, dtype=pl.String).alias("metric"),
        pl.lit(None, dtype=pl.String).alias("arm"),
        pl.lit(None, dtype=pl.String).alias("world"),
        pl.lit(None, dtype=pl.String).alias("crowd_size"),
        pl.lit(None, dtype=pl.Float64).alias("mean_tau"),
        pl.lit(None, dtype=pl.Float64).alias("std_tau"),
        pl.lit(None, dtype=pl.Float64).alias("top_k_preservation_rate"),
    )
    declared = list(RANKING_STABILITY_SCHEMA.names())
    out_frame = pl.concat(
        [
            contextual_rows.select([pl.col(name) for name in declared]),
            interval_rows.select([pl.col(name) for name in declared]),
        ],
        how="vertical_relaxed",
    ).select(
        [pl.col(name).cast(dtype) for name, dtype in RANKING_STABILITY_SCHEMA.items()]
    )
    summary = pl.DataFrame(
        [
            {
                "metric": params.metric,
                "direction": direction,
                "n_swaps": n_swaps,
                "kendalls_w": w_result.w if w_result is not None else None,
                "kendalls_w_p": w_result.p_value if w_result is not None else None,
                "seed_split_mean_tau": seed_split.mean_tau,
                "seed_split_std_tau": seed_split.std_tau,
                "seed_split_top_k_rate": seed_split.top_k_preservation_rate,
                "n_seed_splits": seed_split.n_splits,
                "planners_ranked": seed_split.planners_ranked,
                "seeds_ranked": seed_split.seeds_ranked,
                "partially_evaluated": ",".join(seed_split.partially_evaluated),
                "restriction": seed_split.restriction,
            }
        ]
    )
    return AnalysisResult(
        frame=out_frame,
        summary=summary,
        excluded_rows=excluded,
        seeds_used=tuple(int(s) for s in frame["seed"].drop_nulls().unique().sort()),
        diagnostics=directions.diagnostics,
        snapshot_id=catalog.snapshot_id,
    )


# -- visibility effect --------------------------------------------------------------------------


def _run_visibility_effect(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, VisibilityEffectParams)
    usable, excluded = _valid_episodes(
        catalog, include_references=params.include_references
    )
    frame = usable.filter(
        pl.col(params.metric).is_not_null() & pl.col("planner").is_not_null()
    )
    directions = _direction(catalog, params.metric)
    direction = _direction_text(directions, params.metric)
    if frame.is_empty():
        return AnalysisResult(
            frame=pl.DataFrame(schema=VISIBILITY_EFFECT_SCHEMA),
            summary=pl.DataFrame(
                [
                    {
                        "metric": params.metric,
                        "effect_detected": False,
                        "reason": "no rows",
                    }
                ]
            ),
            excluded_rows=excluded,
            snapshot_id=catalog.snapshot_id,
        )

    conditions = tuple(
        str(c) for c in frame["crowd_condition"].drop_nulls().unique().sort()
    )
    condition_swaps = (
        _swap_count(
            frame,
            params.metric,
            regime_col="crowd_condition",
            regime_a="blind",
            regime_b="visible",
            direction=direction,
        )
        if {"blind", "visible"} <= set(conditions)
        else 0
    )
    tagged, first_half, second_half = _seed_half_frame(frame)
    seed_half_swaps = _swap_count(
        tagged,
        params.metric,
        regime_col="__seed_half",
        regime_a="h1",
        regime_b="h2",
        direction=direction,
    )

    # The difference needs an interval: resample whole seeds and recompute both swap counts.
    seeds = np.array(
        [int(s) for s in frame["seed"].drop_nulls().unique().sort()], dtype=np.int64
    )
    rng = np.random.default_rng(params.seed)
    deltas: list[float] = []
    for _ in range(params.n_boot):
        drawn = rng.choice(seeds, size=len(seeds), replace=True)
        counts = {int(seed): int(np.count_nonzero(drawn == seed)) for seed in seeds}
        weighted = frame.with_columns(
            pl.col("seed")
            .map_elements(
                lambda seed, counts=counts: float(counts.get(int(seed), 0)),
                return_dtype=pl.Float64,
            )
            .alias("__weight")
        ).filter(pl.col("__weight") > 0)
        boot_condition = _weighted_swap_count(
            weighted, params.metric, "crowd_condition", "blind", "visible", direction
        )
        boot_tagged, _, _ = _seed_half_frame(weighted)
        boot_seed_half = _weighted_swap_count(
            boot_tagged, params.metric, "__seed_half", "h1", "h2", direction
        )
        deltas.append(float(boot_condition - boot_seed_half))
    delta_values = np.asarray(deltas, dtype=np.float64)
    observed = float(condition_swaps - seed_half_swaps)
    if delta_values.size:
        low, high = contain_estimate(
            float(np.quantile(delta_values, (1.0 - params.ci_level) / 2.0)),
            float(np.quantile(delta_values, 1.0 - (1.0 - params.ci_level) / 2.0)),
            observed,
        )
        p_value = float(
            (1 + int(np.count_nonzero(delta_values <= 0.0))) / (params.n_boot + 1)
        )
    else:
        low = high = p_value = None
    effect = bool(low is not None and low > 0.0)

    out_frame = pl.DataFrame(
        [
            {
                "metric": params.metric,
                "condition_swaps": condition_swaps,
                "seed_half_swaps": seed_half_swaps,
                "swap_delta": int(observed),
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "effect_detected": effect,
                "n_boot": params.n_boot,
                "seed_half_split": (f"{first_half} vs {second_half}"),
            }
        ],
        schema=VISIBILITY_EFFECT_SCHEMA,
    )
    return AnalysisResult(
        frame=out_frame,
        summary=out_frame,
        excluded_rows=excluded,
        seeds_used=tuple(int(s) for s in seeds),
        diagnostics=directions.diagnostics,
        snapshot_id=catalog.snapshot_id,
    )


def _weighted_swap_count(
    frame: pl.DataFrame,
    metric: str,
    regime_col: str,
    regime_a: object,
    regime_b: object,
    direction: str,
    *,
    entity_col: str = "planner",
) -> int:
    """Swap count where each row stands for the number of times its seed was drawn."""
    ranked = (
        frame.filter(
            pl.col(metric).is_not_null()
            & pl.col(regime_col).is_not_null()
            & pl.col(entity_col).is_not_null()
        )
        .group_by(regime_col, entity_col)
        .agg(
            (
                (pl.col(metric).cast(pl.Float64) * pl.col("__weight")).sum()
                / pl.col("__weight").sum()
            ).alias("__mean")
        )
        .with_columns(
            rank=pl.col("__mean")
            .rank(method="average", descending=direction == "desc")
            .over(regime_col)
        )
    )
    side_a = ranked.filter(pl.col(regime_col).cast(pl.String) == str(regime_a)).select(
        pl.col(entity_col).cast(pl.String).alias("entity"), pl.col("rank")
    )
    side_b = ranked.filter(pl.col(regime_col).cast(pl.String) == str(regime_b)).select(
        pl.col(entity_col).cast(pl.String).alias("entity"), pl.col("rank")
    )
    joined = side_a.join(side_b, on="entity", how="inner", suffix="_b")
    if joined.height < 2:
        return 0
    ranks_a = joined["rank"].to_numpy().astype(np.float64)
    ranks_b = joined["rank_b"].to_numpy().astype(np.float64)
    idx_i, idx_j = np.triu_indices(joined.height, k=1)
    product = (ranks_a[idx_i] - ranks_a[idx_j]) * (ranks_b[idx_i] - ranks_b[idx_j])
    return int(np.count_nonzero(product < -1e-12))


# -- lead measures ------------------------------------------------------------------------------


def _run_lead_measures(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, LeadMeasuresParams)
    usable, excluded = _valid_episodes(
        catalog, include_references=params.include_references
    )
    if usable.is_empty() or usable["seed"].drop_nulls().n_unique() < 2:
        return AnalysisResult(
            frame=pl.DataFrame(schema=LEAD_MEASURES_SCHEMA),
            summary=pl.DataFrame(
                [
                    {
                        "discovery_seeds": [],
                        "verification_seeds": [],
                        "discovery_seeds_csv": "",
                        "verification_seeds_csv": "",
                        "n_selected_measures": 0,
                        "reason": "fewer than two seeds",
                    }
                ]
            ),
            excluded_rows=excluded,
            snapshot_id=catalog.snapshot_id,
        )
    cohort = split_discovery_verification_seeds(usable, seed_col="seed")
    discovery_seeds = tuple(
        int(s)
        for s in cohort.filter(pl.col("cohort") == "discovery")["seed"]
        .drop_nulls()
        .unique()
        .sort()
    )
    verification_seeds = tuple(
        int(s)
        for s in cohort.filter(pl.col("cohort") == "verification")["seed"]
        .drop_nulls()
        .unique()
        .sort()
    )
    discovery = cohort.filter(pl.col("cohort") == "discovery")
    verification = cohort.filter(pl.col("cohort") == "verification")

    rows: list[dict[str, object]] = []
    for metric in params.metrics:
        if metric not in usable.columns:
            continue
        means = _planner_means(discovery, metric)
        if len(means) < 2:
            continue
        separation = float(max(means.values()) - min(means.values()))
        discovery_intervals = _planner_intervals(
            discovery,
            metric,
            n_boot=params.n_boot,
            ci_level=params.ci_level,
            seed=params.seed,
        )
        noise = (
            max((high - low) / 2.0 for low, high in discovery_intervals.values())
            if discovery_intervals
            else 0.0
        )
        verification_intervals = _planner_intervals(
            verification,
            metric,
            n_boot=params.n_boot,
            ci_level=params.ci_level,
            seed=params.seed,
        )
        directions = _direction(catalog, metric)
        separated = _separated_pairs(
            verification_intervals, higher_is_better=_descending(directions, metric)
        )
        pairs_total = (
            len(verification_intervals) * (len(verification_intervals) - 1) // 2
        )
        rows.append(
            {
                "metric": metric,
                "discovery_separation": separation,
                "discovery_seed_noise": noise,
                "selected": bool(separation > noise),
                "verification_separated_pairs": separated,
                "verification_pairs_total": pairs_total,
                "planner_intervals": _format_intervals(verification_intervals),
            }
        )

    out_frame = (
        pl.DataFrame(rows, schema=LEAD_MEASURES_SCHEMA)
        if rows
        else pl.DataFrame(schema=LEAD_MEASURES_SCHEMA)
    )
    selected = [row["metric"] for row in rows if row["selected"]]
    summary = pl.DataFrame(
        [
            {
                "discovery_seeds": list(discovery_seeds),
                "verification_seeds": list(verification_seeds),
                "discovery_seeds_csv": ",".join(str(s) for s in discovery_seeds),
                "verification_seeds_csv": ",".join(str(s) for s in verification_seeds),
                "n_selected_measures": len(selected),
                "selected_measures": ",".join(str(name) for name in selected),
            }
        ]
    )
    return AnalysisResult(
        frame=out_frame,
        summary=summary,
        excluded_rows=excluded,
        seeds_used=tuple(sorted({*discovery_seeds, *verification_seeds})),
        snapshot_id=catalog.snapshot_id,
    )


# -- attribution shares -------------------------------------------------------------------------


def _run_attribution_shares(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, AttributionSharesParams)
    contacts = (
        catalog.get_table("v_contacts")
        .join(
            catalog.get_table("v_episode_summary").select(
                [
                    "run_id",
                    "episode_id",
                    "robot_id",
                    "planner",
                    "crowd_condition",
                    "is_valid_for_ranking",
                    "is_reference",
                    "seed",
                ]
            ),
            on=["run_id", "episode_id", "robot_id"],
            how="left",
        )
        .collect()
    )
    if contacts.is_empty():
        return AnalysisResult(
            frame=pl.DataFrame(schema=ATTRIBUTION_SHARES_SCHEMA),
            summary=pl.DataFrame([{"n_contacts": 0}]),
            snapshot_id=catalog.snapshot_id,
        )

    excluded: dict[str, int] = {}
    pedestrian = contacts.filter(pl.col("collider_kind") == "pedestrian")
    excluded["non_pedestrian_contacts"] = contacts.height - pedestrian.height
    valid = pedestrian.filter(
        pl.col("is_valid_for_ranking").fill_null(False).cast(pl.Boolean)
    )
    if pedestrian.height > valid.height:
        excluded["invalid_for_ranking"] = pedestrian.height - valid.height
    if not params.include_references:
        without_references = valid.filter(
            ~pl.col("is_reference").fill_null(False).cast(pl.Boolean)
        )
        if valid.height > without_references.height:
            excluded[_REFERENCE_FILTER] = valid.height - without_references.height
        valid = without_references
    recorded = valid.filter(
        pl.col("v_close_robot_mps").is_not_null()
        & pl.col("v_close_target_mps").is_not_null()
    )
    if valid.height > recorded.height:
        excluded["closing_speed:not_recorded"] = valid.height - recorded.height
    if recorded.is_empty():
        return AnalysisResult(
            frame=pl.DataFrame(schema=ATTRIBUTION_SHARES_SCHEMA),
            summary=pl.DataFrame(
                [{"n_contacts": 0, "reason": "no recorded closing speeds"}]
            ),
            excluded_rows=excluded,
            snapshot_id=catalog.snapshot_id,
        )

    # Probst responsibility of the encounters these contacts belong to, pooled by contact.
    encounters = (
        catalog.get_table("fact_encounters")
        .select(
            [
                "run_id",
                "episode_id",
                "robot_id",
                "encounter_id",
                "probst_R_robot",
                "probst_R_actor2",
                "probst_E_robot",
                "probst_E_actor2",
            ]
        )
        .collect()
    )
    enriched = recorded.join(
        encounters,
        on=["run_id", "episode_id", "robot_id", "encounter_id"],
        how="left",
    ).with_columns(
        pl.col("v_close_robot_mps").clip(lower_bound=0.0).alias("__robot_part"),
        pl.col("v_close_target_mps").clip(lower_bound=0.0).alias("__target_part"),
        (pl.col("v_close_robot_mps") <= 0.0).alias("__frozen"),
    )
    grouped = (
        enriched.group_by(["planner", "crowd_condition"])
        .agg(
            pl.len().cast(pl.Int64).alias("n_contacts"),
            pl.col("__robot_part").sum().alias("robot_closing_sum"),
            pl.col("__target_part").sum().alias("target_closing_sum"),
            pl.col("robot_closing_share").mean().alias("mean_contact_robot_share"),
            pl.col("__frozen").mean().alias("frozen_robot_share"),
            pl.col("freeze_clearance_m").mean().alias("freeze_clearance_mean_m"),
            pl.col("probst_R_robot").mean().alias("probst_R_robot_pooled"),
            pl.col("probst_R_actor2").mean().alias("probst_R_actor2_pooled"),
            pl.col("probst_E_robot").mean().alias("probst_E_robot_pooled"),
            pl.col("probst_E_actor2").mean().alias("probst_E_actor2_pooled"),
        )
        .with_columns(
            (
                pl.col("robot_closing_sum")
                / (pl.col("robot_closing_sum") + pl.col("target_closing_sum"))
            ).alias("robot_closing_share"),
            (
                pl.col("target_closing_sum")
                / (pl.col("robot_closing_sum") + pl.col("target_closing_sum"))
            ).alias("target_closing_share"),
        )
        .select(list(ATTRIBUTION_SHARES_SCHEMA.names()))
        .sort(["planner", "crowd_condition"])
    )

    ci_frame = cluster_bootstrap_ci(
        enriched,
        "robot_closing_share",
        group_cols=["planner"],
        cluster_col="seed",
        n_boot=params.n_boot,
        ci_level=params.ci_level,
        random_seed=params.seed,
    )
    return AnalysisResult(
        frame=grouped,
        summary=ci_frame,
        excluded_rows=excluded,
        seeds_used=tuple(int(s) for s in enriched["seed"].drop_nulls().unique().sort()),
        snapshot_id=catalog.snapshot_id,
    )


# -- registry -----------------------------------------------------------------------------------

_PROCEDURES: dict[str, ProcedureSpec] = {
    "triage": ProcedureSpec(
        name="triage",
        description=(
            "Episode usability per planner/world/condition with the invalid-reason breakdown and each "
            "metric's missing counts."
        ),
        params_model=TriageParams,
        fn=_run_triage,
    ),
    "compare": ProcedureSpec(
        name="compare",
        description=(
            "Paired planner comparison with cluster-bootstrap CI, exclusion counts, and a typed reason "
            "when the cell has no complete pair."
        ),
        params_model=CompareParams,
        fn=_run_compare,
    ),
    "ranking_stability": ProcedureSpec(
        name="ranking_stability",
        description=(
            "Rank repeatability: pair swaps across crowd conditions, Kendall's W across worlds, "
            "seed-half agreement, and per-planner seed-bootstrap rank intervals."
        ),
        params_model=RankingStabilityParams,
        fn=_run_ranking_stability,
    ),
    "visibility_effect": ProcedureSpec(
        name="visibility_effect",
        description=(
            "Whether the crowd-condition rank swap count exceeds the seed-half swap count, with a "
            "seed-clustered bootstrap interval on the difference."
        ),
        params_model=VisibilityEffectParams,
        fn=_run_visibility_effect,
    ),
    "lead_measures": ProcedureSpec(
        name="lead_measures",
        description=(
            "Select measures by planner separation on the discovery seeds, then count the separated "
            "planner pairs on the verification seeds using each planner's seed interval."
        ),
        params_model=LeadMeasuresParams,
        fn=_run_lead_measures,
    ),
    "attribution_shares": ProcedureSpec(
        name="attribution_shares",
        description=(
            "Pedestrian-contact closing shares of valid episodes, pooled from the contact parts, with the "
            "frozen-robot persistence and the pooled Probst responsibility."
        ),
        params_model=AttributionSharesParams,
        fn=_run_attribution_shares,
    ),
}


def list_procedures() -> list[dict[str, Any]]:
    """Return metadata and parameter schemas for all registered analysis procedures."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.params_model.model_json_schema(),
        }
        for spec in _PROCEDURES.values()
    ]


def get_procedure(name: str) -> ProcedureSpec:
    """Look up a named analysis procedure or raise KeyError."""
    if name not in _PROCEDURES:
        raise KeyError(
            f"Unknown analysis procedure {name!r}; available: {sorted(_PROCEDURES)}"
        )
    return _PROCEDURES[name]


def run_procedure(
    name: str,
    catalog: BenchmarkCatalog,
    params: dict[str, Any] | BaseModel | None = None,
    *,
    evidence_store: EvidenceStore | None = None,
    save_as: str | None = None,
) -> AnalysisResult:
    """Execute a registered analysis procedure and optionally pin its result in EvidenceStore."""
    spec = get_procedure(name)
    parsed = (
        params
        if isinstance(params, spec.params_model)
        else spec.params_model.model_validate(params or {})
    )
    result = spec.fn(catalog, parsed)
    if evidence_store is not None:
        record = evidence_store.record(
            result.frame,
            kind="computation",
            operation=name,
            snapshot_id=result.snapshot_id,
            purpose=save_as or name,
            params=parsed.model_dump(mode="json"),
        )
        return AnalysisResult(
            frame=result.frame,
            summary=result.summary,
            excluded_rows=result.excluded_rows,
            seeds_used=result.seeds_used,
            snapshot_id=result.snapshot_id,
            evidence_id=record.evidence_id,
            diagnostics=result.diagnostics,
        )
    return result


def available_procedures() -> tuple[str, ...]:
    """Every registered procedure name, sorted."""
    return tuple(sorted(_PROCEDURES))
