"""Pure statistical analysis procedures built on BenchmarkCatalog and analytics.stats."""

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

_PAIR_FRAME_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "planner_a": pl.String,
        "planner_b": pl.String,
        "metric": pl.String,
        "value_a": pl.Float64,
        "value_b": pl.Float64,
        "diff": pl.Float64,
    }
)

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
        description=("Per-metric missingness to break down; empty takes every metric the metric dictionary publishes at episode or robot scope."),
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

    metrics: list[str] = Field(default_factory=lambda: ["time_to_goal", "energy_total_wh", "path_efficiency"])
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


def _valid_episodes(catalog: BenchmarkCatalog, *, include_references: bool) -> tuple[pl.DataFrame, dict[str, int]]:
    """The usable episode rows: non-reference (unless asked) and flagged valid for ranking."""
    frame = catalog.get_table("v_episode_summary").collect()
    excluded: dict[str, int] = {}
    if not include_references:
        before = frame.height
        frame = frame.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))
        if before > frame.height:
            excluded[_REFERENCE_FILTER] = before - frame.height
    valid = frame.filter(pl.col("is_valid_for_ranking").fill_null(False).cast(pl.Boolean))
    if frame.height > valid.height:
        excluded["invalid_for_ranking"] = frame.height - valid.height
    return valid, excluded


def _orientation(catalog: BenchmarkCatalog, metric: str) -> tuple[str, bool, MetricDirections]:
    """Resolve a metric's ``stats`` direction text, higher-is-better flag, and source."""
    directions = resolve_metric_directions(catalog, (metric,))
    higher = directions.by_metric[metric] == "higher"
    return ("desc" if higher else "asc"), higher, directions


def _seeds(frame: pl.DataFrame) -> tuple[int, ...]:
    """Sorted distinct non-NULL seed values of a frame."""
    return tuple(int(seed) for seed in frame["seed"].drop_nulls().unique().sort())


def _seed_half_frame(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, tuple[int, ...], tuple[int, ...]]:
    """Split the sorted seeds into two deterministic halves and tag every row with its half."""
    seeds = _seeds(frame)
    middle = len(seeds) // 2
    first, second = seeds[:middle], seeds[middle:]
    tagged = frame.with_columns(pl.when(pl.col("seed").is_in(list(first))).then(pl.lit("h1")).otherwise(pl.lit("h2")).alias("__seed_half"))
    return tagged, first, second


def _swap_count(
    frame: pl.DataFrame,
    metric: str,
    regime_col: str,
    regimes: tuple[object, object],
    direction: str,
    *,
    entity_col: str = "planner",
    weight_col: str | None = None,
) -> int:
    """Rank entities in two regimes and count the pairs whose order the regimes swap."""
    mean = (pl.col(metric).cast(pl.Float64) * pl.col(weight_col)).sum() / pl.col(weight_col).sum() if weight_col is not None else pl.col(metric).cast(pl.Float64).mean()
    ranked = frame.filter(pl.col(metric).is_not_null() & pl.col(regime_col).is_not_null() & pl.col(entity_col).is_not_null()).group_by(regime_col, entity_col).agg(mean.alias("__mean")).with_columns(rank=pl.col("__mean").rank(method="average", descending=direction == "desc").over(regime_col))
    sides = [ranked.filter(pl.col(regime_col).cast(pl.String) == str(regime)).select(pl.col(entity_col).cast(pl.String).alias("entity"), pl.col("rank")) for regime in regimes]
    joined = sides[0].join(sides[1], on="entity", how="inner", suffix="_b")
    if joined.height < 2:
        return 0
    rank_a = joined["rank"].to_numpy().astype(np.float64)
    rank_b = joined["rank_b"].to_numpy().astype(np.float64)
    left, right = np.triu_indices(joined.height, k=1)
    product = (rank_a[left] - rank_a[right]) * (rank_b[left] - rank_b[right])
    return int(np.count_nonzero(product < -1e-12))


def _planner_intervals(
    frame: pl.DataFrame,
    metric: str,
    n_boot: int,
    ci_level: float,
    seed: int,
    *,
    cluster_col: str = "seed",
) -> dict[str, tuple[float, float]]:
    """Per-planner seed-clustered interval of the metric mean: the seed spread D-12 asks for."""
    usable = frame.filter(pl.col(metric).is_not_null() & pl.col(cluster_col).is_not_null())
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
        if row["ci_low"] is not None and row["ci_high"] is not None:
            intervals[str(row["planner"])] = (
                float(row["ci_low"]),
                float(row["ci_high"]),
            )
    return intervals


def _separated_pairs(intervals: Mapping[str, tuple[float, float]], *, higher_is_better: bool) -> int:
    """Count planner pairs whose seed intervals do not overlap in the better direction."""
    lows = np.array([low for low, _ in intervals.values()], dtype=np.float64)
    highs = np.array([high for _, high in intervals.values()], dtype=np.float64)
    separated = lows[:, None] > highs[None, :] if higher_is_better else highs[:, None] < lows[None, :]
    return int(np.count_nonzero(separated))


def _selection(frame: pl.DataFrame, metric: str, n_boot: int, ci_level: float, seed: int) -> tuple[float, float, dict[str, tuple[float, float]]] | None:
    """Discovery separation and the seed-interval noise it must exceed, or ``None`` if unrankable."""
    means = frame.filter(pl.col(metric).is_not_null()).group_by("planner").agg(pl.col(metric).cast(pl.Float64).mean().alias("value")).sort("planner")
    by_planner = {
        str(name): float(value)
        for name, value in means.iter_rows()  # agent-contracts: ignore metadata dict boundary
        if value is not None
    }
    if len(by_planner) < 2:
        return None
    intervals = _planner_intervals(frame, metric, n_boot, ci_level, seed)
    noise = max((high - low) / 2.0 for low, high in intervals.values()) if intervals else 0.0
    return max(by_planner.values()) - min(by_planner.values()), noise, intervals


def _project_rows(frame: pl.DataFrame, schema: pl.Schema, **values: object) -> pl.DataFrame:
    """Add typed NULLs for absent columns, apply literals, and cast to the declared schema."""
    present = frame.with_columns([pl.lit(value, dtype=schema[name]).alias(name) for name, value in values.items()])
    nulls = [pl.lit(None, dtype=dtype).alias(name) for name, dtype in schema.items() if name not in present.columns]
    return present.with_columns(nulls).select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


def _summary(**values: object) -> pl.DataFrame:
    """One summary row from the named values."""
    return pl.DataFrame([values])


def _result(
    catalog: BenchmarkCatalog,
    *,
    frame: pl.DataFrame,
    summary: pl.DataFrame,
    excluded: Mapping[str, int] | None = None,
    seeds_used: tuple[int, ...] = (),
    diagnostics: tuple[str, ...] = (),
) -> AnalysisResult:
    """Bind one procedure's frame, summary, and exclusions to its catalogue snapshot."""
    return AnalysisResult(
        frame=frame,
        summary=summary,
        excluded_rows=excluded or {},
        seeds_used=seeds_used,
        diagnostics=diagnostics,
        snapshot_id=catalog.snapshot_id,
    )


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
    tokens = (
        nulls.select(
            [
                *group_cols,
                pl.concat_str(
                    [pl.col(name).fill_null("") for name in reason_columns],
                    separator=";",
                ).alias("__tokens"),
            ]
        )
        .with_columns(pl.col("__tokens").str.split(";").alias("reason"))
        .explode("reason", empty_as_null=True)
        .filter(pl.col("reason").str.starts_with(f"{metric}:"))
        .with_columns(pl.col("reason").str.slice(len(metric) + 1).alias("reason"))
        .group_by([*group_cols, "reason"])
        .agg(pl.len().cast(pl.Int64).alias("episodes"))
    )
    attributed = tokens.group_by(group_cols).agg(pl.col("episodes").sum().alias("__attributed"))
    remainder = (
        nulls.group_by(group_cols)
        .agg(pl.len().cast(pl.Int64).alias("null_episodes"))
        .join(attributed, on=group_cols, how="left")
        .with_columns((pl.col("null_episodes") - pl.col("__attributed").fill_null(0)).alias("episodes"))
        .filter(pl.col("episodes") > 0)
        .with_columns(pl.lit(_NULL_KEY_REASON, dtype=pl.String).alias("reason"))
    )
    return _project_rows(
        pl.concat(
            [
                tokens.select([*group_cols, "reason", "episodes"]),
                remainder.select([*group_cols, "reason", "episodes"]),
            ],
            how="vertical_relaxed",
        ),
        TRIAGE_SCHEMA,
        row_kind="metric_null",
        metric=metric,
    ).sort(group_cols + ["reason"])


def _run_triage(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, TriageParams)
    usable, excluded = _valid_episodes(catalog, include_references=params.include_references)
    metrics = params.metrics or [str(name) for name in catalog.metric_dictionary().filter(pl.col("scope").is_in(["episodes", "robots"])).select("metric").unique().sort("metric").collect().get_column("metric")]
    group_cols = ["planner", "world", "crowd_condition"]
    counts = (
        usable.group_by(group_cols)
        .agg(
            pl.len().cast(pl.Int64).alias("total_episodes"),
            pl.col("is_valid_for_ranking").fill_null(False).sum().cast(pl.Int64).alias("valid_episodes"),
            (~pl.col("is_valid_for_ranking").fill_null(False)).sum().cast(pl.Int64).alias("excluded_episodes"),
        )
        .sort(group_cols)
        .with_columns(episodes=pl.col("valid_episodes"))
    )
    parts = [_project_rows(counts, TRIAGE_SCHEMA, row_kind="group")]
    invalid_reasons = (
        usable.filter(pl.col("invalid_reason").is_not_null())
        .select([*group_cols, "invalid_reason"])
        .with_columns(pl.col("invalid_reason").str.split(";").alias("reason"))
        .explode("reason", empty_as_null=True)
        .filter(pl.col("reason").is_not_null())
        .group_by([*group_cols, "reason"])
        .agg(pl.len().cast(pl.Int64).alias("episodes"))
    )
    parts.append(_project_rows(invalid_reasons, TRIAGE_SCHEMA, row_kind="invalid_reason").sort(group_cols + ["reason"]))
    reason_columns = [name for name in usable.columns if name.endswith("_null_reason")]
    parts.extend(part for metric in metrics if not (part := _metric_null_rows(usable, metric, group_cols, reason_columns)).is_empty())
    frame = pl.concat([part for part in parts if not part.is_empty()], how="vertical_relaxed")
    summary = _summary(
        total_episodes=usable.height,
        valid_episodes=int(usable.filter(pl.col("is_valid_for_ranking").fill_null(False)).height),
        excluded_episodes=excluded.get("invalid_for_ranking", 0),
        metrics=len(metrics),
    )
    return _result(catalog, frame=frame, summary=summary, excluded=excluded)


def _run_compare(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, CompareParams)
    usable, excluded = _valid_episodes(catalog, include_references=params.include_references)
    if params.metric not in usable.columns:
        return _result(
            catalog,
            frame=pl.DataFrame(),
            summary=_summary(
                reason=f"unknown metric {params.metric!r}",
                hint="pick a metric the metric dictionary publishes",
            ),
            excluded={f"{params.metric}:unknown": usable.height},
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
    pairing = [key for key in _PAIRING_KEYS if key in valid.columns]
    if not pairing:
        return _result(
            catalog,
            frame=pl.DataFrame(),
            summary=_summary(
                planner_a=params.planner_a,
                planner_b=params.planner_b,
                metric=params.metric,
                n_pairs=0,
                reason="no_pairing_key",
                hint=(f"the episode view carries none of {list(_PAIRING_KEYS)}; pair on at least one blocking column"),
            ),
            excluded={f"{params.metric}:no_pairing_key": valid.height},
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
        return _result(
            catalog,
            frame=pl.DataFrame(),
            summary=_summary(
                planner_a=params.planner_a,
                planner_b=params.planner_b,
                metric=params.metric,
                n_pairs=0,
                reason="all_null",
                hint=f"{params.metric} is NULL for both planners; see excluded_rows",
            ),
            excluded=excluded or {f"{params.metric}:all_null": sub.height},
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
        hint = "no complete pair: " + (f"the pairing keys {null_keys} are unrecorded" if null_keys else f"the two planners share no cell on {pairing}")
    elif null_keys:
        hint = f"{len(null_keys)} pairing key(s) had unrecorded rows: {null_keys}"
    summary = _summary(
        planner_a=params.planner_a,
        planner_b=params.planner_b,
        metric=params.metric,
        mean_diff=res.mean_diff,
        ci_low=res.ci_low,
        ci_high=res.ci_high,
        p_value=res.p_value,
        n_pairs=res.n_pairs,
        n_clusters=res.n_clusters,
        cohens_dz=res.cohens_dz,
        pairing_keys=",".join(pairing),
        reason="no_complete_pair" if res.n_pairs == 0 else None,
        hint=hint or None,
    )
    frame = _project_rows(
        res.pairs_frame,
        _PAIR_FRAME_SCHEMA,
        planner_a=params.planner_a,
        planner_b=params.planner_b,
        metric=params.metric,
    )
    return _result(
        catalog,
        frame=frame,
        summary=summary,
        excluded=excluded,
        seeds_used=_seeds(valid),
    )


def _run_ranking_stability(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, RankingStabilityParams)
    usable, excluded = _valid_episodes(catalog, include_references=params.include_references)
    frame = usable.filter(pl.col(params.metric).is_not_null() & pl.col("planner").is_not_null())
    if frame.is_empty():
        return _result(
            catalog,
            frame=pl.DataFrame(schema=RANKING_STABILITY_SCHEMA),
            summary=_summary(metric=params.metric, n_swaps=0, reason="no rows"),
            excluded=excluded,
        )
    direction, higher, directions = _orientation(catalog, params.metric)

    # Agreement across crowd conditions: the pair-swap count the H3 contract reports.
    conditions = tuple(str(c) for c in frame["crowd_condition"].drop_nulls().unique().sort())
    n_swaps = (
        _swap_count(
            frame,
            params.metric,
            "crowd_condition",
            (conditions[0], conditions[-1]),
            direction,
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

    out_frame = pl.concat(
        [
            _project_rows(
                contextual,
                RANKING_STABILITY_SCHEMA,
                table="contextual",
                metric=params.metric,
            ),
            _project_rows(
                intervals,
                RANKING_STABILITY_SCHEMA,
                table="rank_interval",
                metric=params.metric,
            ),
        ],
        how="vertical_relaxed",
    )
    summary = _summary(
        metric=params.metric,
        direction=direction,
        n_swaps=n_swaps,
        kendalls_w=w_result.w if w_result is not None else None,
        kendalls_w_p=w_result.p_value if w_result is not None else None,
        seed_split_mean_tau=seed_split.mean_tau,
        seed_split_std_tau=seed_split.std_tau,
        seed_split_top_k_rate=seed_split.top_k_preservation_rate,
        n_seed_splits=seed_split.n_splits,
        planners_ranked=seed_split.planners_ranked,
        seeds_ranked=seed_split.seeds_ranked,
        partially_evaluated=",".join(seed_split.partially_evaluated),
        restriction=seed_split.restriction,
    )
    return _result(
        catalog,
        frame=out_frame,
        summary=summary,
        excluded=excluded,
        seeds_used=_seeds(frame),
        diagnostics=directions.diagnostics,
    )


def _run_visibility_effect(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, VisibilityEffectParams)
    usable, excluded = _valid_episodes(catalog, include_references=params.include_references)
    frame = usable.filter(pl.col(params.metric).is_not_null() & pl.col("planner").is_not_null())
    direction, _, directions = _orientation(catalog, params.metric)
    if frame.is_empty():
        return _result(
            catalog,
            frame=pl.DataFrame(schema=VISIBILITY_EFFECT_SCHEMA),
            summary=_summary(
                metric=params.metric,
                effect_detected=False,
                reason="no rows",
            ),
            excluded=excluded,
        )

    conditions = tuple(str(c) for c in frame["crowd_condition"].drop_nulls().unique().sort())
    condition_swaps = (
        _swap_count(
            frame,
            params.metric,
            "crowd_condition",
            ("blind", "visible"),
            direction,
        )
        if {"blind", "visible"} <= set(conditions)
        else 0
    )
    tagged, first_half, second_half = _seed_half_frame(frame)
    seed_half_swaps = _swap_count(tagged, params.metric, "__seed_half", ("h1", "h2"), direction)

    # The difference needs an interval: resample whole seeds and recompute both swap counts.
    seeds = np.asarray(_seeds(frame), dtype=np.int64)
    rng = np.random.default_rng(params.seed)
    deltas: list[float] = []
    for _ in range(params.n_boot):
        drawn = rng.choice(seeds, size=len(seeds), replace=True)
        drawn_seeds, draw_counts = np.unique(drawn, return_counts=True)
        # One weight per drawn seed; a seed the replicate did not draw weighs 0 and drops out.
        weights = {int(seed): float(count) for seed, count in zip(drawn_seeds, draw_counts, strict=True)}
        weighted = frame.with_columns(pl.col("seed").replace_strict(weights, default=0.0, return_dtype=pl.Float64).alias("__weight")).filter(pl.col("__weight") > 0)
        boot_condition = _swap_count(
            weighted,
            params.metric,
            "crowd_condition",
            ("blind", "visible"),
            direction,
            weight_col="__weight",
        )
        boot_tagged, _, _ = _seed_half_frame(weighted)
        boot_seed_half = _swap_count(
            boot_tagged,
            params.metric,
            "__seed_half",
            ("h1", "h2"),
            direction,
            weight_col="__weight",
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
        p_value = float((1 + int(np.count_nonzero(delta_values <= 0.0))) / (params.n_boot + 1))
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
    return _result(
        catalog,
        frame=out_frame,
        summary=out_frame,
        excluded=excluded,
        seeds_used=tuple(int(seed) for seed in seeds),
        diagnostics=directions.diagnostics,
    )


def _run_lead_measures(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, LeadMeasuresParams)
    usable, excluded = _valid_episodes(catalog, include_references=params.include_references)
    if usable.is_empty() or usable["seed"].drop_nulls().n_unique() < 2:
        return _result(
            catalog,
            frame=pl.DataFrame(schema=LEAD_MEASURES_SCHEMA),
            summary=_summary(
                discovery_seeds=[],
                verification_seeds=[],
                discovery_seeds_csv="",
                verification_seeds_csv="",
                n_selected_measures=0,
                reason="fewer than two seeds",
            ),
            excluded=excluded,
        )
    cohort = split_discovery_verification_seeds(usable, seed_col="seed")
    discovery = cohort.filter(pl.col("cohort") == "discovery")
    verification = cohort.filter(pl.col("cohort") == "verification")
    discovery_seeds = _seeds(discovery)
    verification_seeds = _seeds(verification)

    rows: list[dict[str, object]] = []
    for metric in params.metrics:
        if metric not in usable.columns:
            continue
        selection = _selection(discovery, metric, params.n_boot, params.ci_level, params.seed)
        if selection is None:
            continue
        separation, noise, _ = selection
        verification_intervals = _planner_intervals(verification, metric, params.n_boot, params.ci_level, params.seed)
        _, higher, _ = _orientation(catalog, metric)
        separated = _separated_pairs(verification_intervals, higher_is_better=higher)
        pairs_total = len(verification_intervals) * (len(verification_intervals) - 1) // 2
        rows.append(
            {
                "metric": metric,
                "discovery_separation": separation,
                "discovery_seed_noise": noise,
                "selected": bool(separation > noise),
                "verification_separated_pairs": separated,
                "verification_pairs_total": pairs_total,
                "planner_intervals": ";".join(f"{name}[{low:.6g},{high:.6g}]" for name, (low, high) in sorted(verification_intervals.items())),
            }
        )

    out_frame = pl.DataFrame(rows, schema=LEAD_MEASURES_SCHEMA)
    selected = [row["metric"] for row in rows if row["selected"]]
    summary = _summary(
        discovery_seeds=list(discovery_seeds),
        verification_seeds=list(verification_seeds),
        discovery_seeds_csv=",".join(str(seed) for seed in discovery_seeds),
        verification_seeds_csv=",".join(str(seed) for seed in verification_seeds),
        n_selected_measures=len(selected),
        selected_measures=",".join(str(name) for name in selected),
    )
    return _result(
        catalog,
        frame=out_frame,
        summary=summary,
        excluded=excluded,
        seeds_used=tuple(sorted({*discovery_seeds, *verification_seeds})),
    )


def _run_attribution_shares(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
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
        return _result(
            catalog,
            frame=pl.DataFrame(schema=ATTRIBUTION_SHARES_SCHEMA),
            summary=_summary(n_contacts=0),
        )

    excluded: dict[str, int] = {}
    pedestrian = contacts.filter(pl.col("collider_kind") == "pedestrian")
    excluded["non_pedestrian_contacts"] = contacts.height - pedestrian.height
    valid = pedestrian.filter(pl.col("is_valid_for_ranking").fill_null(False).cast(pl.Boolean))
    if pedestrian.height > valid.height:
        excluded["invalid_for_ranking"] = pedestrian.height - valid.height
    if not params.include_references:
        without_references = valid.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))
        if valid.height > without_references.height:
            excluded[_REFERENCE_FILTER] = valid.height - without_references.height
        valid = without_references
    recorded = valid.filter(pl.col("v_close_robot_mps").is_not_null() & pl.col("v_close_target_mps").is_not_null())
    if valid.height > recorded.height:
        excluded["closing_speed:not_recorded"] = valid.height - recorded.height
    if recorded.is_empty():
        return _result(
            catalog,
            frame=pl.DataFrame(schema=ATTRIBUTION_SHARES_SCHEMA),
            summary=_summary(n_contacts=0, reason="no recorded closing speeds"),
            excluded=excluded,
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
            (pl.col("robot_closing_sum") / (pl.col("robot_closing_sum") + pl.col("target_closing_sum"))).alias("robot_closing_share"),
            (pl.col("target_closing_sum") / (pl.col("robot_closing_sum") + pl.col("target_closing_sum"))).alias("target_closing_share"),
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
    return _result(
        catalog,
        frame=grouped,
        summary=ci_frame,
        excluded=excluded,
        seeds_used=_seeds(enriched),
    )


_PROCEDURES: dict[str, ProcedureSpec] = {
    spec_name: ProcedureSpec(spec_name, description, model, fn)
    for spec_name, description, model, fn in (
        (
            "triage",
            "Episode usability per planner/world/condition with the invalid-reason breakdown and each metric's missing counts.",
            TriageParams,
            _run_triage,
        ),
        (
            "compare",
            "Paired planner comparison with cluster-bootstrap CI, exclusion counts, and a typed reason when the cell has no complete pair.",
            CompareParams,
            _run_compare,
        ),
        (
            "ranking_stability",
            "Rank repeatability: pair swaps across crowd conditions, Kendall's W across worlds, seed-half agreement, and per-planner seed-bootstrap rank intervals.",
            RankingStabilityParams,
            _run_ranking_stability,
        ),
        (
            "visibility_effect",
            "Whether the crowd-condition rank swap count exceeds the seed-half swap count, with a seed-clustered bootstrap interval on the difference.",
            VisibilityEffectParams,
            _run_visibility_effect,
        ),
        (
            "lead_measures",
            "Select measures by planner separation on the discovery seeds, then count the separated planner pairs on the verification seeds using each planner's seed interval.",
            LeadMeasuresParams,
            _run_lead_measures,
        ),
        (
            "attribution_shares",
            "Pedestrian-contact closing shares of valid episodes, pooled from the contact parts, with the frozen-robot persistence and the pooled Probst responsibility.",
            AttributionSharesParams,
            _run_attribution_shares,
        ),
    )
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
        raise KeyError(f"Unknown analysis procedure {name!r}; available: {sorted(_PROCEDURES)}")
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
    parsed = params if isinstance(params, spec.params_model) else spec.params_model.model_validate(params or {})
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
