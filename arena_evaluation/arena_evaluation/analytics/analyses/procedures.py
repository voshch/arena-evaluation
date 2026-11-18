"""Pure statistical analysis procedures built on BenchmarkCatalog and analytics.stats."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import resolve_metric_directions
from arena_evaluation.analytics.evidence import AnalysisResult, EvidenceStore
from arena_evaluation.analytics.stats import (
    cluster_bootstrap_ci,
    kendalls_w,
    paired_treatment_diff,
    rank_swap_matrix,
    split_discovery_verification_seeds,
)

_PAIRING_KEYS: tuple[str, ...] = ("seed", "world", "scenario", "crowd_condition")


class TriageParams(BaseModel):
    """Parameters for episode usability and missingness triage."""

    include_references: bool = False


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
    include_references: bool = False


class VisibilityEffectParams(BaseModel):
    """Parameters for H1 visibility effect test comparing blind vs visible rank swaps."""

    metric: str = "time_to_goal"
    n_boot: int = Field(default=500, ge=50)
    seed: int = 42
    include_references: bool = False


class LeadMeasuresParams(BaseModel):
    """Parameters for discovery/verification seed-split lead measure selection."""

    metrics: list[str] = Field(
        default_factory=lambda: ["time_to_goal", "energy_total_wh", "path_efficiency"]
    )
    seed: int = 42
    include_references: bool = False


class AttributionSharesParams(BaseModel):
    """Parameters for summed-parts contact attribution shares."""

    n_boot: int = Field(default=500, ge=50)
    seed: int = 42


@dataclasses.dataclass(frozen=True, slots=True)
class ProcedureSpec:
    """Registered named analysis procedure."""

    name: str
    description: str
    params_model: type[BaseModel]
    fn: Callable[[BenchmarkCatalog, BaseModel], AnalysisResult]


def _valid_episodes(
    catalog: BenchmarkCatalog, *, include_references: bool
) -> pl.DataFrame:
    lf = catalog.get_table("v_episode_summary")
    if not include_references:
        lf = lf.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))
    return lf.collect()


def _count_null_reasons(frame: pl.DataFrame, metric: str) -> dict[str, int]:
    null_count = int(frame[metric].null_count())
    if null_count == 0:
        return {}
    prefix = f"{metric}:"
    matched = int(
        frame.filter(
            pl.col(metric).is_null()
            & pl.col("fact_robot_null_reason").is_not_null()
            & pl.col("fact_robot_null_reason").str.contains(prefix, literal=True)
        ).height
    )
    if matched > 0:
        return {f"{metric}:recorded_null_reason": matched}
    return {f"{metric}:missing": null_count}


def _run_triage(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, TriageParams)
    df = _valid_episodes(catalog, include_references=params.include_references)
    grouped = (
        df.group_by(["planner", "world", "crowd_condition"])
        .agg(
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
            pl.col("success").cast(pl.Float64).mean().alias("success_rate"),
        )
        .sort(["planner", "world", "crowd_condition"])
    )
    excluded = {
        "invalid_episodes": int(
            df.filter(~pl.col("is_valid_for_ranking").fill_null(False)).height
        )
    }
    summary = pl.DataFrame(
        [
            {
                "total_episodes": len(df),
                "valid_episodes": int(
                    df.filter(pl.col("is_valid_for_ranking").fill_null(False)).height
                ),
                "excluded_episodes": excluded["invalid_episodes"],
            }
        ]
    )
    return AnalysisResult(
        frame=grouped,
        summary=summary,
        excluded_rows=excluded,
        snapshot_id=catalog.snapshot_id,
    )


def _run_compare(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, CompareParams)
    df = _valid_episodes(catalog, include_references=params.include_references).filter(
        pl.col("is_valid_for_ranking").fill_null(False)
    )
    if params.metric not in df.columns:
        return AnalysisResult(
            frame=pl.DataFrame(),
            summary=pl.DataFrame([{"reason": f"unknown metric {params.metric!r}"}]),
            excluded_rows={f"{params.metric}:unknown": len(df)},
            snapshot_id=catalog.snapshot_id,
        )

    sub = df.filter(pl.col("planner").is_in([params.planner_a, params.planner_b]))
    excluded = _count_null_reasons(sub, params.metric)
    valid = sub.filter(pl.col(params.metric).is_not_null())
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
                    }
                ]
            ),
            excluded_rows=excluded or {f"{params.metric}:all_null": sub.height},
            snapshot_id=catalog.snapshot_id,
        )

    pairing = [c for c in _PAIRING_KEYS if c in valid.columns]
    res = paired_treatment_diff(
        valid,
        params.metric,
        treatment_col="planner",
        treatment_a=params.planner_a,
        treatment_b=params.planner_b,
        pair_cols=pairing,
        cluster_col=params.cluster_col
        if params.cluster_col in valid.columns
        else "seed",
        n_boot=params.n_boot,
        ci_level=params.ci_level,
        random_seed=params.seed,
    )
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
                "cohens_dz": res.cohens_dz,
            }
        ]
    )
    seeds = tuple(int(s) for s in valid["seed"].drop_nulls().unique().sort())
    return AnalysisResult(
        frame=valid.select([*pairing, "planner", params.metric]),
        summary=summary,
        excluded_rows=excluded,
        seeds_used=seeds,
        snapshot_id=catalog.snapshot_id,
    )


def _rank_swaps_between(
    df: pl.DataFrame,
    group_col: str,
    group_a: object,
    group_b: object,
    metric: str,
    *,
    higher_is_better: bool,
) -> tuple[int, list[dict[str, Any]]]:
    a_df = (
        df.filter(pl.col(group_col) == group_a)
        .group_by("planner")
        .agg(pl.col(metric).cast(pl.Float64).mean().alias("val"))
    )
    b_df = (
        df.filter(pl.col(group_col) == group_b)
        .group_by("planner")
        .agg(pl.col(metric).cast(pl.Float64).mean().alias("val"))
    )
    joined = a_df.join(b_df, on="planner", suffix="_b")
    swaps: list[dict[str, Any]] = []
    rows = joined.sort("planner").to_dicts()
    for i, r1 in enumerate(rows):
        for r2 in rows[i + 1 :]:
            if (
                r1["val"] is None
                or r2["val"] is None
                or r1["val_b"] is None
                or r2["val_b"] is None
            ):
                continue
            d_a = float(r1["val"]) - float(r2["val"])
            d_b = float(r1["val_b"]) - float(r2["val_b"])
            if not higher_is_better:
                d_a, d_b = -d_a, -d_b
            if d_a * d_b < 0:
                swaps.append(
                    {
                        "planner_1": r1["planner"],
                        "planner_2": r2["planner"],
                        "group_a": str(group_a),
                        "group_b": str(group_b),
                    }
                )
    return len(swaps), swaps


def _run_ranking_stability(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, RankingStabilityParams)
    df = _valid_episodes(catalog, include_references=params.include_references).filter(
        pl.col("is_valid_for_ranking").fill_null(False)
        & pl.col(params.metric).is_not_null()
    )
    directions = resolve_metric_directions(catalog, (params.metric,))
    higher = directions[params.metric] == "higher"

    swap_rows: list[dict[str, Any]] = []
    conds = tuple(str(c) for c in df["crowd_condition"].drop_nulls().unique().sort())
    if len(conds) >= 2:
        _, swaps = _rank_swaps_between(
            df,
            "crowd_condition",
            conds[0],
            conds[1],
            params.metric,
            higher_is_better=higher,
        )
        swap_rows.extend(swaps)

    direction = "desc" if higher else "asc"
    has_multi_world = df["world"].n_unique() >= 2
    matrix = (
        rank_swap_matrix(
            df,
            params.metric,
            entity_col="planner",
            regime_col="world",
            direction=direction,
        )
        if has_multi_world
        else pl.DataFrame()
    )
    w_res = (
        kendalls_w(
            df,
            params.metric,
            entity_col="planner",
            judge_col="world",
            direction=direction,
        )
        if has_multi_world
        else None
    )
    summary = pl.DataFrame(
        [
            {
                "metric": params.metric,
                "n_swaps": len(swap_rows),
                "kendalls_w": w_res.w if w_res is not None else None,
            }
        ]
    )
    return AnalysisResult(
        frame=matrix if not matrix.is_empty() else pl.DataFrame(swap_rows),
        summary=summary,
        snapshot_id=catalog.snapshot_id,
    )


def _run_visibility_effect(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, VisibilityEffectParams)
    df = _valid_episodes(catalog, include_references=params.include_references).filter(
        pl.col("is_valid_for_ranking").fill_null(False)
        & pl.col(params.metric).is_not_null()
    )
    directions = resolve_metric_directions(catalog, (params.metric,))
    higher = directions[params.metric] == "higher"

    cond_swaps, _ = _rank_swaps_between(
        df,
        "crowd_condition",
        "blind",
        "visible",
        params.metric,
        higher_is_better=higher,
    )
    seeds = tuple(int(s) for s in df["seed"].drop_nulls().unique().sort())
    mid = len(seeds) // 2
    half_df = df.with_columns(
        pl.when(pl.col("seed").is_in(list(seeds[:mid])))
        .then(pl.lit("h1"))
        .otherwise(pl.lit("h2"))
        .alias("__seed_half")
    )
    seed_swaps, _ = _rank_swaps_between(
        half_df, "__seed_half", "h1", "h2", params.metric, higher_is_better=higher
    )
    effect_detected = cond_swaps > seed_swaps
    summary = pl.DataFrame(
        [
            {
                "metric": params.metric,
                "condition_swaps": cond_swaps,
                "seed_half_swaps": seed_swaps,
                "swap_delta": cond_swaps - seed_swaps,
                "effect_detected": effect_detected,
            }
        ]
    )
    return AnalysisResult(
        frame=summary,
        summary=summary,
        seeds_used=tuple(seeds),
        snapshot_id=catalog.snapshot_id,
    )


def _run_lead_measures(catalog: BenchmarkCatalog, params: BaseModel) -> AnalysisResult:
    assert isinstance(params, LeadMeasuresParams)
    df = _valid_episodes(catalog, include_references=params.include_references).filter(
        pl.col("is_valid_for_ranking").fill_null(False)
    )
    cohort_df = split_discovery_verification_seeds(df)
    disc_seeds = tuple(
        int(s)
        for s in cohort_df.filter(pl.col("cohort") == "discovery")["seed"]
        .drop_nulls()
        .unique()
        .sort()
    )
    ver_seeds = tuple(
        int(s)
        for s in cohort_df.filter(pl.col("cohort") == "verification")["seed"]
        .drop_nulls()
        .unique()
        .sort()
    )
    seeds = sorted({*disc_seeds, *ver_seeds})
    ver_df = cohort_df.filter(pl.col("cohort") == "verification")
    rows: list[dict[str, Any]] = []
    for metric in params.metrics:
        if metric not in ver_df.columns:
            continue
        non_null = ver_df.filter(pl.col(metric).is_not_null())
        if non_null.is_empty():
            continue
        by_planner = non_null.group_by("planner").agg(
            pl.col(metric).cast(pl.Float64).mean().alias("mean_val")
        )
        hi = by_planner["mean_val"].max()
        lo = by_planner["mean_val"].min()
        spread = float(hi - lo) if (hi is not None and lo is not None) else None
        rows.append({"metric": metric, "verification_spread": spread})

    out_frame = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={"metric": pl.String, "verification_spread": pl.Float64}
        )
    )
    summary = pl.DataFrame(
        [
            {
                "discovery_seeds": list(disc_seeds),
                "verification_seeds": list(ver_seeds),
                "n_selected_measures": len(rows),
            }
        ]
    )
    return AnalysisResult(
        frame=out_frame,
        summary=summary,
        seeds_used=tuple(seeds),
        snapshot_id=catalog.snapshot_id,
    )


def _run_attribution_shares(
    catalog: BenchmarkCatalog, params: BaseModel
) -> AnalysisResult:
    assert isinstance(params, AttributionSharesParams)
    contacts = (
        catalog.get_table("v_contacts")
        .join(
            catalog.get_table("v_episode_summary").select(
                ["run_id", "episode_id", "robot_id", "planner", "crowd_condition"]
            ),
            on=["run_id", "episode_id", "robot_id"],
            how="left",
        )
        .collect()
    )
    if contacts.is_empty():
        return AnalysisResult(
            frame=pl.DataFrame(),
            summary=pl.DataFrame([{"n_contacts": 0}]),
            snapshot_id=catalog.snapshot_id,
        )

    grouped = (
        contacts.group_by(["planner", "crowd_condition"])
        .agg(
            pl.len().cast(pl.Int64).alias("n_contacts"),
            pl.col("v_close_robot_mps")
            .clip(lower_bound=0.0)
            .sum()
            .cast(pl.Float64)
            .alias("robot_closing_sum"),
            pl.col("v_close_target_mps")
            .clip(lower_bound=0.0)
            .sum()
            .cast(pl.Float64)
            .alias("target_closing_sum"),
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
        .sort(["planner", "crowd_condition"])
    )

    ci_df = cluster_bootstrap_ci(
        contacts,
        "robot_closing_share",
        group_cols=["planner"],
        cluster_col="episode_id",
        n_boot=params.n_boot,
        random_seed=params.seed,
    )
    return AnalysisResult(
        frame=grouped,
        summary=ci_df,
        snapshot_id=catalog.snapshot_id,
    )


_PROCEDURES: dict[str, ProcedureSpec] = {
    "triage": ProcedureSpec(
        name="triage",
        description="Count usable and excluded episodes across planners, worlds, and conditions.",
        params_model=TriageParams,
        fn=_run_triage,
    ),
    "compare": ProcedureSpec(
        name="compare",
        description="Paired planner comparison with cluster-bootstrap CI and exclusion counts.",
        params_model=CompareParams,
        fn=_run_compare,
    ),
    "ranking_stability": ProcedureSpec(
        name="ranking_stability",
        description="Rank repeatability across worlds and conditions with Kendall's W.",
        params_model=RankingStabilityParams,
        fn=_run_ranking_stability,
    ),
    "visibility_effect": ProcedureSpec(
        name="visibility_effect",
        description="Test whether planner rank swaps differ between blind and visible crowds.",
        params_model=VisibilityEffectParams,
        fn=_run_visibility_effect,
    ),
    "lead_measures": ProcedureSpec(
        name="lead_measures",
        description="Select separating measures on discovery seeds and check on verification seeds.",
        params_model=LeadMeasuresParams,
        fn=_run_lead_measures,
    ),
    "attribution_shares": ProcedureSpec(
        name="attribution_shares",
        description="Compute closing-velocity attribution shares from summed parts.",
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
        )
    return result
