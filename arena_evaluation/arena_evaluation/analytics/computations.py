"""Registered Layer 3 computations producing schema-backed :class:`AnalysisResult` artifacts."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from arena_evaluation.storage.registry.analysis import (
    ClusterBootstrapCIAnalysis,
    CompositeRankingAnalysis,
    CorrelationMatrixAnalysis,
    DistributionBinsAnalysis,
    GroupSummaryAnalysis,
    PairedDiffAnalysis,
    RadarProfileAnalysis,
    SeriesProfileAnalysis,
)
from arena_evaluation.storage.registry.table import Table

from .catalog import BenchmarkCatalog
from .evidence import AnalysisResult, EvidenceStore
from .stats import cluster_bootstrap_ci, paired_treatment_diff

__all__ = [
    "COMPUTATIONS",
    "Computation",
    "get_computation",
    "list_computations",
    "register_computation",
    "resolve_metric_directions",
    "resolve_metric_units",
    "run_computation",
]

_WILSON_Z = 1.96
_SOURCE_TABLE = "v_episode_summary"


_FilterScalar = str | int | float | bool
_FilterValue = _FilterScalar | tuple[_FilterScalar, ...]


class _BaseParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str = _SOURCE_TABLE
    filter: dict[str, _FilterValue] | None = None
    exclude_references: bool = True

    @property
    def group_columns(self) -> tuple[str, ...]:
        """Group columns prepended to the fixed analysis schema."""
        return ()


class GroupSummaryParams(_BaseParams):
    group_cols: tuple[str, ...] = ()
    metrics: tuple[str, ...] = Field(min_length=1)
    trim_percentile: float = Field(default=0.0, ge=0.0, lt=0.5)
    error_mode: Literal["std", "sem", "ci95"] = "std"

    @property
    def group_columns(self) -> tuple[str, ...]:
        return self.group_cols


class DistributionBinsParams(_BaseParams):
    metric: str = Field(min_length=1)
    group_cols: tuple[str, ...] = ()
    nbins: int = Field(default=15, ge=1)
    density: bool = False

    @property
    def group_columns(self) -> tuple[str, ...]:
        return self.group_cols


class RadarProfileParams(_BaseParams):
    group_col: str = "planner"
    metrics: tuple[str, ...] = Field(min_length=1)
    log_scale: bool = False

    @property
    def group_columns(self) -> tuple[str, ...]:
        return (self.group_col,)


class CompositeRankingParams(_BaseParams):
    group_col: str = "planner"
    metrics: tuple[str, ...] = Field(min_length=1)
    weights: dict[str, float] | None = None
    top_n: int | None = Field(default=None, ge=1)

    @property
    def group_columns(self) -> tuple[str, ...]:
        return (self.group_col,)


class CorrelationMatrixParams(_BaseParams):
    metrics: tuple[str, ...] = Field(min_length=2)


class PairedDiffParams(_BaseParams):
    metric: str = Field(min_length=1)
    treatment: str = Field(min_length=1)
    control: str = Field(min_length=1)
    arm_col: str = "stage"
    pair_cols: tuple[str, ...] = ("planner", "seed")
    cluster_col: str = "seed"
    n_boot: int = Field(default=2000, ge=1)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 42


class ClusterBootstrapCIParams(_BaseParams):
    metric: str = Field(min_length=1)
    group_cols: tuple[str, ...] = ("planner",)
    cluster_col: str = "seed"
    n_boot: int = Field(default=2000, ge=1)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 42

    @property
    def group_columns(self) -> tuple[str, ...]:
        return self.group_cols


class SeriesProfileParams(_BaseParams):
    x: str
    metric: str
    group_cols: tuple[str, ...] = ()
    bin_x: float | None = Field(default=None, gt=0)
    reduce: Literal["mean", "max", "leq"] = "mean"
    error_mode: Literal["std", "sem", "ci95"] = "std"
    trim_percentile: float = Field(default=0.0, ge=0.0, lt=0.5)
    smooth_window: int | None = Field(default=None, ge=2)

    @property
    def group_columns(self) -> tuple[str, ...]:
        return self.group_cols


@dataclasses.dataclass(frozen=True, slots=True)
class Computation:
    """Registered named analytical computation with parameter model and output table schema."""

    name: str
    params: type[BaseModel]
    output: type[Table]
    description: str
    fn: Callable[[BenchmarkCatalog, Any], AnalysisResult]


COMPUTATIONS: dict[str, Computation] = {}


def register_computation(
    name: str,
    *,
    params: type[BaseModel],
    output: type[Table],
    description: str = "",
) -> Callable[
    [Callable[[BenchmarkCatalog, Any], AnalysisResult]],
    Callable[[BenchmarkCatalog, Any], AnalysisResult],
]:
    """Register a computation function under ``name`` with its param model and output table."""

    def _decorator(
        fn: Callable[[BenchmarkCatalog, Any], AnalysisResult],
    ) -> Callable[[BenchmarkCatalog, Any], AnalysisResult]:
        COMPUTATIONS[name] = Computation(
            name=name,
            params=params,
            output=output,
            description=description,
            fn=fn,
        )
        return fn

    return _decorator


def get_computation(name: str) -> Computation:
    """Look up a registered computation by name."""
    if name not in COMPUTATIONS:
        raise KeyError(
            f"Unknown computation '{name}'. Available: {sorted(COMPUTATIONS)}"
        )
    return COMPUTATIONS[name]


def list_computations() -> tuple[str, ...]:
    """Return all registered computation names in sorted order."""
    return tuple(sorted(COMPUTATIONS))


def _dim_metrics_rows(
    catalog: BenchmarkCatalog,
) -> dict[str, tuple[str | None, str | None]]:
    if "dim_metrics" not in catalog.table_names():
        return {}
    df = (
        catalog.get_table("dim_metrics").select("metric", "direction", "unit").collect()
    )
    rows = df.iter_rows()  # agent-contracts: ignore metadata dict boundary
    return {str(m): (d, u) for m, d, u in rows}


def resolve_metric_directions(
    catalog: BenchmarkCatalog, metrics: Sequence[str]
) -> dict[str, str]:
    """Resolve required ``higher``/``lower`` directions from ``dim_metrics``, raising on missing/NULL."""
    rows = _dim_metrics_rows(catalog)
    resolved: dict[str, str] = {}
    for metric in metrics:
        entry = rows.get(metric)
        direction = entry[0] if entry is not None else None
        if direction not in ("higher", "lower"):
            raise ValueError(
                f"Metric '{metric}' lacks a valid direction ('higher' or 'lower') in dim_metrics: {direction!r}"
            )
        resolved[metric] = direction
    return resolved


def resolve_metric_units(
    catalog: BenchmarkCatalog, metrics: Sequence[str]
) -> dict[str, str]:
    """Resolve metric unit strings from ``dim_metrics`` (empty string if NULL/absent)."""
    rows = _dim_metrics_rows(catalog)
    return {
        metric: (rows[metric][1] or "") if metric in rows else "" for metric in metrics
    }


def _allowed_sources(catalog: BenchmarkCatalog) -> list[str]:
    return [
        _SOURCE_TABLE,
        *(n for n in sorted(catalog.table_names()) if n.startswith("v_series_")),
    ]


def _prepare_source_frame(
    catalog: BenchmarkCatalog,
    *,
    source: str = _SOURCE_TABLE,
    row_filter: Mapping[str, _FilterValue] | None = None,
    exclude_references: bool,
    needed_cols: Sequence[str],
) -> tuple[pl.DataFrame, dict[str, int], tuple[int, ...]]:
    """Project ``needed_cols`` + ``is_reference``/``seed`` from an allowed source view and apply ``row_filter``."""
    allowed = _allowed_sources(catalog)
    if source not in allowed:
        raise ValueError(
            f"Unsupported computation source {source!r}. Allowed: {allowed}"
        )
    lf = catalog.get_table(source)
    filter_cols = list(row_filter) if row_filter else []
    select_cols = list(
        dict.fromkeys([*needed_cols, *filter_cols, "is_reference", "seed"])
    )
    lf = lf.select(select_cols)
    if row_filter:
        for col, val in row_filter.items():
            if isinstance(val, tuple):
                lf = lf.filter(pl.col(col).is_in(list(val)))
            else:
                lf = lf.filter(pl.col(col) == val)
    df = lf.collect()

    excluded: dict[str, int] = {}
    if exclude_references:
        before = df.height
        df = df.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))
        if before > df.height:
            excluded["references"] = before - df.height

    seeds = tuple(
        int(s) for s in df.get_column("seed").drop_nulls().unique().sort().to_numpy()
    )
    return df, excluded, seeds


def _validate_output_schema(
    df: pl.DataFrame, table_cls: type[Table], group_cols: Sequence[str] = ()
) -> None:
    """Enforce exact column order and Polars dtypes against a registered ``TableKind.ANALYSIS`` schema."""
    expected_tail = table_cls.polars_schema()
    expected_cols = [*group_cols, *expected_tail.names()]
    if df.columns != expected_cols:
        raise AssertionError(
            f"{table_cls.table_name} columns {df.columns} != expected {expected_cols}"
        )
    for col, dtype in expected_tail.items():
        if df.schema[col] != dtype:
            raise AssertionError(
                f"{table_cls.table_name}.{col} dtype {df.schema[col]} != {dtype}"
            )


@register_computation(
    "group_summary",
    params=GroupSummaryParams,
    output=GroupSummaryAnalysis,
    description="Per-group summary statistics (Wilson CI for booleans; mean/std/sem/ci95 for numeric).",
)
def _compute_group_summary(
    catalog: BenchmarkCatalog, p: GroupSummaryParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[*p.group_cols, *p.metrics],
    )
    units = resolve_metric_units(catalog, p.metrics)
    parts: list[pl.DataFrame] = []
    for metric in p.metrics:
        work = df
        if p.trim_percentile > 0.0 and work.height > 0:
            low_q = p.trim_percentile / 2.0
            high_q = 1.0 - low_q
            if p.group_cols:
                work = work.filter(
                    (
                        pl.col(metric)
                        >= pl.col(metric).quantile(low_q).over(p.group_cols)
                    )
                    & (
                        pl.col(metric)
                        <= pl.col(metric).quantile(high_q).over(p.group_cols)
                    )
                )
            else:
                work = work.filter(
                    (pl.col(metric) >= pl.col(metric).quantile(low_q))
                    & (pl.col(metric) <= pl.col(metric).quantile(high_q))
                )

        if work.schema[metric] == pl.Boolean:
            prob = pl.col(metric).cast(pl.Float64).mean()
            cnt = pl.col(metric).count().cast(pl.Int64)
            z = _WILSON_Z
            denom = 1.0 + (z * z) / cnt
            center = (prob + (z * z) / (2.0 * cnt)) / denom
            radicand = (prob * (1.0 - prob) / cnt) + (z * z) / (4.0 * cnt * cnt)
            half = z * radicand.sqrt() / denom
            std_expr = (prob * (1.0 - prob)).sqrt()
            sem_expr = std_expr / cnt.cast(pl.Float64).sqrt()
            err_plus = (center + half - prob).clip(lower_bound=0.0)
            err_minus = (prob - (center - half)).clip(lower_bound=0.0)
            aggs = [
                cnt.alias("n"),
                prob.alias("mean"),
                std_expr.alias("std"),
                sem_expr.alias("sem"),
                (prob - err_minus).alias("ci_low"),
                (prob + err_plus).alias("ci_high"),
                err_plus.alias("err_plus"),
                err_minus.alias("err_minus"),
            ]
        else:
            val = pl.col(metric).cast(pl.Float64)
            cnt = val.count().cast(pl.Int64)
            mean_expr = val.mean()
            raw_std = val.std()
            std_expr = pl.when(cnt >= 2).then(raw_std).otherwise(None).cast(pl.Float64)
            sem_expr = (
                pl.when(cnt >= 2)
                .then(raw_std / cnt.cast(pl.Float64).sqrt())
                .otherwise(None)
                .cast(pl.Float64)
            )
            ci_half = _WILSON_Z * sem_expr
            err_expr = (
                ci_half
                if p.error_mode == "ci95"
                else (sem_expr if p.error_mode == "sem" else std_expr)
            )
            aggs = [
                cnt.alias("n"),
                mean_expr.alias("mean"),
                std_expr.alias("std"),
                sem_expr.alias("sem"),
                (mean_expr - ci_half).alias("ci_low"),
                (mean_expr + ci_half).alias("ci_high"),
                err_expr.alias("err_plus"),
                err_expr.alias("err_minus"),
            ]

        grouped = (
            work.group_by(list(p.group_cols)).agg(aggs)
            if p.group_cols
            else work.select(aggs)
        )
        parts.append(
            grouped.with_columns(
                pl.lit(metric).cast(pl.String).alias("metric"),
                pl.lit(units[metric]).cast(pl.String).alias("unit"),
            ).select([*p.group_cols, *GroupSummaryAnalysis.polars_schema().names()])
        )

    out = pl.concat(parts).sort([*p.group_cols, "metric"])
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


@register_computation(
    "distribution_bins",
    params=DistributionBinsParams,
    output=DistributionBinsAnalysis,
    description="Fixed-edge histogram bins matching HistogramRenderer (nbins=15, +-1 flat widening).",
)
def _compute_distribution_bins(
    catalog: BenchmarkCatalog, p: DistributionBinsParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[*p.group_cols, p.metric],
    )
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    metric_f64 = pl.col(p.metric).cast(pl.Float64)
    valid = df.filter(metric_f64.is_not_null() & ~metric_f64.is_nan())
    if valid.is_empty():
        empty = pl.DataFrame(
            schema={
                **{c: df.schema[c] for c in p.group_cols},
                **DistributionBinsAnalysis.polars_schema(),
            }
        )
        return AnalysisResult(
            frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds
        )

    all_vals = valid.get_column(p.metric).cast(pl.Float64).to_numpy()
    low, high = float(all_vals.min()), float(all_vals.max())
    if low == high:
        low -= 1.0
        high += 1.0
    edges = np.linspace(low, high, p.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    groups: list[tuple[dict[str, object], np.ndarray]] = []
    if p.group_cols:
        for keys, sub in valid.sort(list(p.group_cols)).group_by(
            list(p.group_cols), maintain_order=True
        ):
            key_dict = dict(zip(p.group_cols, keys, strict=True))
            groups.append(
                (key_dict, sub.get_column(p.metric).cast(pl.Float64).to_numpy())
            )
    else:
        groups.append(({}, all_vals))

    rows: list[dict[str, object]] = []
    for key_dict, vals in groups:
        counts, _ = np.histogram(vals, bins=edges)
        total = float(counts.sum())
        for idx in range(p.nbins):
            cnt = int(counts[idx])
            dens = (
                (cnt / (total * width))
                if (p.density and total > 0.0 and width > 0.0)
                else ((cnt / total) if total > 0.0 else 0.0)
            )
            rows.append(
                {
                    **key_dict,
                    "metric": p.metric,
                    "unit": unit,
                    "bin_index": idx,
                    "bin_low": float(edges[idx]),
                    "bin_high": float(edges[idx + 1]),
                    "bin_center": float(centers[idx]),
                    "count": cnt,
                    "density": float(dens),
                }
            )

    out = pl.DataFrame(
        rows,
        schema={
            **{c: df.schema[c] for c in p.group_cols},
            **DistributionBinsAnalysis.polars_schema(),
        },
    )
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


def _radar_normalize(
    raw: np.ndarray, directions: Sequence[str], *, log_scale: bool
) -> np.ndarray:
    matrix = np.log1p(raw) if log_scale else raw.copy()
    for col_idx, direction in enumerate(directions):
        col = matrix[:, col_idx]
        valid = ~np.isnan(col)
        if not np.any(valid):
            continue
        col_valid = col[valid]
        high = float(col_valid.max())
        low = float(col_valid.min())
        if high == 0.0 and low == 0.0:
            col[valid] = 1.0
        elif direction == "higher":
            col[valid] = col_valid / high if high > 0.0 else 1.0
        elif low > 0.0:
            col[valid] = low / col_valid
        elif high == 0.0:
            col[valid] = col_valid / low
        else:
            col[valid] = 1.0 - col_valid / high
    return matrix


@register_computation(
    "radar_profile",
    params=RadarProfileParams,
    output=RadarProfileAnalysis,
    description="Capability tradeoff radar profile matching RadarRenderer normalization (null means stay null).",
)
def _compute_radar_profile(
    catalog: BenchmarkCatalog, p: RadarProfileParams
) -> AnalysisResult:
    directions = resolve_metric_directions(catalog, p.metrics)
    units = resolve_metric_units(catalog, p.metrics)
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[p.group_col, *p.metrics],
    )
    grouped = (
        df.group_by(p.group_col)
        .agg([pl.col(m).cast(pl.Float64).mean().alias(m) for m in p.metrics])
        .sort(p.group_col)
    )
    if grouped.is_empty():
        empty = pl.DataFrame(
            schema={
                p.group_col: df.schema[p.group_col],
                **RadarProfileAnalysis.polars_schema(),
            }
        )
        return AnalysisResult(
            frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds
        )

    raw = grouped.select(list(p.metrics)).to_numpy().astype(np.float64, copy=False)
    norm = _radar_normalize(
        raw, [directions[m] for m in p.metrics], log_scale=p.log_scale
    )
    norm_df = pl.DataFrame(
        {
            p.group_col: grouped.get_column(p.group_col),
            **{
                m: pl.Series(m, norm[:, idx], dtype=pl.Float64).fill_nan(None)
                for idx, m in enumerate(p.metrics)
            },
        }
    )
    raw_long = grouped.unpivot(
        index=p.group_col,
        on=list(p.metrics),
        variable_name="metric",
        value_name="raw_mean",
    )
    norm_long = norm_df.unpivot(
        index=p.group_col,
        on=list(p.metrics),
        variable_name="metric",
        value_name="normalized_score",
    )
    meta_df = pl.DataFrame(
        {
            "metric": list(p.metrics),
            "unit": [units[m] for m in p.metrics],
            "direction": [directions[m] for m in p.metrics],
        }
    )
    out = (
        raw_long.join(
            norm_long, on=[p.group_col, "metric"], how="inner", nulls_equal=True
        )
        .join(meta_df, on="metric", how="inner")
        .select([p.group_col, *RadarProfileAnalysis.polars_schema().names()])
        .sort([p.group_col, "metric"])
    )
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


def _run_composite_ranking(
    catalog: BenchmarkCatalog,
    p: CompositeRankingParams,
    *,
    default_top_n: int | None,
) -> AnalysisResult:
    directions = resolve_metric_directions(catalog, p.metrics)
    units = resolve_metric_units(catalog, p.metrics)
    weights = {m: float(p.weights.get(m, 1.0)) if p.weights else 1.0 for m in p.metrics}
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[p.group_col, *p.metrics],
    )
    grouped = (
        df.group_by(p.group_col)
        .agg([pl.col(m).cast(pl.Float64).mean().alias(m) for m in p.metrics])
        .sort(p.group_col)
    )
    group_rows = grouped.to_dicts()
    norm_scores: dict[tuple[int, str], float | None] = {}
    for m in p.metrics:
        vals = [r[m] for r in group_rows if r[m] is not None]
        if not vals:
            continue
        low, high = min(vals), max(vals)
        for idx, r in enumerate(group_rows):
            v = r[m]
            if v is None:
                norm_scores[(idx, m)] = None
            else:
                norm = 0.5 if high == low else (v - low) / (high - low)
                norm_scores[(idx, m)] = (
                    norm if directions[m] == "higher" else 1.0 - norm
                )

    scored_groups: list[tuple[int, float | None]] = []
    for idx, _ in enumerate(group_rows):
        pairs = [
            (weights[m], norm_scores[(idx, m)])
            for m in p.metrics
            if norm_scores.get((idx, m)) is not None
        ]
        total_w = sum(w for w, _ in pairs)
        comp_score = (
            sum(w * v for w, v in pairs if v is not None) / total_w
            if pairs and total_w > 0.0
            else None
        )
        scored_groups.append((idx, comp_score))

    scored_groups.sort(key=lambda item: (item[1] is None, -(item[1] or 0.0)))
    top_n = p.top_n if p.top_n is not None else default_top_n
    if top_n is not None:
        scored_groups = scored_groups[:top_n]

    rows: list[dict[str, object]] = []
    for rank_pos, (g_idx, comp_score) in enumerate(scored_groups, start=1):
        rank_val = rank_pos if comp_score is not None else None
        g_val = group_rows[g_idx][p.group_col]
        for m in p.metrics:
            rows.append(
                {
                    p.group_col: g_val,
                    "rank": rank_val,
                    "composite_score": comp_score,
                    "metric": m,
                    "unit": units[m],
                    "direction": directions[m],
                    "raw_mean": group_rows[g_idx][m],
                    "normalized_score": norm_scores.get((g_idx, m)),
                    "weight": weights[m],
                }
            )
    out = pl.DataFrame(
        rows,
        schema={
            p.group_col: df.schema[p.group_col],
            **CompositeRankingAnalysis.polars_schema(),
        },
    )
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


@register_computation(
    "composite_ranking",
    params=CompositeRankingParams,
    output=CompositeRankingAnalysis,
    description="Weighted min-max normalized composite ranking (unrounded scores).",
)
def _compute_composite_ranking(
    catalog: BenchmarkCatalog, p: CompositeRankingParams
) -> AnalysisResult:
    return _run_composite_ranking(catalog, p, default_top_n=None)


@register_computation(
    "top_n_ranking",
    params=CompositeRankingParams,
    output=CompositeRankingAnalysis,
    description="Top-N contestants by weighted normalized composite score (unrounded scores).",
)
def _compute_top_n_ranking(
    catalog: BenchmarkCatalog, p: CompositeRankingParams
) -> AnalysisResult:
    return _run_composite_ranking(catalog, p, default_top_n=3)


@register_computation(
    "correlation_matrix",
    params=CorrelationMatrixParams,
    output=CorrelationMatrixAnalysis,
    description="Pairwise-complete Pearson correlation matrix using pl.corr.",
)
def _compute_correlation_matrix(
    catalog: BenchmarkCatalog, p: CorrelationMatrixParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=p.metrics,
    )
    work = df.select([pl.col(m).cast(pl.Float64) for m in p.metrics])
    rows: list[dict[str, object]] = []
    for mx in p.metrics:
        for my in p.metrics:
            pair = work.select(
                n_pairs=(pl.col(mx).is_not_null() & pl.col(my).is_not_null())
                .sum()
                .cast(pl.Int64),
                r=pl.corr(mx, my).cast(pl.Float64),
            )
            n_pairs = int(pair["n_pairs"][0])
            raw_r = pair["r"][0]
            pearson_r = (
                float(raw_r)
                if raw_r is not None and math.isfinite(float(raw_r))
                else None
            )
            rows.append(
                {
                    "metric_x": mx,
                    "metric_y": my,
                    "pearson_r": pearson_r,
                    "n_pairs": n_pairs,
                }
            )
    out = pl.DataFrame(rows, schema=CorrelationMatrixAnalysis.polars_schema()).sort(
        ["metric_x", "metric_y"]
    )
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


@register_computation(
    "paired_diff",
    params=PairedDiffParams,
    output=PairedDiffAnalysis,
    description="Cluster-paired treatment minus control difference with bootstrap CI and p-value.",
)
def _compute_paired_diff(
    catalog: BenchmarkCatalog, p: PairedDiffParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[p.arm_col, *p.pair_cols, p.cluster_col, p.metric],
    )
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    res = paired_treatment_diff(
        df,
        p.metric,
        treatment_col=p.arm_col,
        treatment_a=p.treatment,
        treatment_b=p.control,
        pair_cols=p.pair_cols,
        cluster_col=p.cluster_col,
        n_boot=p.n_boot,
        ci_level=p.ci_level,
        random_seed=p.seed,
    )
    summary = res.summary_frame.select(
        pl.lit(p.metric).cast(pl.String).alias("metric"),
        pl.lit(unit).cast(pl.String).alias("unit"),
        pl.col("n_pairs").cast(pl.Int64),
        pl.col("mean_diff").cast(pl.Float64),
        pl.col("ci_low").cast(pl.Float64),
        pl.col("ci_high").cast(pl.Float64),
        pl.col("p_value").cast(pl.Float64),
        pl.col("n_clusters").cast(pl.Int64),
    )
    return AnalysisResult(
        frame=res.pairs_frame,
        summary=summary,
        excluded_rows=excluded,
        seeds_used=seeds,
    )


@register_computation(
    "cluster_bootstrap_ci",
    params=ClusterBootstrapCIParams,
    output=ClusterBootstrapCIAnalysis,
    description="Cluster-bootstrap confidence interval per group using Layer 3 stats.",
)
def _compute_cluster_bootstrap_ci(
    catalog: BenchmarkCatalog, p: ClusterBootstrapCIParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[*p.group_cols, p.cluster_col, p.metric],
    )
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    raw = cluster_bootstrap_ci(
        df,
        p.metric,
        group_cols=p.group_cols,
        cluster_col=p.cluster_col,
        n_boot=p.n_boot,
        ci_level=p.ci_level,
        random_seed=p.seed,
    )
    out = raw.select(
        *[pl.col(c) for c in p.group_cols],
        pl.lit(p.metric).cast(pl.String).alias("metric"),
        pl.lit(unit).cast(pl.String).alias("unit"),
        pl.col("mean").cast(pl.Float64).alias("estimate"),
        pl.col("ci_low").cast(pl.Float64),
        pl.col("ci_high").cast(pl.Float64),
        pl.col("std_error").cast(pl.Float64).alias("se"),
        pl.col("n_clusters").cast(pl.Int64),
        pl.col("n_obs").cast(pl.Int64).alias("n_observations"),
    )
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


def _smooth_window_expr(col_name: str, width: int) -> pl.Expr:
    pad_left = width // 2
    terms = [
        pl.when(pl.int_range(pl.len()) + offset < 0)
        .then(pl.col(col_name).first())
        .when(pl.int_range(pl.len()) + offset >= pl.len())
        .then(pl.col(col_name).last())
        .otherwise(pl.col(col_name).shift(-offset))
        for offset in range(-pad_left, width - pad_left)
    ]
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total / float(width)


def _apply_series_smoothing(
    df: pl.DataFrame, group_cols: Sequence[str], smooth_window: int
) -> pl.DataFrame:
    if df.height < 3:
        return df
    max_w = min(smooth_window, df.height)
    exprs: list[pl.Expr] = []
    for col_name in ("value", "err_plus", "err_minus"):
        branch: pl.Expr = _smooth_window_expr(col_name, max_w)
        for width in range(max_w - 1, 2, -1):
            branch = (
                pl.when(pl.len() == width)
                .then(_smooth_window_expr(col_name, width))
                .otherwise(branch)
            )
        expr = pl.when(pl.len() < 3).then(pl.col(col_name)).otherwise(branch)
        if group_cols:
            expr = expr.over(list(group_cols))
        exprs.append(expr.alias(col_name))
    return df.with_columns(exprs)


@register_computation(
    "series_profile",
    params=SeriesProfileParams,
    output=SeriesProfileAnalysis,
    description="Series-grain binned/trimmed/reduced profile along an x axis with optional smoothing.",
)
def _compute_series_profile(
    catalog: BenchmarkCatalog, p: SeriesProfileParams
) -> AnalysisResult:
    df, excluded, seeds = _prepare_source_frame(
        catalog,
        source=p.source,
        row_filter=p.filter,
        exclude_references=p.exclude_references,
        needed_cols=[*p.group_cols, p.x, p.metric],
    )
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    out_cols = [*p.group_cols, *SeriesProfileAnalysis.polars_schema().names()]
    work = df.filter(pl.col(p.x).is_not_null())
    if work.is_empty():
        empty_schema = {
            **{col: df.schema[col] for col in p.group_cols},
            **SeriesProfileAnalysis.polars_schema(),
        }
        empty = pl.DataFrame(schema=empty_schema).select(out_cols)
        return AnalysisResult(
            frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds
        )

    x_col = pl.col(p.x).cast(pl.Float64)
    x_expr = (
        (x_col / p.bin_x).round(0) * p.bin_x if p.bin_x is not None else x_col
    ).alias("x")
    work = work.with_columns(x_expr, pl.col(p.metric).cast(pl.Float64).alias("_y"))

    cell_keys = [*p.group_cols, "x"]
    if p.trim_percentile > 0.0:
        low_q = p.trim_percentile / 2.0
        high_q = 1.0 - low_q
        work = work.filter(
            (pl.col("_y") >= pl.col("_y").quantile(low_q).over(cell_keys))
            & (pl.col("_y") <= pl.col("_y").quantile(high_q).over(cell_keys))
        )

    n_expr = pl.col("_y").count().cast(pl.Int64).alias("n")
    if p.reduce == "leq":
        value_expr = (10.0 * (10.0 ** (pl.col("_y") / 10.0)).mean().log10()).alias(
            "value"
        )
    elif p.reduce == "max":
        value_expr = pl.col("_y").max().alias("value")
    else:
        value_expr = pl.col("_y").mean().alias("value")

    if p.reduce != "mean":
        err_expr = pl.lit(None, dtype=pl.Float64)
    else:
        std_expr = pl.col("_y").std(ddof=1)
        count_f64 = pl.col("_y").count().cast(pl.Float64)
        if p.error_mode == "std":
            raw_err = std_expr
        elif p.error_mode == "sem":
            raw_err = std_expr / count_f64.sqrt()
        else:
            raw_err = _WILSON_Z * std_expr / count_f64.sqrt()
        err_expr = (
            pl.when(pl.col("_y").count() >= 2)
            .then(raw_err)
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )

    agg = (
        work.group_by(cell_keys)
        .agg(
            n_expr,
            value_expr,
            err_expr.alias("err_plus"),
            err_expr.alias("err_minus"),
        )
        .sort(cell_keys)
    )
    if p.smooth_window is not None:
        agg = _apply_series_smoothing(agg, p.group_cols, p.smooth_window)

    out = agg.with_columns(
        pl.lit(p.metric, dtype=pl.String).alias("metric"),
        pl.lit(unit, dtype=pl.String).alias("unit"),
    ).select(out_cols)
    return AnalysisResult(
        frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds
    )


def run_computation(
    name: str,
    catalog: BenchmarkCatalog,
    params: Mapping[str, object] | None = None,
    *,
    evidence_store: EvidenceStore | None = None,
    purpose: str = "",
) -> AnalysisResult:
    """Validate parameters, run ``name``, check output schema, and optionally record evidence."""
    comp = get_computation(name)
    validated = comp.params.model_validate(dict(params or {}))
    result = comp.fn(catalog, validated)
    _validate_output_schema(result.summary, comp.output, validated.group_columns)
    resolved_snapshot = catalog.snapshot_id
    evidence_id = ""
    if evidence_store is not None:
        record = evidence_store.record(
            result.summary,
            kind="computation",
            operation=comp.name,
            purpose=purpose,
            params=validated.model_dump(mode="json"),
            snapshot_id=resolved_snapshot,
        )
        evidence_id = record.evidence_id
    return AnalysisResult(
        frame=result.frame,
        summary=result.summary,
        excluded_rows=result.excluded_rows,
        seeds_used=result.seeds_used,
        snapshot_id=resolved_snapshot,
        evidence_id=evidence_id,
    )
