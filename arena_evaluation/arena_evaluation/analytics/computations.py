"""Registered Layer 3 computations producing schema-backed :class:`AnalysisResult` artifacts."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from arena_evaluation.storage.registry import NullReason
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
    "MetricDirections",
    "get_computation",
    "list_computations",
    "null_reason_counts",
    "register_computation",
    "resolve_metric_directions",
    "resolve_metric_units",
    "run_computation",
]

_WILSON_Z = 1.96
_SOURCE_TABLE = "v_episode_summary"
_VALIDITY_COLUMN = "is_valid_for_ranking"
_NULL_REASON_COLUMNS: tuple[str, ...] = (
    "dim_episode_null_reason",
    "dim_robot_null_reason",
    "fact_episode_null_reason",
    "fact_robot_null_reason",
)


class MetricDirection(StrEnum):
    """The better side of a metric, the vocabulary ``dim_metrics.direction`` publishes."""

    HIGHER = "higher"
    LOWER = "lower"


_DIRECTIONS: tuple[str, ...] = tuple(direction.value for direction in MetricDirection)

_FilterScalar = str | int | float | bool
_FilterValue = _FilterScalar | tuple[_FilterScalar, ...]


class _BaseParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str = _SOURCE_TABLE
    filter: dict[str, _FilterValue] | None = None
    exclude_references: bool = True
    valid_only: bool = Field(
        default=True,
        description=("Drop every episode the benchmark flagged invalid for ranking; the excluded count and each NULL metric's reason are returned with the result."),
    )
    directions: dict[str, str] | None = Field(
        default=None,
        description=("Better-side override per metric; a metric left out resolves from dim_metrics and then from the metric registry."),
    )

    @property
    def group_columns(self) -> tuple[str, ...]:
        """Group columns prepended to the fixed analysis schema."""
        return ()


class _GroupedParams(_BaseParams):
    group_cols: tuple[str, ...] = ()

    @property
    def group_columns(self) -> tuple[str, ...]:
        return self.group_cols


class _SingleGroupParams(_BaseParams):
    group_col: str = "planner"

    @property
    def group_columns(self) -> tuple[str, ...]:
        return (self.group_col,)


class GroupSummaryParams(_GroupedParams):
    metrics: tuple[str, ...] = Field(min_length=1)
    trim_percentile: float = Field(default=0.0, ge=0.0, lt=0.5)
    error_mode: Literal["std", "sem", "ci95"] = "std"


class DistributionBinsParams(_GroupedParams):
    metric: str = Field(min_length=1)
    nbins: int = Field(default=15, ge=1)
    density: bool = False


class RadarProfileParams(_SingleGroupParams):
    metrics: tuple[str, ...] = Field(min_length=1)
    log_scale: bool = False


class CompositeRankingParams(_SingleGroupParams):
    metrics: tuple[str, ...] = Field(min_length=1)
    weights: dict[str, float] | None = None
    top_n: int | None = Field(default=None, ge=1)


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


class ClusterBootstrapCIParams(_GroupedParams):
    group_cols: tuple[str, ...] = ("planner",)
    metric: str = Field(min_length=1)
    cluster_col: str = "seed"
    n_boot: int = Field(default=2000, ge=1)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 42


class SeriesProfileParams(_GroupedParams):
    x: str
    metric: str
    bin_x: float | None = Field(default=None, gt=0)
    reduce: Literal["mean", "max", "leq"] = "mean"
    error_mode: Literal["std", "sem", "ci95"] = "std"
    trim_percentile: float = Field(default=0.0, ge=0.0, lt=0.5)
    smooth_window: int | None = Field(default=None, ge=2)


_ComputeFn = Callable[[BenchmarkCatalog, Any], AnalysisResult]


@dataclasses.dataclass(frozen=True, slots=True)
class Computation:
    """Registered named analytical computation with parameter model and output table schema."""

    name: str
    params: type[_BaseParams]
    output: type[Table]
    description: str
    fn: _ComputeFn


COMPUTATIONS: dict[str, Computation] = {}


def register_computation(
    name: str,
    *,
    params: type[_BaseParams],
    output: type[Table],
    description: str = "",
) -> Callable[[_ComputeFn], _ComputeFn]:
    """Register a computation function under ``name`` with its param model and output table."""

    def _decorator(fn: _ComputeFn) -> _ComputeFn:
        COMPUTATIONS[name] = Computation(name=name, params=params, output=output, description=description, fn=fn)
        return fn

    return _decorator


def get_computation(name: str) -> Computation:
    """Look up a registered computation by name."""
    if name not in COMPUTATIONS:
        raise KeyError(f"Unknown computation '{name}'. Available: {sorted(COMPUTATIONS)}")
    return COMPUTATIONS[name]


def list_computations() -> tuple[str, ...]:
    """Return all registered computation names in sorted order."""
    return tuple(sorted(COMPUTATIONS))


def _dim_metrics_rows(
    catalog: BenchmarkCatalog,
) -> dict[str, tuple[str | None, str | None]]:
    """Read the metric dictionary once, through the catalog's single accessor."""
    df = catalog.metric_dictionary().select("metric", "direction", "unit").collect()
    rows = df.iter_rows()  # agent-contracts: ignore metadata dict boundary
    return {str(m): (d, u) for m, d, u in rows}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MetricDirections:
    """The ``higher``/``lower`` side of every requested metric plus where each side was declared."""

    by_metric: Mapping[str, str]
    from_override: tuple[str, ...] = ()

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """One line per non-default direction source, carried on the analysis result."""
        if not self.from_override:
            return ()
        return (f"directions_from_param={len(self.from_override)}:{','.join(self.from_override)}",)


def resolve_metric_directions(
    catalog: BenchmarkCatalog,
    metrics: Sequence[str],
    *,
    declared: Mapping[str, str] | None = None,
) -> MetricDirections:
    """Resolve each metric's ``higher``/``lower`` side from the caller override or ``dim_metrics``, one error naming every unresolved metric."""
    rows = _dim_metrics_rows(catalog)
    overrides = dict(declared or {})
    by_metric: dict[str, str] = {}
    from_override: list[str] = []
    unresolved: list[str] = []
    for metric in metrics:
        override = overrides.get(metric)
        if override is not None:
            if override not in _DIRECTIONS:
                raise ValueError(f"Metric '{metric}' declares direction {override!r}; expected one of {list(_DIRECTIONS)}.")
            by_metric[metric] = override
            from_override.append(metric)
            continue
        published = rows.get(metric, (None, None))[0]
        if published in _DIRECTIONS:
            by_metric[metric] = str(published)
            continue
        unresolved.append(metric)
    if unresolved:
        raise ValueError(f"{len(unresolved)} metric(s) have no 'higher'/'lower' direction in dim_metrics: {unresolved}. Publish them in the metric dictionary or pass them through the computation's directions parameter.")
    return MetricDirections(by_metric=by_metric, from_override=tuple(from_override))


def resolve_metric_units(catalog: BenchmarkCatalog, metrics: Sequence[str]) -> dict[str, str]:
    """Resolve metric unit strings from ``dim_metrics`` (empty string if NULL/absent)."""
    rows = _dim_metrics_rows(catalog)
    return {metric: (rows[metric][1] or "") if metric in rows else "" for metric in metrics}


def null_reason_counts(df: pl.DataFrame, columns: Sequence[str], reason_columns: Sequence[str]) -> dict[str, int]:
    """Count NULL cells of every ``columns`` entry by the ``<column>:<code>`` reason its row carries."""
    if df.is_empty():
        return {}
    reasons = df.select(pl.concat_str([pl.col(name).fill_null("") for name in reason_columns], separator=";").alias("_reason"))
    frame = df.select(*[pl.col(name) for name in dict.fromkeys(columns)]).with_columns(reasons.get_column("_reason"))
    counts: dict[str, int] = {}
    for column in dict.fromkeys(columns):
        null_rows = frame.filter(pl.col(column).is_null()).with_row_index("_row")
        if null_rows.is_empty():
            continue
        codes = (
            null_rows.select("_row", pl.col("_reason").str.split(";").alias("token"))
            .explode("token", empty_as_null=True)
            .filter(pl.col("token").str.starts_with(f"{column}:"))
            .with_columns(pl.col("token").str.slice(len(column) + 1).alias("code"))
            .unique(subset=["_row", "code"])
            .group_by("code")
            .agg(pl.len().alias("rows"))
        )
        attributed = 0
        # agent-contracts: ignore AC004 - per-reason counts are result metadata, not data rows
        for row in codes.iter_rows(named=True):
            counts[f"{column}:{row['code']}"] = int(row["rows"])
            attributed += int(row["rows"])
        unreasoned = null_rows.height - attributed
        if unreasoned > 0:
            key = f"{column}:{NullReason.NOT_RECORDED}"
            counts[key] = counts.get(key, 0) + unreasoned
    return counts


def _prepare(
    catalog: BenchmarkCatalog,
    params: _BaseParams,
    needed_cols: Sequence[str],
) -> tuple[pl.DataFrame, dict[str, int], tuple[int, ...]]:
    """Project ``needed_cols`` from the params' source view and drop excluded rows, counting every exclusion by reason."""
    allowed = [
        _SOURCE_TABLE,
        *(n for n in sorted(catalog.table_names()) if n.startswith("v_series_")),
    ]
    if params.source not in allowed:
        raise ValueError(f"Unsupported computation source {params.source!r}. Allowed: {allowed}")
    lf = catalog.get_table(params.source)
    row_filter = params.filter or {}
    available = lf.collect_schema().names()
    reason_cols = [name for name in _NULL_REASON_COLUMNS if name in available]
    select_cols = list(
        dict.fromkeys(
            [
                *needed_cols,
                *row_filter,
                "is_reference",
                _VALIDITY_COLUMN,
                "seed",
                *reason_cols,
            ]
        )
    )
    lf = lf.select(select_cols)
    for col, val in row_filter.items():
        lf = lf.filter(pl.col(col).is_in(list(val)) if isinstance(val, tuple) else pl.col(col) == val)
    df = lf.collect()

    excluded: dict[str, int] = {}
    for enabled, column, reason, keep in (
        (params.exclude_references, "is_reference", "references", False),
        (params.valid_only, _VALIDITY_COLUMN, "invalid_for_ranking", True),
    ):
        if not enabled:
            continue
        before = df.height
        valid = pl.col(column).fill_null(False).cast(pl.Boolean)
        df = df.filter(valid if keep else ~valid)
        if before > df.height:
            excluded[reason] = before - df.height

    if reason_cols:
        excluded.update(null_reason_counts(df, needed_cols, reason_cols))
        df = df.drop(reason_cols)

    seeds = tuple(int(s) for s in df.get_column("seed").drop_nulls().unique().sort().to_numpy())
    return df.drop(_VALIDITY_COLUMN), excluded, seeds


def _validate_output_schema(df: pl.DataFrame, table_cls: type[Table], group_cols: Sequence[str] = ()) -> None:
    """Enforce exact column order and Polars dtypes against a registered ``TableKind.ANALYSIS`` schema."""
    expected_tail = table_cls.polars_schema()
    expected_cols = [*group_cols, *expected_tail.names()]
    if df.columns != expected_cols:
        raise AssertionError(f"{table_cls.table_name} columns {df.columns} != expected {expected_cols}")
    for col, dtype in expected_tail.items():
        if df.schema[col] != dtype:
            raise AssertionError(f"{table_cls.table_name}.{col} dtype {df.schema[col]} != {dtype}")


def _compute_group_summary(catalog: BenchmarkCatalog, p: GroupSummaryParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, [*p.group_cols, *p.metrics])
    units = resolve_metric_units(catalog, p.metrics)
    analysis_cols = GroupSummaryAnalysis.polars_schema().names()
    parts: list[pl.DataFrame] = []
    for metric in p.metrics:
        work = df
        if p.trim_percentile > 0.0 and work.height > 0:
            low_q = p.trim_percentile / 2.0
            high_q = 1.0 - low_q
            bounds = [pl.col(metric).quantile(q) for q in (low_q, high_q)]
            if p.group_cols:
                bounds = [bound.over(p.group_cols) for bound in bounds]
            work = work.filter((pl.col(metric) >= bounds[0]) & (pl.col(metric) <= bounds[1]))

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
            sem_expr = pl.when(cnt >= 2).then(raw_std / cnt.cast(pl.Float64).sqrt()).otherwise(None).cast(pl.Float64)
            ci_half = _WILSON_Z * sem_expr
            err_expr = ci_half if p.error_mode == "ci95" else (sem_expr if p.error_mode == "sem" else std_expr)
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

        grouped = work.group_by(list(p.group_cols)).agg(aggs) if p.group_cols else work.select(aggs)
        parts.append(
            grouped.with_columns(
                pl.lit(metric).cast(pl.String).alias("metric"),
                pl.lit(units[metric]).cast(pl.String).alias("unit"),
            ).select([*p.group_cols, *analysis_cols])
        )

    out = pl.concat(parts).sort([*p.group_cols, "metric"])
    return AnalysisResult(frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds)


def _compute_distribution_bins(catalog: BenchmarkCatalog, p: DistributionBinsParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, [*p.group_cols, p.metric])
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    schema = {
        **{c: df.schema[c] for c in p.group_cols},
        **DistributionBinsAnalysis.polars_schema(),
    }
    metric_f64 = pl.col(p.metric).cast(pl.Float64)
    valid = df.filter(metric_f64.is_not_null() & ~metric_f64.is_nan())
    if valid.is_empty():
        empty = pl.DataFrame(schema=schema)
        return AnalysisResult(frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds)

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
        for keys, sub in valid.sort(list(p.group_cols)).group_by(list(p.group_cols), maintain_order=True):
            groups.append(
                (
                    dict(zip(p.group_cols, keys, strict=True)),
                    sub.get_column(p.metric).cast(pl.Float64).to_numpy(),
                )
            )
    else:
        groups.append(({}, all_vals))

    rows: list[dict[str, object]] = []
    for key_dict, vals in groups:
        counts, _ = np.histogram(vals, bins=edges)
        total = float(counts.sum())
        for idx in range(p.nbins):
            cnt = int(counts[idx])
            scale = total * width if p.density and total > 0.0 and width > 0.0 else total
            density = cnt / scale if scale > 0.0 else 0.0
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
                    "density": float(density),
                }
            )

    out = pl.DataFrame(rows, schema=schema)
    return AnalysisResult(frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds)


def _compute_radar_profile(catalog: BenchmarkCatalog, p: RadarProfileParams) -> AnalysisResult:
    directions = resolve_metric_directions(catalog, p.metrics, declared=p.directions)
    units = resolve_metric_units(catalog, p.metrics)
    df, excluded, seeds = _prepare(catalog, p, [p.group_col, *p.metrics])
    grouped = df.group_by(p.group_col).agg([pl.col(m).cast(pl.Float64).mean().alias(m) for m in p.metrics]).sort(p.group_col)
    if grouped.is_empty():
        empty = pl.DataFrame(
            schema={
                p.group_col: df.schema[p.group_col],
                **RadarProfileAnalysis.polars_schema(),
            }
        )
        return AnalysisResult(frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds)

    raw = grouped.select(list(p.metrics)).to_numpy().astype(np.float64, copy=False)
    norm = np.full_like(raw, np.nan, dtype=np.float64)
    untransformable: dict[str, int] = {}
    for col_idx, (metric, direction) in enumerate(
        zip(
            p.metrics,
            (MetricDirection(directions.by_metric[m]) for m in p.metrics),
            strict=True,
        )
    ):
        col = raw[:, col_idx].astype(np.float64, copy=True)
        valid = ~np.isnan(col)
        if p.log_scale:
            loggable = valid & (col > -1.0)
            rejected = int(np.count_nonzero(valid & ~loggable))
            if rejected:
                untransformable["log1p_domain"] = untransformable.get("log1p_domain", 0) + rejected
            col = np.log1p(np.where(loggable, col, np.nan))
            valid = loggable
        if not np.any(valid):
            continue
        col_valid = col[valid]
        high = float(col_valid.max())
        low = float(col_valid.min())
        match direction:
            case MetricDirection.HIGHER | MetricDirection.LOWER if high == 0.0 and low == 0.0:
                normalized = np.ones_like(col_valid)
            case MetricDirection.HIGHER if high > 0.0:
                normalized = col_valid / high
            case MetricDirection.HIGHER:
                untransformable[metric] = untransformable.get(metric, 0) + int(col_valid.size)
                continue
            case MetricDirection.LOWER if low > 0.0:
                normalized = low / col_valid
            case MetricDirection.LOWER if high <= 0.0:
                normalized = col_valid / low
            case MetricDirection.LOWER:
                normalized = 1.0 - col_valid / high
            case _:
                raise ValueError(f"metric {metric!r} declares direction {direction!r}; expected one of {list(_DIRECTIONS)}")
        column = np.full(col.shape, np.nan, dtype=np.float64)
        column[valid] = np.clip(normalized, 0.0, 1.0)
        norm[:, col_idx] = column

    norm_df = pl.DataFrame(
        {
            p.group_col: grouped.get_column(p.group_col),
            **{m: pl.Series(m, norm[:, idx], dtype=pl.Float64).fill_nan(None) for idx, m in enumerate(p.metrics)},
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
            "direction": [directions.by_metric[m] for m in p.metrics],
        }
    )
    out = raw_long.join(norm_long, on=[p.group_col, "metric"], how="inner", nulls_equal=True).join(meta_df, on="metric", how="inner").select([p.group_col, *RadarProfileAnalysis.polars_schema().names()]).sort([p.group_col, "metric"])
    return AnalysisResult(
        frame=out,
        summary=out,
        excluded_rows=excluded,
        seeds_used=seeds,
        diagnostics=(
            *directions.diagnostics,
            *(f"radar_untransformable:{metric}={count}" for metric, count in sorted(untransformable.items())),
        ),
    )


def _run_composite_ranking(
    catalog: BenchmarkCatalog,
    p: CompositeRankingParams,
    *,
    default_top_n: int | None = None,
) -> AnalysisResult:
    directions = resolve_metric_directions(catalog, p.metrics, declared=p.directions)
    units = resolve_metric_units(catalog, p.metrics)
    weights = {m: float(p.weights.get(m, 1.0)) if p.weights else 1.0 for m in p.metrics}
    df, excluded, seeds = _prepare(catalog, p, [p.group_col, *p.metrics])
    grouped = df.group_by(p.group_col).agg([pl.col(m).cast(pl.Float64).mean().alias(m) for m in p.metrics]).sort(p.group_col)
    group_rows = grouped.to_dicts()
    norm_scores: dict[tuple[int, str], float | None] = {}
    for metric in p.metrics:
        vals = [row[metric] for row in group_rows if row[metric] is not None]
        if not vals:
            continue
        low, high = min(vals), max(vals)
        higher_is_better = MetricDirection(directions.by_metric[metric]) is MetricDirection.HIGHER
        for idx, row in enumerate(group_rows):
            value = row[metric]
            normalized = None if value is None else 0.5 if high == low else (value - low) / (high - low)
            norm_scores[(idx, metric)] = normalized if normalized is None or higher_is_better else 1.0 - normalized

    scored: list[tuple[int, float | None]] = []
    for idx in range(len(group_rows)):
        pairs = [(weights[metric], norm_scores[(idx, metric)]) for metric in p.metrics if norm_scores.get((idx, metric)) is not None]
        total_w = sum(weight for weight, _ in pairs)
        scored.append(
            (
                idx,
                sum(weight * value for weight, value in pairs if value is not None) / total_w if pairs and total_w > 0.0 else None,
            )
        )

    scored.sort(key=lambda item: (item[1] is None, -(item[1] or 0.0)))
    top_n = p.top_n if p.top_n is not None else default_top_n
    if top_n is not None:
        scored = scored[:top_n]

    rows: list[dict[str, object]] = []
    for rank_pos, (group_idx, comp_score) in enumerate(scored, start=1):
        rank_val = rank_pos if comp_score is not None else None
        group_row = group_rows[group_idx]
        for metric in p.metrics:
            rows.append(
                {
                    p.group_col: group_row[p.group_col],
                    "rank": rank_val,
                    "composite_score": comp_score,
                    "metric": metric,
                    "unit": units[metric],
                    "direction": directions.by_metric[metric],
                    "raw_mean": group_row[metric],
                    "normalized_score": norm_scores.get((group_idx, metric)),
                    "weight": weights[metric],
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
        frame=out,
        summary=out,
        excluded_rows=excluded,
        seeds_used=seeds,
        diagnostics=directions.diagnostics,
    )


def _compute_correlation_matrix(catalog: BenchmarkCatalog, p: CorrelationMatrixParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, p.metrics)
    work = df.select([pl.col(m).cast(pl.Float64) for m in p.metrics])
    rows: list[dict[str, object]] = []
    for mx in p.metrics:
        for my in p.metrics:
            pair = work.select(
                n_pairs=(pl.col(mx).is_not_null() & pl.col(my).is_not_null()).sum().cast(pl.Int64),
                r=pl.corr(mx, my).cast(pl.Float64),
            )
            raw_r = pair["r"][0]
            rows.append(
                {
                    "metric_x": mx,
                    "metric_y": my,
                    "pearson_r": (float(raw_r) if raw_r is not None and math.isfinite(float(raw_r)) else None),
                    "n_pairs": int(pair["n_pairs"][0]),
                }
            )
    out = pl.DataFrame(rows, schema=CorrelationMatrixAnalysis.polars_schema()).sort(["metric_x", "metric_y"])
    return AnalysisResult(frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds)


def _compute_paired_diff(catalog: BenchmarkCatalog, p: PairedDiffParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, [p.arm_col, *p.pair_cols, p.cluster_col, p.metric])
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
    effect_note = (f"cohens_dz_undefined:std_diff:{NullReason.ZERO_DENOMINATOR.value}",) if res.cohens_dz is None and res.n_pairs > 0 else ()
    return AnalysisResult(
        frame=res.pairs_frame,
        summary=summary,
        excluded_rows=excluded,
        seeds_used=seeds,
        diagnostics=effect_note,
    )


def _compute_cluster_bootstrap_ci(catalog: BenchmarkCatalog, p: ClusterBootstrapCIParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, [*p.group_cols, p.cluster_col, p.metric])
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
    dropped = int(raw["dropped_rows"][0]) if raw.height else 0
    if dropped:
        excluded = {**excluded, f"{p.metric}:{NullReason.NOT_RECORDED}": dropped}
    return AnalysisResult(frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds)


def _compute_series_profile(catalog: BenchmarkCatalog, p: SeriesProfileParams) -> AnalysisResult:
    df, excluded, seeds = _prepare(catalog, p, [*p.group_cols, p.x, p.metric])
    unit = resolve_metric_units(catalog, (p.metric,))[p.metric]
    out_cols = [*p.group_cols, *SeriesProfileAnalysis.polars_schema().names()]
    work = df.filter(pl.col(p.x).is_not_null())
    if work.is_empty():
        empty = pl.DataFrame(
            schema={
                **{col: df.schema[col] for col in p.group_cols},
                **SeriesProfileAnalysis.polars_schema(),
            }
        ).select(out_cols)
        return AnalysisResult(frame=empty, summary=empty, excluded_rows=excluded, seeds_used=seeds)

    x_col = pl.col(p.x).cast(pl.Float64)
    x_expr = ((x_col / p.bin_x).round(0) * p.bin_x if p.bin_x is not None else x_col).alias("x")
    work = work.with_columns(x_expr, pl.col(p.metric).cast(pl.Float64).alias("_y"))

    cell_keys = [*p.group_cols, "x"]
    if p.trim_percentile > 0.0:
        low_q = p.trim_percentile / 2.0
        high_q = 1.0 - low_q
        work = work.filter((pl.col("_y") >= pl.col("_y").quantile(low_q).over(cell_keys)) & (pl.col("_y") <= pl.col("_y").quantile(high_q).over(cell_keys)))

    n_expr = pl.col("_y").count().cast(pl.Int64).alias("n")
    if p.reduce == "leq":
        value_expr = (10.0 * (10.0 ** (pl.col("_y") / 10.0)).mean().log10()).alias("value")
    elif p.reduce == "max":
        value_expr = pl.col("_y").max().alias("value")
    else:
        value_expr = pl.col("_y").mean().alias("value")

    if p.reduce != "mean":
        err_expr = pl.lit(None, dtype=pl.Float64)
    else:
        std_expr = pl.col("_y").std(ddof=1)
        count_f64 = pl.col("_y").count().cast(pl.Float64)
        error_terms = {
            "std": std_expr,
            "sem": std_expr / count_f64.sqrt(),
            "ci95": _WILSON_Z * std_expr / count_f64.sqrt(),
        }
        err_expr = pl.when(pl.col("_y").count() >= 2).then(error_terms[p.error_mode]).otherwise(pl.lit(None, dtype=pl.Float64))

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
    if p.smooth_window is not None and agg.height >= 3:
        max_w = min(p.smooth_window, agg.height)

        def _smooth_window(col_name: str, width: int) -> pl.Expr:
            pad_left = width // 2
            terms = [pl.when(pl.int_range(pl.len()) + offset < 0).then(pl.col(col_name).first()).when(pl.int_range(pl.len()) + offset >= pl.len()).then(pl.col(col_name).last()).otherwise(pl.col(col_name).shift(-offset)) for offset in range(-pad_left, width - pad_left)]
            return sum(terms[1:], terms[0]) / float(width)

        exprs: list[pl.Expr] = []
        for col_name in ("value", "err_plus", "err_minus"):
            branch: pl.Expr = _smooth_window(col_name, max_w)
            for width in range(max_w - 1, 2, -1):
                branch = pl.when(pl.len() == width).then(_smooth_window(col_name, width)).otherwise(branch)
            expr = pl.when(pl.len() < 3).then(pl.col(col_name)).otherwise(branch)
            if p.group_cols:
                expr = expr.over(list(p.group_cols))
            exprs.append(expr.alias(col_name))
        agg = agg.with_columns(exprs)

    out = agg.with_columns(
        pl.lit(p.metric, dtype=pl.String).alias("metric"),
        pl.lit(unit, dtype=pl.String).alias("unit"),
    ).select(out_cols)
    return AnalysisResult(frame=out, summary=out, excluded_rows=excluded, seeds_used=seeds)


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
        diagnostics=result.diagnostics,
    )


_ComputationSpec = tuple[str, type[_BaseParams], type[Table], _ComputeFn, str]

_COMPUTATION_SPECS: tuple[_ComputationSpec, ...] = (
    (
        "group_summary",
        GroupSummaryParams,
        GroupSummaryAnalysis,
        _compute_group_summary,
        "Per-group summary statistics (Wilson CI for booleans; mean/std/sem/ci95 for numeric).",
    ),
    (
        "distribution_bins",
        DistributionBinsParams,
        DistributionBinsAnalysis,
        _compute_distribution_bins,
        "Fixed-edge histogram bins matching HistogramRenderer (nbins=15, +-1 flat widening); the density column holds a probability mass per bin (count / total) unless density=True, which divides by the bin width to give a probability density.",
    ),
    (
        "radar_profile",
        RadarProfileParams,
        RadarProfileAnalysis,
        _compute_radar_profile,
        "Capability tradeoff radar profile matching RadarRenderer normalization (null means stay null).",
    ),
    (
        "composite_ranking",
        CompositeRankingParams,
        CompositeRankingAnalysis,
        _run_composite_ranking,
        "Weighted min-max normalized composite ranking (unrounded scores).",
    ),
    (
        "top_n_ranking",
        CompositeRankingParams,
        CompositeRankingAnalysis,
        functools.partial(_run_composite_ranking, default_top_n=3),
        "Top-N contestants by weighted normalized composite score (unrounded scores).",
    ),
    (
        "correlation_matrix",
        CorrelationMatrixParams,
        CorrelationMatrixAnalysis,
        _compute_correlation_matrix,
        "Pairwise-complete Pearson correlation matrix using pl.corr.",
    ),
    (
        "paired_diff",
        PairedDiffParams,
        PairedDiffAnalysis,
        _compute_paired_diff,
        "Cluster-paired treatment minus control difference with bootstrap CI and p-value.",
    ),
    (
        "cluster_bootstrap_ci",
        ClusterBootstrapCIParams,
        ClusterBootstrapCIAnalysis,
        _compute_cluster_bootstrap_ci,
        "Cluster-bootstrap confidence interval per group using Layer 3 stats.",
    ),
    (
        "series_profile",
        SeriesProfileParams,
        SeriesProfileAnalysis,
        _compute_series_profile,
        "Series-grain binned/trimmed/reduced profile along an x axis with optional smoothing.",
    ),
)

for _name, _params, _output, _fn, _description in _COMPUTATION_SPECS:
    COMPUTATIONS[_name] = Computation(
        name=_name,
        params=_params,
        output=_output,
        description=_description,
        fn=_fn,
    )
