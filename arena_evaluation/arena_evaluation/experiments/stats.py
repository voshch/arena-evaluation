"""Shared statistics for the paired experiments (EVALUATION.md "Statistics")."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from arena_evaluation.analytics.stats import (
    cluster_bootstrap_means,
    cluster_bootstrap_p_value,
    contain_estimate,
    draw_cluster_indices,
)

DEFAULT_BOOT = 10_000
DEFAULT_P_SEED = 42
ALPHA = 0.05


@dataclasses.dataclass(frozen=True, slots=True)
class BootResult:
    estimate: float
    ci_low: float
    ci_high: float
    p: float
    n_scenarios: int
    n_pairs: int


def cluster_bootstrap(
    clusters: Sequence[np.ndarray],
    statistic: Callable[[list[np.ndarray]], float] | None = None,
    *,
    n_boot: int = DEFAULT_BOOT,
    rng: np.random.Generator | None = None,
    level: float = 0.95,
    p_seed: int = DEFAULT_P_SEED,
) -> BootResult:
    """Percentile bootstrap over whole clusters, with the shifted-null two-sided p-value.

    ``p_seed`` seeds the shifted-null test on its own so the interval and the test are each
    reproducible; the p-value itself is :func:`arena_evaluation.analytics.stats.cluster_bootstrap_p_value`
    and tests the pooled mean of the drawn clusters.
    """
    clusters = [np.asarray(c, dtype=float) for c in clusters]
    clusters = [c[np.isfinite(c)] if c.ndim == 1 else c for c in clusters]
    clusters = [c for c in clusters if len(c) > 0]
    n_pairs = int(sum(len(c) for c in clusters))
    if not clusters:
        return BootResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)
    rng = rng if rng is not None else np.random.default_rng(0)
    k = len(clusters)
    cluster_sums = np.array([float(np.sum(c)) for c in clusters], dtype=np.float64)
    cluster_counts = np.array([float(len(c)) for c in clusters], dtype=np.float64)

    if statistic is None:
        estimate = float(np.sum(cluster_sums) / np.sum(cluster_counts))
        draws = cluster_bootstrap_means(
            cluster_sums, cluster_counts, n_boot=n_boot, rng=rng
        )
    else:
        estimate = statistic(clusters)
        idx_matrix = draw_cluster_indices(k, n_boot, rng=rng)
        draws = np.array(
            [statistic([clusters[j] for j in row]) for row in idx_matrix],
            dtype=np.float64,
        )
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return BootResult(
            estimate, float("nan"), float("nan"), float("nan"), k, n_pairs
        )
    tail = (1.0 - level) / 2.0
    lo, hi = contain_estimate(
        float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail)), estimate
    )
    p = cluster_bootstrap_p_value(
        cluster_sums,
        cluster_counts,
        observed=estimate,
        n_boot=n_boot,
        random_seed=p_seed,
    )
    return BootResult(float(estimate), float(lo), float(hi), float(p), k, n_pairs)


def holm(pvalues: Sequence[float] | np.ndarray) -> list[float]:
    """Holm step-down adjusted p-values, NaN passes through and does not count toward the family size."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    m = len(valid)
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid], kind="stable")]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def paired_differences(
    df: pl.DataFrame,
    *,
    cell: Sequence[str],
    metric: str,
    arm: str,
    treatment: str,
    control: str,
    scenario: str = "scenario",
    seed: str = "seed",
) -> pl.DataFrame:
    """treatment - control per (cell, scenario, seed); rows missing either arm or the metric drop out."""
    keys = [*cell, scenario, seed]
    wide = (
        df.filter(
            pl.col(arm).is_in([treatment, control]) & pl.col(metric).is_not_null()
        )
        .select([*keys, arm, pl.col(metric).cast(pl.Float64)])
        .unique(subset=[*keys, arm], keep="first")
        .pivot(on=arm, index=keys, values=metric)
    )
    if treatment not in wide.columns or control not in wide.columns:
        return pl.DataFrame(
            schema={**{k: df.schema[k] for k in keys}, "diff": pl.Float64}
        )
    return (
        wide.with_columns((pl.col(treatment) - pl.col(control)).alias("diff"))
        .drop_nulls("diff")
        .select([*keys, "diff"])
    )


def summarize_family(
    diffs: pl.DataFrame,
    *,
    cell: Sequence[str],
    scenario: str = "scenario",
    n_boot: int = DEFAULT_BOOT,
    seed: int = 0,
    alpha: float = ALPHA,
) -> pl.DataFrame:
    """One row per cell: estimate, unadjusted 95 % CI, bootstrap p, Holm p over the family, excludes_zero."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for key, group in diffs.group_by(list(cell), maintain_order=True):
        clusters = [
            g["diff"].to_numpy()
            for _, g in group.group_by(scenario, maintain_order=True)
        ]
        res = cluster_bootstrap(clusters, n_boot=n_boot, rng=rng)
        rows.append({**dict(zip(cell, key, strict=True)), **dataclasses.asdict(res)})
    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows)
    out = out.with_columns(pl.Series("p_holm", holm(out["p"].to_numpy())))
    return out.with_columns(
        (pl.col("p_holm") < alpha).fill_null(False).alias("excludes_zero")
    )
