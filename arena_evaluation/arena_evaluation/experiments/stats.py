"""Shared statistics for the paired experiments (EVALUATION.md "Statistics").

- Seeds are fixed per scenario and shared by both arms, so a difference is taken per (scenario, seed).
- Resampling is a cluster bootstrap over scenarios: a drawn scenario brings all of its seeds.
- 95 % percentile intervals, printed unadjusted.
- A difference "excludes zero" when its two-sided bootstrap p < alpha after Holm within its family.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

DEFAULT_BOOT = 10_000
ALPHA = 0.05


@dataclasses.dataclass(frozen=True)
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
) -> BootResult:
    """Resample clusters (scenarios) with replacement, each carrying all of its rows (seeds).

    statistic maps the drawn clusters to a number, default the mean over every drawn row, so a
    scenario with a missing seed weighs less instead of being re-weighted to full.
    p is two-sided, 2 * min(share of draws <= 0, share >= 0), so p < 1 - level exactly when the
    percentile interval excludes zero.
    """
    clusters = [np.asarray(c, dtype=float) for c in clusters]
    clusters = [c[np.isfinite(c)] if c.ndim == 1 else c for c in clusters]
    clusters = [c for c in clusters if len(c) > 0]
    if statistic is None:

        def statistic(drawn: list[np.ndarray]) -> float:
            return float(np.mean(np.concatenate(drawn)))

    n_pairs = int(sum(len(c) for c in clusters))
    if not clusters:
        return BootResult(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0)
    rng = rng if rng is not None else np.random.default_rng(0)
    estimate = statistic(clusters)
    k = len(clusters)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, k, size=k)
        draws[i] = statistic([clusters[j] for j in idx])
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return BootResult(estimate, float("nan"), float("nan"), float("nan"), k, n_pairs)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    p = min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
    return BootResult(float(estimate), float(lo), float(hi), float(p), k, n_pairs)


def holm(pvalues: Sequence[float]) -> list[float]:
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
        df.filter(pl.col(arm).is_in([treatment, control]) & pl.col(metric).is_not_null())
        .select([*keys, arm, pl.col(metric).cast(pl.Float64)])
        .unique(subset=[*keys, arm], keep="first")
        .pivot(on=arm, index=keys, values=metric)
    )
    if treatment not in wide.columns or control not in wide.columns:
        return pl.DataFrame(schema={**{k: df.schema[k] for k in keys}, "diff": pl.Float64})
    return wide.with_columns((pl.col(treatment) - pl.col(control)).alias("diff")).drop_nulls("diff").select([*keys, "diff"])


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
    rows: list[dict[str, typing.Any]] = []
    rng = np.random.default_rng(seed)
    for key, group in diffs.group_by(list(cell), maintain_order=True):
        clusters = [g["diff"].to_numpy() for _, g in group.group_by(scenario, maintain_order=True)]
        res = cluster_bootstrap(clusters, n_boot=n_boot, rng=rng)
        rows.append({**dict(zip(cell, key, strict=True)), **dataclasses.asdict(res)})
    if not rows:
        return pl.DataFrame()
    out = pl.DataFrame(rows)
    out = out.with_columns(pl.Series("p_holm", holm(out["p"].to_list())))
    return out.with_columns((pl.col("p_holm") < alpha).fill_null(False).alias("excludes_zero"))
