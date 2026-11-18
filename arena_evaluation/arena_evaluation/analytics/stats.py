"""Statistical Procedures for Benchmark OLAP Analytics (Layer 3).

Provides vectorized cluster-bootstrap confidence intervals and their two-sided
shifted-null p-value, paired treatment difference tests, rank-swap matrices,
Kendall's coefficient of concordance W, and deterministic
discovery/verification seed cohort partitioning.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Sequence
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.analytics.catalog import _to_lazy
from arena_evaluation.storage.registry import NullReason

__all__ = [
    "CLUSTER_BOOTSTRAP_SCHEMA",
    "KENDALL_RANKINGS_SCHEMA",
    "KendallWResult",
    "PAIRED_DIFF_PAIRS_SCHEMA",
    "PAIRED_DIFF_SUMMARY_SCHEMA",
    "PairedDiffResult",
    "RANK_SWAP_MATRIX_SCHEMA",
    "cluster_bootstrap_ci",
    "cluster_bootstrap_means",
    "contain_estimate",
    "cluster_bootstrap_p_value",
    "draw_cluster_indices",
    "kendalls_w",
    "paired_treatment_diff",
    "rank_swap_matrix",
    "split_discovery_verification_seeds",
]

CLUSTER_BOOTSTRAP_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "metric": pl.String,
        "n_obs": pl.Int64,
        "n_clusters": pl.Int64,
        "mean": pl.Float64,
        "std_error": pl.Float64,
        "ci_level": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "basic_low": pl.Float64,
        "basic_high": pl.Float64,
        "dropped_rows": pl.Int64,
    }
)

PAIRED_DIFF_PAIRS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "value_a": pl.Float64,
        "value_b": pl.Float64,
        "diff": pl.Float64,
    }
)

PAIRED_DIFF_SUMMARY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "treatment_col": pl.String,
        "treatment_a": pl.String,
        "treatment_b": pl.String,
        "metric": pl.String,
        "n_pairs": pl.Int64,
        "n_clusters": pl.Int64,
        "mean_a": pl.Float64,
        "mean_b": pl.Float64,
        "mean_diff": pl.Float64,
        "std_diff": pl.Float64,
        "cohens_dz": pl.Float64,
        "cohens_dz_null_reason": pl.String,
        "ci_level": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "p_value": pl.Float64,
    }
)

RANK_SWAP_MATRIX_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "regime_a": pl.String,
        "regime_b": pl.String,
        "n_entities": pl.Int64,
        "total_pairs": pl.Int64,
        "swap_count": pl.Int64,
        "swap_rate": pl.Float64,
        "kendall_tau": pl.Float64,
    }
)

KENDALL_RANKINGS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "judge": pl.String,
        "entity": pl.String,
        "metric_mean": pl.Float64,
        "rank": pl.Float64,
    }
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PairedDiffResult:
    """Result of a paired treatment comparison across matched blocking factors."""

    treatment_col: str
    treatment_a: str
    treatment_b: str
    metric_col: str
    n_pairs: int
    n_clusters: int
    mean_a: float | None
    mean_b: float | None
    mean_diff: float | None
    std_diff: float | None
    cohens_dz: float | None
    ci_level: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    pairs_frame: pl.DataFrame
    summary_frame: pl.DataFrame


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class KendallWResult:
    """Result of Kendall's coefficient of concordance W across judges and entities."""

    entity_col: str
    judge_col: str
    metric_col: str
    n_judges: int
    n_entities: int
    w: float | None
    friedman_chi2: float
    df: int
    p_value: float | None
    rankings_frame: pl.DataFrame
    note: str = ""


def _validate_n_boot(n_boot: int) -> None:
    """Require at least one bootstrap resample."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")


def _validate_ci_level(ci_level: float) -> None:
    """Require a confidence level strictly inside (0, 1)."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")


def _validate_direction(direction: str) -> bool:
    """Validate ranking direction and return True when descending."""
    if direction not in ("asc", "desc"):
        raise ValueError(f"direction must be 'asc' or 'desc', got {direction!r}")
    return direction == "desc"


def _chi2_sf(chi2_val: float, df: int) -> float:
    """Survival function P(X >= chi2_val) for Chi-square distribution with df degrees of freedom."""
    if df <= 0 or chi2_val <= 0.0:
        return 1.0
    if df == 1:
        return float(math.erfc(math.sqrt(chi2_val / 2.0)))
    if df == 2:
        return float(math.exp(-0.5 * chi2_val))
    # Wilson-Hilferty normal approximation for df > 2
    k = float(df)
    z = ((chi2_val / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
        2.0 / (9.0 * k)
    )
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def draw_cluster_indices(
    n_clusters: int,
    n_boot: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_boot`` whole-cluster resample index rows of size ``n_clusters`` (cluster unit: ``seed`` by default)."""
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    _validate_n_boot(n_boot)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def cluster_bootstrap_means(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample whole clusters (default unit: ``seed``) and return weighted cluster means along the last axis."""
    n_clusters = cluster_sums.shape[-1]
    idx = draw_cluster_indices(n_clusters, n_boot, rng=rng)
    boot_sums = np.sum(cluster_sums[..., idx], axis=-1)
    boot_counts = np.sum(cluster_counts[..., idx], axis=-1)
    return np.where(
        boot_counts > 0.0,
        boot_sums / np.where(boot_counts > 0.0, boot_counts, 1.0),
        np.nan,
    )


def contain_estimate(low: float, high: float, estimate: float) -> tuple[float, float]:
    """Widen a percentile interval so it contains its own point estimate.

    A bootstrap whose replicates differ only by rounding can lie entirely to one side of its estimate.
    Both interval constructions in the layer (the OLAP analytics and the paired experiments) apply this
    one rule, so they cannot disagree about whether they contain the estimate.
    """
    return min(low, estimate), max(high, estimate)


def _bootstrap_partition_ci(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Compute point estimate, SE, percentile CI, and basic CI for one cluster partition.

    A bootstrap whose replicates differ only by rounding can lie entirely to one side of
    its own point estimate, so the percentile interval is widened to contain the estimate.
    """
    n_clusters = len(sums)
    n_obs = int(np.sum(counts))
    total_count = float(np.sum(counts))
    theta_hat = float(np.sum(sums) / total_count) if total_count > 0.0 else float("nan")
    if n_clusters == 1:
        return (
            n_obs,
            n_clusters,
            theta_hat,
            0.0,
            theta_hat,
            theta_hat,
            theta_hat,
            theta_hat,
        )

    alpha = 1.0 - ci_level
    boot_means = cluster_bootstrap_means(sums, counts, n_boot=n_boot, rng=rng)
    observed_means = boot_means[np.isfinite(boot_means)]
    if observed_means.size == 0:
        nan = float("nan")
        return n_obs, n_clusters, theta_hat, nan, nan, nan, nan, nan
    std_err = float(np.std(observed_means, ddof=1)) if observed_means.size > 1 else 0.0
    ci_low, ci_high = contain_estimate(
        float(np.quantile(observed_means, alpha / 2.0)),
        float(np.quantile(observed_means, 1.0 - alpha / 2.0)),
        theta_hat,
    )
    basic_low = float(2.0 * theta_hat - ci_high)
    basic_high = float(2.0 * theta_hat - ci_low)
    return n_obs, n_clusters, theta_hat, std_err, ci_low, ci_high, basic_low, basic_high


def cluster_bootstrap_ci(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    group_cols: Sequence[str] = ("planner",),
    cluster_col: str = "seed",
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Compute cluster-bootstrap confidence intervals (cluster unit: ``cluster_col``, default ``seed``)."""
    _validate_n_boot(n_boot)
    _validate_ci_level(ci_level)

    group_list = list(group_cols)
    cluster_agg_cols = [*group_list, cluster_col] if group_list else [cluster_col]
    source = _to_lazy(frame)
    total_rows = source.select(pl.len()).collect().item()
    cluster_df = (
        source.filter(
            pl.col(metric_col).is_not_null() & pl.col(cluster_col).is_not_null()
        )
        .group_by(cluster_agg_cols)
        .agg(
            cluster_sum=pl.col(metric_col).cast(pl.Float64).sum(),
            cluster_count=pl.col(metric_col).count().cast(pl.Int64),
        )
        .sort(cluster_agg_cols)
        .collect()
    )

    out_schema = pl.Schema(
        {
            **{col: cluster_df.schema[col] for col in group_list},
            **CLUSTER_BOOTSTRAP_SCHEMA,
        }
    )
    if cluster_df.is_empty():
        return pl.DataFrame(schema=out_schema)
    dropped_rows = int(total_rows) - int(
        cluster_df.select(pl.col("cluster_count").sum()).item()
    )

    partitions = (
        cluster_df.partition_by(group_list, maintain_order=True)
        if group_list
        else [cluster_df]
    )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for part_df in partitions:
        sums = part_df.get_column("cluster_sum").to_numpy().astype(np.float64)
        counts = part_df.get_column("cluster_count").to_numpy().astype(np.float64)
        (
            n_obs,
            n_clusters,
            theta_hat,
            std_err,
            ci_low,
            ci_high,
            basic_low,
            basic_high,
        ) = _bootstrap_partition_ci(
            sums, counts, n_boot=n_boot, ci_level=ci_level, rng=rng
        )
        row: dict[str, object] = {col: part_df.item(0, col) for col in group_list}
        row.update(
            {
                "metric": metric_col,
                "n_obs": n_obs,
                "n_clusters": n_clusters,
                "mean": theta_hat,
                "std_error": std_err,
                "ci_level": float(ci_level),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "basic_low": basic_low,
                "basic_high": basic_high,
                "dropped_rows": dropped_rows,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=out_schema)


def cluster_bootstrap_p_value(
    cluster_sums: Sequence[float],
    cluster_counts: Sequence[float],
    *,
    observed: float,
    n_boot: int = 2000,
    random_seed: int = 42,
) -> float:
    """Two-sided p-value of H0: the pooled cluster mean is zero, from a shifted-null cluster bootstrap.

    ``cluster_sums`` and ``cluster_counts`` are one pooled difference sum and one pair count per
    cluster (scenario, seed, ...), the sufficient statistics of the pooled mean. A resample draws
    whole clusters with replacement, so it keeps every pair inside a drawn cluster. The null
    distribution is the centred bootstrap: each cluster sum is shifted by ``observed`` times its
    count, which moves the resampled mean onto H0 while preserving the between-cluster spread.
    The p-value is the share of null draws at least as far from zero as the observed estimate,
    with the ``(1 + extremes) / (n_boot + 1)`` correction so a finite number of draws cannot
    report p = 0. This is the single implementation both ``paired_treatment_diff`` and
    ``experiments.stats.cluster_bootstrap`` report.
    """
    _validate_n_boot(n_boot)

    sums = np.asarray(cluster_sums, dtype=np.float64)
    counts = np.asarray(cluster_counts, dtype=np.float64)
    if int(sums.size) < 2:
        # A single cluster resamples to itself and cannot reject H0.
        return 1.0

    rng = np.random.default_rng(random_seed)
    centered_sums = sums - observed * counts
    null_means = cluster_bootstrap_means(centered_sums, counts, n_boot=n_boot, rng=rng)
    extreme_count = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
    return float((1 + extreme_count) / (n_boot + 1))


def _cohens_dz(mean_diff: float, std_diff: float) -> float | None:
    """Return paired Cohen's dz, or NULL when the paired differences do not vary.

    A constant difference has no standardised effect size: dividing by a zero standard deviation
    would publish an infinite (or, at zero difference, a meaningless zero) effect.
    """
    if std_diff > 1e-12:
        return float(mean_diff / std_diff)
    return None


def _cluster_diff_totals(
    pairs_df: pl.DataFrame,
    pair_list: list[str],
    cluster_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate paired differences into per-cluster sums and pair counts."""
    if cluster_col not in pair_list:
        diffs = pairs_df.get_column("diff").to_numpy().astype(np.float64)
        return diffs, np.ones(len(diffs), dtype=np.float64)
    cluster_df = (
        pairs_df.group_by(cluster_col)
        .agg(
            c_sum=pl.col("diff").sum(),
            c_cnt=pl.col("diff").count().cast(pl.Float64),
        )
        .sort(cluster_col)
    )
    return (
        cluster_df.get_column("c_sum").to_numpy().astype(np.float64),
        cluster_df.get_column("c_cnt").to_numpy().astype(np.float64),
    )


def _build_paired_diff_result(
    *,
    treatment_col: str,
    treatment_a: str,
    treatment_b: str,
    metric_col: str,
    n_pairs: int,
    n_clusters: int,
    mean_a: float | None,
    mean_b: float | None,
    mean_diff: float | None,
    std_diff: float | None,
    cohens_dz: float | None,
    ci_level: float,
    ci_low: float | None,
    ci_high: float | None,
    p_value: float | None,
    pairs_df: pl.DataFrame,
) -> PairedDiffResult:
    """Construct a PairedDiffResult dataclass instance."""
    summary_df = pl.DataFrame(
        [
            {
                "treatment_col": treatment_col,
                "treatment_a": treatment_a,
                "treatment_b": treatment_b,
                "metric": metric_col,
                "n_pairs": n_pairs,
                "n_clusters": n_clusters,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "cohens_dz": cohens_dz,
                "cohens_dz_null_reason": (
                    None
                    if cohens_dz is not None or n_pairs == 0
                    else f"std_diff:{NullReason.ZERO_DENOMINATOR.value}"
                ),
                "ci_level": float(ci_level),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        ],
        schema=PAIRED_DIFF_SUMMARY_SCHEMA,
    )
    return PairedDiffResult(
        treatment_col=treatment_col,
        treatment_a=treatment_a,
        treatment_b=treatment_b,
        metric_col=metric_col,
        n_pairs=n_pairs,
        n_clusters=n_clusters,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_diff=std_diff,
        cohens_dz=cohens_dz,
        ci_level=float(ci_level),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        pairs_frame=pairs_df,
        summary_frame=summary_df,
    )


def paired_treatment_diff(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    treatment_col: str = "planner",
    treatment_a: str,
    treatment_b: str,
    pair_cols: Sequence[str] = ("world", "scenario", "crowd_size", "seed"),
    cluster_col: str = "seed",
    n_boot: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> PairedDiffResult:
    """Compute paired differences (A - B), Cohen's dz, cluster-bootstrap CI (cluster unit: ``cluster_col``, default ``seed``), and p-value."""
    _validate_n_boot(n_boot)
    _validate_ci_level(ci_level)
    pair_list = list(pair_cols)
    if not pair_list:
        raise ValueError("pair_cols must contain at least one blocking column.")

    cell_means = (
        _to_lazy(frame)
        .filter(
            pl.col(treatment_col).is_in([treatment_a, treatment_b])
            & pl.col(metric_col).is_not_null()
        )
        .group_by([*pair_list, treatment_col])
        .agg(cell_value=pl.col(metric_col).cast(pl.Float64).mean())
    )
    side_a = cell_means.filter(pl.col(treatment_col) == treatment_a).select(
        [*pair_list, pl.col("cell_value").alias("value_a")]
    )
    side_b = cell_means.filter(pl.col(treatment_col) == treatment_b).select(
        [*pair_list, pl.col("cell_value").alias("value_b")]
    )
    pairs_df = (
        side_a.join(side_b, on=pair_list, how="inner")
        .with_columns(diff=pl.col("value_a") - pl.col("value_b"))
        .sort(pair_list)
        .collect()
    )

    if pairs_df.is_empty():
        return _build_paired_diff_result(
            treatment_col=treatment_col,
            treatment_a=treatment_a,
            treatment_b=treatment_b,
            metric_col=metric_col,
            n_pairs=0,
            n_clusters=0,
            mean_a=None,
            mean_b=None,
            mean_diff=None,
            std_diff=None,
            cohens_dz=None,
            ci_level=ci_level,
            ci_low=None,
            ci_high=None,
            p_value=None,
            pairs_df=pairs_df,
        )

    vals_a = pairs_df.get_column("value_a").to_numpy().astype(np.float64)
    vals_b = pairs_df.get_column("value_b").to_numpy().astype(np.float64)
    diffs = pairs_df.get_column("diff").to_numpy().astype(np.float64)
    n_pairs = len(diffs)
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    c_sums, c_cnts = _cluster_diff_totals(pairs_df, pair_list, cluster_col)
    _, n_clusters, mean_diff, _, ci_low, ci_high, _, _ = _bootstrap_partition_ci(
        c_sums,
        c_cnts,
        n_boot=n_boot,
        ci_level=ci_level,
        rng=np.random.default_rng(random_seed),
    )
    std_diff = float(np.std(diffs, ddof=1)) if n_pairs > 1 else 0.0
    cohens_dz = _cohens_dz(mean_diff, std_diff)

    p_value = cluster_bootstrap_p_value(
        c_sums, c_cnts, observed=mean_diff, n_boot=n_boot, random_seed=random_seed
    )
    return _build_paired_diff_result(
        treatment_col=treatment_col,
        treatment_a=treatment_a,
        treatment_b=treatment_b,
        metric_col=metric_col,
        n_pairs=n_pairs,
        n_clusters=n_clusters,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_diff=std_diff,
        cohens_dz=cohens_dz,
        ci_level=ci_level,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        pairs_df=pairs_df,
    )


def _pairwise_rank_swap(
    reg_a: str,
    reg_b: str,
    frame_a: pl.DataFrame,
    frame_b: pl.DataFrame,
) -> dict[str, object]:
    """Compute pairwise rank-swap statistics for one ``(reg_a, reg_b)`` regime pair."""
    joined = frame_a.join(frame_b, on="entity", how="inner", suffix="_b").sort("entity")
    k = joined.height
    total_pairs = k * (k - 1) // 2
    if total_pairs == 0:
        return {
            "regime_a": reg_a,
            "regime_b": reg_b,
            "n_entities": k,
            "total_pairs": 0,
            "swap_count": 0,
            "swap_rate": 0.0,
            "kendall_tau": 1.0,
        }

    r_a = joined.get_column("rank").to_numpy().astype(np.float64)
    r_b = joined.get_column("rank_b").to_numpy().astype(np.float64)
    idx_i, idx_j = np.triu_indices(k, k=1)
    prod = (r_a[idx_i] - r_a[idx_j]) * (r_b[idx_i] - r_b[idx_j])
    discordant = int(np.sum(prod < -1e-12))
    concordant = int(np.sum(prod > 1e-12))
    return {
        "regime_a": reg_a,
        "regime_b": reg_b,
        "n_entities": k,
        "total_pairs": total_pairs,
        "swap_count": discordant,
        "swap_rate": float(discordant / total_pairs),
        "kendall_tau": float((concordant - discordant) / total_pairs),
    }


def rank_swap_matrix(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    entity_col: str = "planner",
    regime_col: str = "crowd_condition",
    direction: str = "asc",
) -> pl.DataFrame:
    """Compute pairwise rank-swap counts and Kendall tau distances across regimes."""
    descending = _validate_direction(direction)
    ranked_df = (
        _to_lazy(frame)
        .filter(
            pl.col(metric_col).is_not_null()
            & pl.col(entity_col).is_not_null()
            & pl.col(regime_col).is_not_null()
        )
        .group_by(regime_col, entity_col)
        .agg(metric_mean=pl.col(metric_col).cast(pl.Float64).mean())
        .with_columns(
            rank=pl.col("metric_mean")
            .rank(method="average", descending=descending)
            .over(regime_col)
        )
        .sort(regime_col, entity_col)
        .collect()
    )
    if ranked_df.is_empty():
        return pl.DataFrame(schema=RANK_SWAP_MATRIX_SCHEMA)

    regimes = [
        str(r)
        for r in ranked_df.get_column(regime_col).unique(maintain_order=True).to_numpy()
    ]
    by_regime: dict[str, pl.DataFrame] = {
        str(part.item(0, regime_col)): part.select(
            pl.col(entity_col).cast(pl.String).alias("entity"),
            pl.col("rank").cast(pl.Float64),
        )
        for part in ranked_df.partition_by(regime_col, maintain_order=True)
    }

    rows = [
        _pairwise_rank_swap(reg_a, reg_b, by_regime[reg_a], by_regime[reg_b])
        for i, reg_a in enumerate(regimes)
        for reg_b in regimes[i + 1 :]
    ]
    if not rows:
        return pl.DataFrame(schema=RANK_SWAP_MATRIX_SCHEMA)
    return pl.DataFrame(rows, schema=RANK_SWAP_MATRIX_SCHEMA)


def _compute_kendalls_w_stats(
    complete_df: pl.DataFrame,
    *,
    n_judges: int,
    n_entities: int,
) -> tuple[float, float, int, float]:
    """Compute Kendall's W, Friedman's chi2, and chi2 survival p-value."""
    rank_sums = (
        complete_df.group_by("entity")
        .agg(r_sum=pl.col("rank").sum())
        .sort("entity")
        .get_column("r_sum")
        .to_numpy()
        .astype(np.float64)
    )
    mean_r = 0.5 * n_judges * (n_entities + 1)
    s_dev = float(np.sum((rank_sums - mean_r) ** 2))

    # Tie correction T_j = sum(t^3 - t) per judge
    tie_groups = (
        complete_df.group_by("judge", "rank")
        .agg(t_cnt=pl.len().cast(pl.Int64))
        .filter(pl.col("t_cnt") > 1)
    )
    tie_sum = 0.0
    if not tie_groups.is_empty():
        t_vals = tie_groups.get_column("t_cnt").to_numpy().astype(np.float64)
        tie_sum = float(np.sum(t_vals**3 - t_vals))

    denom = float(n_judges**2 * (n_entities**3 - n_entities) - n_judges * tie_sum)
    w = float(np.clip((12.0 * s_dev) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    df = n_entities - 1
    friedman_chi2 = float(n_judges * df * w)
    return w, friedman_chi2, df, _chi2_sf(friedman_chi2, df)


def kendalls_w(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    entity_col: str = "planner",
    judge_col: str = "crowd_condition",
    direction: str = "asc",
) -> KendallWResult:
    """Compute Kendall's coefficient of concordance W, Friedman's chi2, and p-value across judges."""
    descending = _validate_direction(direction)
    cell_df = (
        _to_lazy(frame)
        .filter(
            pl.col(metric_col).is_not_null()
            & pl.col(entity_col).is_not_null()
            & pl.col(judge_col).is_not_null()
        )
        .group_by(
            pl.col(judge_col).cast(pl.String).alias("judge"),
            pl.col(entity_col).cast(pl.String).alias("entity"),
        )
        .agg(metric_mean=pl.col(metric_col).cast(pl.Float64).mean())
        .collect()
    )
    if cell_df.is_empty():
        return KendallWResult(
            entity_col=entity_col,
            judge_col=judge_col,
            metric_col=metric_col,
            n_judges=0,
            n_entities=0,
            w=None,
            friedman_chi2=0.0,
            df=0,
            p_value=None,
            rankings_frame=pl.DataFrame(schema=KENDALL_RANKINGS_SCHEMA),
            note="no complete entity is judged, so concordance is undefined",
        )

    total_judges = cell_df.get_column("judge").n_unique()
    complete_entities = (
        cell_df.group_by("entity")
        .agg(j_cnt=pl.col("judge").n_unique())
        .filter(pl.col("j_cnt") == total_judges)
        .select("entity")
    )
    complete_df = (
        cell_df.join(complete_entities, on="entity", how="inner")
        .with_columns(
            rank=pl.col("metric_mean")
            .rank(method="average", descending=descending)
            .over("judge")
        )
        .sort("judge", "entity")
        .select(list(KENDALL_RANKINGS_SCHEMA.names()))
        .cast(KENDALL_RANKINGS_SCHEMA)
    )

    k = complete_df.get_column("entity").n_unique()
    m = complete_df.get_column("judge").n_unique()
    # Concordance needs at least two judges to agree with each other and two entities to be ordered.
    if m < 2 or k < 2:
        note = (
            f"{m} judge(s) and {k} complete entity(ies) cannot be tested for concordance; "
            "at least two of each are required"
        )
        return KendallWResult(
            entity_col=entity_col,
            judge_col=judge_col,
            metric_col=metric_col,
            n_judges=m,
            n_entities=k,
            w=None,
            friedman_chi2=0.0,
            df=max(0, k - 1),
            p_value=None,
            rankings_frame=complete_df,
            note=note,
        )

    w, friedman_chi2, df, p_value = _compute_kendalls_w_stats(
        complete_df, n_judges=m, n_entities=k
    )
    return KendallWResult(
        entity_col=entity_col,
        judge_col=judge_col,
        metric_col=metric_col,
        n_judges=m,
        n_entities=k,
        w=w,
        friedman_chi2=friedman_chi2,
        df=df,
        p_value=p_value,
        rankings_frame=complete_df,
    )


@typing.overload
def split_discovery_verification_seeds(
    frame: pl.LazyFrame,
    *,
    seed_col: str = "seed",
    discovery_fraction: float = 0.5,
    cohort_col: str = "cohort",
) -> pl.LazyFrame: ...


@typing.overload
def split_discovery_verification_seeds(
    frame: pl.DataFrame,
    *,
    seed_col: str = "seed",
    discovery_fraction: float = 0.5,
    cohort_col: str = "cohort",
) -> pl.DataFrame: ...


def split_discovery_verification_seeds(
    frame: pl.LazyFrame | pl.DataFrame,
    *,
    seed_col: str = "seed",
    discovery_fraction: float = 0.5,
    cohort_col: str = "cohort",
) -> pl.LazyFrame | pl.DataFrame:
    """Partition seeds into ``'discovery'`` and ``'verification'`` cohorts by dense seed rank.

    The split is deterministic: seeds are ordered by rank, never drawn at random, so the same frame
    always yields the same cohorts. Fewer than two distinct seeds cannot form both cohorts and raise
    instead of returning an empty verification side.
    """
    if not (0.0 < discovery_fraction < 1.0):
        raise ValueError(
            f"discovery_fraction must be in (0, 1), got {discovery_fraction}"
        )
    distinct_seeds = (
        frame.select(pl.col(seed_col).n_unique()).collect().item()
        if isinstance(frame, pl.LazyFrame)
        else frame.get_column(seed_col).n_unique()
    )
    if int(distinct_seeds) < 2:
        raise ValueError(
            f"cannot split {distinct_seeds} distinct seed(s) into discovery and verification cohorts; "
            "at least two are required"
        )

    is_eager = isinstance(frame, pl.DataFrame)
    seed_rank_0 = pl.col(seed_col).rank(method="dense").cast(pl.Int64) - 1
    n_unique_seeds = pl.col(seed_col).n_unique().cast(pl.Int64)
    cutoff = (
        (n_unique_seeds.cast(pl.Float64) * discovery_fraction)
        .round()
        .cast(pl.Int64)
        .clip(1, pl.max_horizontal(pl.lit(1, dtype=pl.Int64), n_unique_seeds - 1))
    )
    cohort_expr = (
        pl.when(pl.col(seed_col).is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(seed_rank_0 < cutoff)
        .then(pl.lit("discovery"))
        .otherwise(pl.lit("verification"))
        .alias(cohort_col)
    )
    result_lf = _to_lazy(frame).with_columns(cohort_expr)
    return result_lf.collect() if is_eager else result_lf
