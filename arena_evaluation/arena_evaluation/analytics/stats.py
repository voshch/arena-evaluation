"""Statistical Procedures for Benchmark OLAP Analytics (Layer 3)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np
import polars as pl

from arena_evaluation.analytics.catalog import _to_lazy
from arena_evaluation.storage.registry import NullReason

__all__ = [
    "CLUSTER_BOOTSTRAP_SCHEMA",
    "KENDALL_RANKINGS_SCHEMA",
    "KendallWResult",
    "PAIRED_DIFF_SUMMARY_SCHEMA",
    "PairedDiffResult",
    "RANK_SWAP_MATRIX_SCHEMA",
    "aggregate_timeseries_band",
    "cluster_bootstrap_ci",
    "cluster_bootstrap_means",
    "contain_estimate",
    "cluster_bootstrap_p_value",
    "draw_cluster_indices",
    "kendalls_w",
    "paired_treatment_diff",
    "rank_swap_matrix",
    "split_discovery_verification_seeds",
    "split_display_outliers",
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


_VALIDATION: Final[dict[str, tuple[Callable[[float], bool], str]]] = {
    "n_boot": (lambda value: value >= 1, "n_boot must be >= 1, got {value}"),
    "ci_level": (
        lambda value: 0.0 < value < 1.0,
        "ci_level must be in (0, 1), got {value}",
    ),
    "discovery_fraction": (
        lambda value: 0.0 < value < 1.0,
        "discovery_fraction must be in (0, 1), got {value}",
    ),
}


def _require(name: str, value: float) -> None:
    """Raise the ValueError registered for ``name`` unless ``value`` satisfies it."""
    predicate, message = _VALIDATION[name]
    if not predicate(value):
        raise ValueError(message.format(value=value))


def _descending(direction: str) -> bool:
    """Validate ranking direction and return True when descending."""
    if direction not in ("asc", "desc"):
        raise ValueError(f"direction must be 'asc' or 'desc', got {direction!r}")
    return direction == "desc"


def draw_cluster_indices(
    n_clusters: int,
    n_boot: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_boot`` whole-cluster resample index rows of size ``n_clusters``."""
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    _require("n_boot", n_boot)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def cluster_bootstrap_means(
    cluster_sums: np.ndarray,
    cluster_counts: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample whole clusters and return weighted cluster means along the last axis."""
    idx = draw_cluster_indices(cluster_sums.shape[-1], n_boot, rng=rng)
    boot_sums = np.sum(cluster_sums[..., idx], axis=-1)
    boot_counts = np.sum(cluster_counts[..., idx], axis=-1)
    safe_counts = np.where(boot_counts > 0.0, boot_counts, 1.0)
    return np.where(boot_counts > 0.0, boot_sums / safe_counts, np.nan)


def contain_estimate(low: float, high: float, estimate: float) -> tuple[float, float]:
    """Widen a percentile interval so it contains its own point estimate.

    A bootstrap whose replicates differ only by rounding can lie entirely to one side of its estimate.
    Both interval constructions in the layer (the OLAP analytics and the paired experiments) apply this
    one rule, so they cannot disagree about whether they contain the estimate.
    """
    return min(low, estimate), max(high, estimate)


@dataclasses.dataclass(frozen=True, slots=True)
class _ClusterBootstrap:
    """One pooled cluster sufficient-statistic pair resampled under a shared RNG."""

    sums: np.ndarray
    counts: np.ndarray
    n_boot: int
    rng: np.random.Generator

    def interval(
        self, ci_level: float
    ) -> tuple[int, int, float, float, float, float, float, float]:
        """Point estimate, SE, percentile CI, and basic CI for this cluster partition."""
        n_clusters = len(self.sums)
        n_obs = int(np.sum(self.counts))
        total_count = float(np.sum(self.counts))
        estimate = (
            float(np.sum(self.sums) / total_count)
            if total_count > 0.0
            else float("nan")
        )
        if n_clusters == 1:
            return (
                n_obs,
                n_clusters,
                estimate,
                0.0,
                estimate,
                estimate,
                estimate,
                estimate,
            )

        alpha = 1.0 - ci_level
        boot_means = cluster_bootstrap_means(
            self.sums, self.counts, n_boot=self.n_boot, rng=self.rng
        )
        finite_means = boot_means[np.isfinite(boot_means)]
        if finite_means.size == 0:
            nan = float("nan")
            return n_obs, n_clusters, estimate, nan, nan, nan, nan, nan
        std_err = float(np.std(finite_means, ddof=1)) if finite_means.size > 1 else 0.0
        ci_low, ci_high = contain_estimate(
            float(np.quantile(finite_means, alpha / 2.0)),
            float(np.quantile(finite_means, 1.0 - alpha / 2.0)),
            estimate,
        )
        return (
            n_obs,
            n_clusters,
            estimate,
            std_err,
            ci_low,
            ci_high,
            float(2.0 * estimate - ci_high),
            float(2.0 * estimate - ci_low),
        )

    def p_value(self, observed: float) -> float:
        """Two-sided shifted-null p-value; a single cluster cannot reject H0."""
        if int(self.sums.size) < 2:
            return 1.0
        null_means = cluster_bootstrap_means(
            self.sums - observed * self.counts,
            self.counts,
            n_boot=self.n_boot,
            rng=self.rng,
        )
        extreme_count = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
        return float((1 + extreme_count) / (self.n_boot + 1))


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
    """Compute cluster-bootstrap confidence intervals (cluster unit: ``cluster_col``)."""
    _require("n_boot", n_boot)
    _require("ci_level", ci_level)

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
        stats = _ClusterBootstrap(
            sums=part_df.get_column("cluster_sum").to_numpy().astype(np.float64),
            counts=part_df.get_column("cluster_count").to_numpy().astype(np.float64),
            n_boot=n_boot,
            rng=rng,
        ).interval(ci_level)
        n_obs, n_clusters, mean, std_err, ci_low, ci_high, basic_low, basic_high = stats
        row: dict[str, object] = {col: part_df.item(0, col) for col in group_list}
        row.update(
            {
                "metric": metric_col,
                "n_obs": n_obs,
                "n_clusters": n_clusters,
                "mean": mean,
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
    _require("n_boot", n_boot)
    return _ClusterBootstrap(
        sums=np.asarray(cluster_sums, dtype=np.float64),
        counts=np.asarray(cluster_counts, dtype=np.float64),
        n_boot=n_boot,
        rng=np.random.default_rng(random_seed),
    ).p_value(observed)


def _build_paired_diff_result(
    fields: dict[str, object], pairs_df: pl.DataFrame
) -> PairedDiffResult:
    """Build the scalar summary row and typed result from one field mapping."""
    row = {**fields}
    row["metric"] = row.pop("metric_col")
    row["cohens_dz_null_reason"] = (
        None
        if fields["cohens_dz"] is not None or fields["n_pairs"] == 0
        else f"std_diff:{NullReason.ZERO_DENOMINATOR.value}"
    )
    summary = pl.DataFrame([row], schema=PAIRED_DIFF_SUMMARY_SCHEMA)
    return PairedDiffResult(**fields, pairs_frame=pairs_df, summary_frame=summary)


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
    """Compute paired differences (A - B), Cohen's dz, cluster-bootstrap CI, and p-value."""
    _require("n_boot", n_boot)
    _require("ci_level", ci_level)
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

    fields: dict[str, object] = dict.fromkeys(
        (
            "mean_a",
            "mean_b",
            "mean_diff",
            "std_diff",
            "cohens_dz",
            "ci_low",
            "ci_high",
            "p_value",
        )
    )
    fields.update(
        treatment_col=treatment_col,
        treatment_a=treatment_a,
        treatment_b=treatment_b,
        metric_col=metric_col,
        n_pairs=0,
        n_clusters=0,
        ci_level=float(ci_level),
    )
    if pairs_df.is_empty():
        return _build_paired_diff_result(fields, pairs_df)

    vals_a = pairs_df.get_column("value_a").to_numpy().astype(np.float64)
    vals_b = pairs_df.get_column("value_b").to_numpy().astype(np.float64)
    diffs = pairs_df.get_column("diff").to_numpy().astype(np.float64)
    n_pairs = len(diffs)
    mean_a = float(np.mean(vals_a))
    mean_b = float(np.mean(vals_b))
    if cluster_col in pair_list:
        cluster_df = (
            pairs_df.group_by(cluster_col)
            .agg(
                c_sum=pl.col("diff").sum(),
                c_cnt=pl.col("diff").count().cast(pl.Float64),
            )
            .sort(cluster_col)
        )
        c_sums = cluster_df.get_column("c_sum").to_numpy().astype(np.float64)
        c_cnts = cluster_df.get_column("c_cnt").to_numpy().astype(np.float64)
    else:
        c_sums = diffs
        c_cnts = np.ones(len(diffs), dtype=np.float64)

    _, n_clusters, mean_diff, _, ci_low, ci_high, _, _ = _ClusterBootstrap(
        sums=c_sums,
        counts=c_cnts,
        n_boot=n_boot,
        rng=np.random.default_rng(random_seed),
    ).interval(ci_level)
    std_diff = float(np.std(diffs, ddof=1)) if n_pairs > 1 else 0.0
    cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else None
    fields.update(
        n_pairs=n_pairs,
        n_clusters=n_clusters,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_diff=std_diff,
        cohens_dz=cohens_dz,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=cluster_bootstrap_p_value(
            c_sums, c_cnts, observed=mean_diff, n_boot=n_boot, random_seed=random_seed
        ),
    )
    return _build_paired_diff_result(fields, pairs_df)


def rank_swap_matrix(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    entity_col: str = "planner",
    regime_col: str = "crowd_condition",
    direction: str = "asc",
) -> pl.DataFrame:
    """Compute pairwise rank-swap counts and Kendall tau distances across regimes."""
    descending = _descending(direction)
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

    by_regime: dict[str, pl.DataFrame] = {
        str(part.item(0, regime_col)): part.select(
            pl.col(entity_col).cast(pl.String).alias("entity"),
            pl.col("rank").cast(pl.Float64),
        )
        for part in ranked_df.partition_by(regime_col, maintain_order=True)
    }
    regimes = list(by_regime)
    rows: list[dict[str, object]] = []
    for i, reg_a in enumerate(regimes):
        for reg_b in regimes[i + 1 :]:
            joined = (
                by_regime[reg_a]
                .join(by_regime[reg_b], on="entity", how="inner", suffix="_b")
                .sort("entity")
            )
            k = joined.height
            total_pairs = k * (k - 1) // 2
            row: dict[str, object] = {
                "regime_a": reg_a,
                "regime_b": reg_b,
                "n_entities": k,
                "total_pairs": total_pairs,
            }
            if total_pairs == 0:
                rows.append(
                    row | {"swap_count": 0, "swap_rate": 0.0, "kendall_tau": 1.0}
                )
                continue
            r_a = joined.get_column("rank").to_numpy().astype(np.float64)
            r_b = joined.get_column("rank_b").to_numpy().astype(np.float64)
            idx_i, idx_j = np.triu_indices(k, k=1)
            prod = (r_a[idx_i] - r_a[idx_j]) * (r_b[idx_i] - r_b[idx_j])
            discordant = int(np.sum(prod < -1e-12))
            concordant = int(np.sum(prod > 1e-12))
            rows.append(
                row
                | {
                    "swap_count": discordant,
                    "swap_rate": float(discordant / total_pairs),
                    "kendall_tau": float((concordant - discordant) / total_pairs),
                }
            )
    if not rows:
        return pl.DataFrame(schema=RANK_SWAP_MATRIX_SCHEMA)
    return pl.DataFrame(rows, schema=RANK_SWAP_MATRIX_SCHEMA)


def kendalls_w(
    frame: pl.LazyFrame | pl.DataFrame,
    metric_col: str,
    *,
    entity_col: str = "planner",
    judge_col: str = "crowd_condition",
    direction: str = "asc",
) -> KendallWResult:
    """Compute Kendall's coefficient of concordance W, Friedman's chi2, and p-value across judges."""
    descending = _descending(direction)
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
            note=(
                f"{m} judge(s) and {k} complete entity(ies) cannot be tested for concordance; "
                "at least two of each are required"
            ),
        )

    rank_sums = (
        complete_df.group_by("entity")
        .agg(r_sum=pl.col("rank").sum())
        .sort("entity")
        .get_column("r_sum")
        .to_numpy()
        .astype(np.float64)
    )
    mean_r = 0.5 * m * (k + 1)
    s_dev = float(np.sum((rank_sums - mean_r) ** 2))

    tie_groups = (
        complete_df.group_by("judge", "rank")
        .agg(t_cnt=pl.len().cast(pl.Int64))
        .filter(pl.col("t_cnt") > 1)
    )
    tie_sum = 0.0
    if not tie_groups.is_empty():
        t_vals = tie_groups.get_column("t_cnt").to_numpy().astype(np.float64)
        tie_sum = float(np.sum(t_vals**3 - t_vals))

    denom = float(m**2 * (k**3 - k) - m * tie_sum)
    w = float(np.clip((12.0 * s_dev) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    df = k - 1
    friedman_chi2 = float(m * df * w)
    if df <= 0 or friedman_chi2 <= 0.0:
        p_value = 1.0
    elif df == 1:
        p_value = float(math.erfc(math.sqrt(friedman_chi2 / 2.0)))
    elif df == 2:
        p_value = float(math.exp(-0.5 * friedman_chi2))
    else:
        # Wilson-Hilferty normal approximation for df > 2
        degrees = float(df)
        z = (
            (friedman_chi2 / degrees) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * degrees))
        ) / math.sqrt(2.0 / (9.0 * degrees))
        p_value = float(0.5 * math.erfc(z / math.sqrt(2.0)))

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
    _require("discovery_fraction", discovery_fraction)
    distinct_seeds = (
        _to_lazy(frame).select(pl.col(seed_col).n_unique()).collect().item()
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


_BEYOND_FENCE: Final[str] = "_beyond_fence"


def _quantile_per_group(values: pl.Expr, quantile: float, groups: list[str]) -> pl.Expr:
    expr = values.quantile(quantile, interpolation="linear")
    return expr.over(groups) if groups else expr


def split_display_outliers(
    frame: pl.DataFrame, *, value: str, groups: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split off the rows whose ``value`` lies beyond 1.5 IQR of their group, for drawing only."""
    group_list = list(groups)
    values = pl.col(value).cast(pl.Float64)
    lower_quartile = _quantile_per_group(values, 0.25, group_list)
    upper_quartile = _quantile_per_group(values, 0.75, group_list)
    spread = upper_quartile - lower_quartile
    beyond = (values < lower_quartile - 1.5 * spread) | (
        values > upper_quartile + 1.5 * spread
    )
    flagged = frame.with_columns(beyond.fill_null(value=False).alias(_BEYOND_FENCE))
    drawn = flagged.filter(~pl.col(_BEYOND_FENCE)).drop(_BEYOND_FENCE)
    if not group_list:
        return drawn, pl.DataFrame({"hidden": [int(flagged[_BEYOND_FENCE].sum())]})
    hidden = (
        flagged.group_by(group_list)
        .agg(pl.col(_BEYOND_FENCE).sum().cast(pl.Int64).alias("hidden"))
        .sort(group_list)
    )
    return drawn, hidden


def aggregate_timeseries_band(
    frame: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    colour_col: str | None,
    bin_width_s: float,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
) -> pl.DataFrame:
    """Mean and percentile band of ``y_col`` over episodes, per colour value and time bin."""
    if colour_col:
        reason_col = f"{colour_col}_null_reason"
        labels = pl.col(colour_col).cast(pl.Utf8)
        if reason_col in frame.columns:
            labels = pl.coalesce([labels, pl.col(reason_col).cast(pl.Utf8)])
        colour_expr = labels.fill_null("NULL").alias("colour")
    else:
        colour_expr = pl.lit("all", dtype=pl.Utf8).alias("colour")

    binned = frame.select(
        colour_expr,
        ((pl.col(x_col).cast(pl.Float64) / bin_width_s).round() * bin_width_s).alias(
            "x"
        ),
        pl.col(y_col).cast(pl.Float64).fill_nan(None).alias("value"),
    ).drop_nulls(["x", "value"])
    return (
        binned.group_by(["colour", "x"])
        .agg(
            pl.col("value").mean().alias("mean"),
            pl.col("value").quantile(low_quantile, interpolation="linear").alias("low"),
            pl.col("value")
            .quantile(high_quantile, interpolation="linear")
            .alias("high"),
            pl.len().alias("n"),
        )
        .sort(["colour", "x"])
    )
