"""Ranking Stability and Clustered Bootstrap Procedures (Section 6).

Provides deterministic seed-split rank stability analysis, contextual stability
across operational regimes, and paired cluster bootstrap rank intervals for
evaluating planner ranking concordance.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
import polars as pl
import scipy.stats

from arena_evaluation.analytics.catalog import _to_lazy

from .stats import cluster_bootstrap_means

__all__ = [
    "CONTEXTUAL_STABILITY_SCHEMA", "RANK_INTERVAL_SCHEMA", "RANK_SPLITS_SCHEMA",
    "RANK_STABILITY_SCHEMA", "ContextualStabilityResult", "RankInterval",
    "RankStabilityResult", "contextual_rank_stability",
    "paired_cluster_bootstrap_rank_intervals", "seed_split_rank_stability",
]

RANK_SPLITS_SCHEMA: Final[pl.Schema] = pl.Schema({
    "split_index": pl.Int64, "kendall_tau": pl.Float64,
    "top_k_preserved": pl.Boolean, "planners_ranked": pl.Int64})

RANK_STABILITY_SCHEMA: Final[pl.Schema] = pl.Schema({
    "metric": pl.String, "planner_col": pl.String, "seed_col": pl.String,
    "n_splits": pl.Int64, "k": pl.Int64, "mean_tau": pl.Float64,
    "std_tau": pl.Float64, "top_k_preservation_rate": pl.Float64,
    "planners_ranked": pl.Int64, "seeds_ranked": pl.Int64,
    "partially_evaluated": pl.String})

CONTEXTUAL_STABILITY_SCHEMA: Final[pl.Schema] = pl.Schema({
    "arm": pl.String, "world": pl.String, "crowd_size": pl.String,
    "metric": pl.String, "mean_tau": pl.Float64, "std_tau": pl.Float64,
    "top_k_preservation_rate": pl.Float64})

RANK_INTERVAL_SCHEMA: Final[pl.Schema] = pl.Schema({
    "planner": pl.String, "median_rank": pl.Float64, "ci_low": pl.Float64,
    "ci_high": pl.Float64, "seeds_ranked": pl.Int64})


def _records(frame: pl.DataFrame, *columns: str) -> list[tuple[object, ...]]:
    """Read the named frame columns as strict row tuples, preserving column order."""
    values = frame.select(columns).to_dict(as_series=False)
    return list(zip(*(values[column] for column in columns), strict=True))


def _optional_float(value: object) -> float | None:
    """A nullable float column entry with ``None`` preserved."""
    return None if value is None else float(value)


def _rank(values: np.ndarray, method: str) -> np.ndarray:
    """Rank a score vector, the one ranking primitive shared by splits and bootstrap replicates."""
    return scipy.stats.rankdata(values, method=method)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RankInterval:
    """Confidence interval for planner ordinal rank derived from bootstrap resampling."""

    planner: str
    median_rank: float | None
    ci_low: float | None
    ci_high: float | None

    @classmethod
    def from_dataframe(cls, frame: pl.DataFrame) -> list[RankInterval]:
        """Convert a rank intervals dataframe into structured RankInterval instances."""
        return [
            cls(planner=str(planner), median_rank=_optional_float(median),
                ci_low=_optional_float(ci_low), ci_high=_optional_float(ci_high))
            for planner, median, ci_low, ci_high in _records(
                frame, "planner", "median_rank", "ci_low", "ci_high")
        ]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ContextualStabilityResult:
    """Concordance metrics for a specific operational arm, world, and crowd context."""

    arm: str
    world: str
    crowd_size: str
    metric: str
    mean_tau: float
    std_tau: float
    top_k_preservation_rate: float

    @classmethod
    def from_dataframe(cls, frame: pl.DataFrame) -> list[ContextualStabilityResult]:
        """Convert contextual stability dataframe into structured result instances."""
        return [
            cls(arm=str(arm), world=str(world), crowd_size=str(crowd),
                metric=str(metric), mean_tau=float(mean_tau),
                std_tau=float(std_tau), top_k_preservation_rate=float(top_k))
            for arm, world, crowd, metric, mean_tau, std_tau, top_k in _records(
                frame, "arm", "world", "crowd_size", "metric", "mean_tau",
                "std_tau", "top_k_preservation_rate")
        ]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RankStabilityResult:
    """Summary of seed-split rank stability, Kendall tau distribution, and top-k preservation."""

    metric_col: str
    planner_col: str
    seed_col: str
    n_splits: int
    k: int
    mean_tau: float
    std_tau: float
    top_k_preservation_rate: float
    splits_frame: pl.DataFrame
    planners_ranked: int = 0
    seeds_ranked: int = 0
    partially_evaluated: tuple[str, ...] = ()

    @property
    def restriction(self) -> str:
        """One line naming the planners a seed half could not rank, or ``""`` when every split ranked all of them."""
        if not self.partially_evaluated:
            return ""
        return ("a split ranks only the planners with data on both of its seed halves; "
                f"partially evaluated planners={list(self.partially_evaluated)}")

    @property
    def empirical_distribution(self) -> np.ndarray:
        """Empirical distribution of Kendall tau statistics across splits."""
        return self.splits_frame.get_column("kendall_tau").to_numpy()

    def to_frame(self) -> pl.DataFrame:
        """Represent summary metrics as a single-row Polars dataframe."""
        row = dict(zip(RANK_STABILITY_SCHEMA, (
            self.metric_col, self.planner_col, self.seed_col, self.n_splits, self.k,
            self.mean_tau, self.std_tau, self.top_k_preservation_rate,
            self.planners_ranked, self.seeds_ranked,
            ",".join(self.partially_evaluated),
        ), strict=True))
        return pl.DataFrame([row], schema=RANK_STABILITY_SCHEMA)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PlannerSeedMatrices:
    """Dense ``(planner, seed)`` sums and counts of one metric, plus the planners some seed does not cover."""

    planners: np.ndarray
    seeds: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    partially_evaluated: tuple[str, ...] = ()

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame | pl.LazyFrame, metric_col: str, *,
        planner_col: str, seed_col: str,
    ) -> PlannerSeedMatrices | None:
        """Aggregate into dense sums/counts; a half that never observes a planner drops it instead of scoring 0.0."""
        agg = (
            _to_lazy(frame)
            .filter(pl.all_horizontal(
                pl.col(metric_col).is_not_null(), pl.col(planner_col).is_not_null(),
                pl.col(seed_col).is_not_null()))
            .group_by([planner_col, seed_col])
            .agg(metric_sum=pl.col(metric_col).cast(pl.Float64).sum(),
                 metric_count=pl.col(metric_col).count().cast(pl.Float64))
            .sort([planner_col, seed_col])
            .collect()
        )
        if agg.is_empty():
            return None
        planners = agg.get_column(planner_col).unique().sort().to_numpy()
        seeds = agg.get_column(seed_col).unique().sort().to_numpy()
        cells = (np.searchsorted(planners, agg.get_column(planner_col).to_numpy()),
                 np.searchsorted(seeds, agg.get_column(seed_col).to_numpy()))
        sums = np.zeros((len(planners), len(seeds)))
        counts = np.zeros(sums.shape)
        sums[cells] = agg.get_column("metric_sum").to_numpy()
        counts[cells] = agg.get_column("metric_count").to_numpy()
        covered = np.all(counts > 0.0, axis=1)
        partial = tuple(str(p) for p, ok in zip(planners, covered, strict=True) if not ok)
        return cls(planners=planners, seeds=seeds, sums=sums, counts=counts,
                   partially_evaluated=partial)

    def _half_means(self, seed_indices: np.ndarray) -> np.ndarray:
        """Weighted planner means over a half-sample of seed columns, NULL where the half has no data."""
        half_sums = np.sum(self.sums[:, seed_indices], axis=1)
        half_counts = np.sum(self.counts[:, seed_indices], axis=1)
        return np.where(half_counts > 0, half_sums / np.maximum(half_counts, 1.0), np.nan)

    def split_metrics(
        self, *, idx_1: np.ndarray, idx_2: np.ndarray, score_sign: float, eff_k: int,
    ) -> tuple[float, bool, int]:
        """Kendall's tau, top-k match and ranked planner count for one split; partial planners drop."""
        means_1, means_2 = self._half_means(idx_1), self._half_means(idx_2)
        ranked = np.isfinite(means_1) & np.isfinite(means_2)
        n_ranked = int(np.count_nonzero(ranked))
        if n_ranked == 0:
            return float("nan"), False, 0
        if n_ranked == 1:
            return 1.0, True, 1
        observed_1 = score_sign * means_1[ranked]
        observed_2 = score_sign * means_2[ranked]
        ranks_1 = _rank(observed_1, "average")
        ranks_2 = _rank(observed_2, "average")
        order_1, order_2 = np.argsort(observed_1), np.argsort(observed_2)
        tau_raw = float(scipy.stats.kendalltau(ranks_1, ranks_2)[0])
        tau_val = (tau_raw if not math.isnan(tau_raw)
                   else (1.0 if np.array_equal(ranks_1, ranks_2) else 0.0))
        split_k = min(eff_k, n_ranked)
        top_k_idx_1, top_k_idx_2 = order_1[:split_k], order_2[:split_k]
        is_preserved = set(top_k_idx_1) == set(top_k_idx_2) and np.all(
            ranks_1[top_k_idx_1] == ranks_2[top_k_idx_1])
        return tau_val, bool(is_preserved), n_ranked


def _seed_split_statistics(taus: np.ndarray, preserved: np.ndarray) -> tuple[float, float, float]:
    """Mean and standard deviation of comparable Kendall taus plus their top-k preservation rate."""
    comparable = np.isfinite(taus)
    if not comparable.any():
        return float("nan"), float("nan"), float("nan")
    selected = taus[comparable]
    std = float(selected.std(ddof=1)) if selected.size > 1 else 0.0
    return float(selected.mean()), std, float(preserved[comparable].mean())


def seed_split_rank_stability(
    frame: pl.DataFrame | pl.LazyFrame,
    metric_col: str,
    *,
    planner_col: str = "planner",
    seed_col: str = "seed",
    n_splits: int = 100,
    k: int = 3,
    descending: bool = True,
    seed: int = 42,
) -> RankStabilityResult:
    """Compute ranking concordance across random seed halvings and top-k preservation rate.

    Each split ranks only planners observed in both halves (see ``split_metrics``); a planner
    missing from a half is dropped and named on the result, never scored with a fabricated 0.0.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    matrices = PlannerSeedMatrices.from_frame(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col)
    if matrices is None or matrices.sums.shape[1] < 2:
        return RankStabilityResult(
            metric_col=metric_col, planner_col=planner_col, seed_col=seed_col,
            n_splits=0, k=k, mean_tau=0.0, std_tau=0.0, top_k_preservation_rate=0.0,
            splits_frame=pl.DataFrame(schema=RANK_SPLITS_SCHEMA),
            planners_ranked=0 if matrices is None else int(matrices.sums.shape[0]),
            seeds_ranked=0 if matrices is None else int(matrices.sums.shape[1]),
            partially_evaluated=() if matrices is None else matrices.partially_evaluated,
        )
    n_planners, n_seeds = matrices.sums.shape
    rng = np.random.default_rng(seed)
    half_size = n_seeds // 2
    eff_k = min(k, n_planners)
    score_sign = -1.0 if descending else 1.0
    taus: list[float] = []
    preserved: list[bool] = []
    ranked_counts: list[int] = []
    for _ in range(n_splits):
        perm = rng.permutation(n_seeds)
        tau_val, is_preserved, n_ranked = matrices.split_metrics(
            idx_1=perm[:half_size], idx_2=perm[half_size:],
            score_sign=score_sign, eff_k=eff_k)
        taus.append(tau_val)
        preserved.append(is_preserved)
        ranked_counts.append(n_ranked)
    tau_arr = np.array(taus)
    preserved_arr = np.array(preserved, dtype=bool)
    mean_tau, std_tau, top_k_rate = _seed_split_statistics(tau_arr, preserved_arr)
    splits_frame = pl.DataFrame(
        {"split_index": np.arange(len(taus), dtype=np.int64),
         "kendall_tau": tau_arr, "top_k_preserved": preserved_arr,
         "planners_ranked": np.array(ranked_counts, dtype=np.int64)},
        schema=RANK_SPLITS_SCHEMA)
    return RankStabilityResult(
        metric_col=metric_col, planner_col=planner_col, seed_col=seed_col,
        n_splits=len(taus), k=eff_k, mean_tau=mean_tau, std_tau=std_tau,
        top_k_preservation_rate=top_k_rate, splits_frame=splits_frame,
        planners_ranked=int(max(ranked_counts)), seeds_ranked=n_seeds,
        partially_evaluated=matrices.partially_evaluated)


def _stability_row(
    result: RankStabilityResult, *, arm: str, world: str, crowd: str, metric: str
) -> dict[str, object]:
    """Project a seed-split summary onto one contextual schema row."""
    return {"arm": arm, "world": world, "crowd_size": crowd, "metric": metric,
            "mean_tau": result.mean_tau, "std_tau": result.std_tau,
            "top_k_preservation_rate": result.top_k_preservation_rate}


def contextual_rank_stability(
    frame: pl.DataFrame | pl.LazyFrame,
    metric_col: str,
    *,
    arm_col: str = "arm",
    world_col: str = "world",
    crowd_col: str = "crowd_size",
    planner_col: str = "planner",
    seed_col: str = "seed",
    n_splits: int = 50,
    descending: bool = True,
    k: int = 3,
    seed: int = 42,
) -> pl.DataFrame:
    """Compute Kendall tau rank stability across contextual environments and crowd densities."""
    df = _to_lazy(frame).filter(pl.all_horizontal(
        pl.col(metric_col).is_not_null(), pl.col(planner_col).is_not_null(),
        pl.col(seed_col).is_not_null(), pl.col(arm_col).is_not_null(),
        pl.col(world_col).is_not_null(), pl.col(crowd_col).is_not_null())).collect()
    if df.is_empty():
        return pl.DataFrame(schema=CONTEXTUAL_STABILITY_SCHEMA)
    rows = [
        _stability_row(
            seed_split_rank_stability(
                group, metric_col, planner_col=planner_col, seed_col=seed_col,
                n_splits=n_splits, k=k, descending=descending, seed=seed),
            arm=str(group.item(0, arm_col)), world=str(group.item(0, world_col)),
            crowd=str(group.item(0, crowd_col)), metric=metric_col)
        for group in df.partition_by([arm_col, world_col, crowd_col], maintain_order=True)
    ]
    return pl.DataFrame(rows, schema=CONTEXTUAL_STABILITY_SCHEMA)


def _interval_rows(
    planners: np.ndarray, ranks: np.ndarray, seed_counts: np.ndarray, alpha: float,
) -> list[dict[str, object]]:
    """Project bootstrap rank replicates onto the interval schema, one row per planner."""
    return [
        {"planner": str(planner), "median_rank": float(median), "ci_low": float(low),
         "ci_high": float(high), "seeds_ranked": int(seed_count)}
        for planner, median, low, high, seed_count in zip(
            planners, np.median(ranks, axis=0),
            np.quantile(ranks, alpha / 2.0, axis=0),
            np.quantile(ranks, 1.0 - alpha / 2.0, axis=0),
            seed_counts, strict=True)
    ]


def paired_cluster_bootstrap_rank_intervals(
    frame: pl.DataFrame | pl.LazyFrame,
    metric_col: str,
    *,
    planner_col: str = "planner",
    seed_col: str = "seed",
    n_bootstrap: int = 500,
    ci: float = 0.95,
    descending: bool = True,
    seed: int = 42,
    rank_method: str = "dense",
) -> pl.DataFrame:
    """Compute median ranks and paired cluster-bootstrap CIs (cluster unit: ``seed_col``); only
    planners observed on every seed are ranked, and partial planners keep a NULL interval row."""
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    matrices = PlannerSeedMatrices.from_frame(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col)
    if matrices is None:
        return pl.DataFrame(schema=RANK_INTERVAL_SCHEMA)
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci
    score_sign = -1.0 if descending else 1.0
    covered = np.all(matrices.counts > 0.0, axis=1)
    seed_counts = np.count_nonzero(matrices.counts > 0.0, axis=1)
    rows: list[dict[str, object]] = []
    if covered.any():
        boot_means = cluster_bootstrap_means(
            matrices.sums[covered], matrices.counts[covered],
            n_boot=n_bootstrap, rng=rng)
        ranks = np.array([
            _rank(score_sign * boot_means[:, replicate], rank_method)
            for replicate in range(n_bootstrap)])
        rows.extend(_interval_rows(
            matrices.planners[covered], ranks, seed_counts[covered], alpha))
    rows.extend(
        {"planner": str(planner), "median_rank": None, "ci_low": None,
         "ci_high": None, "seeds_ranked": int(seed_count)}
        for planner, seed_count in zip(
            matrices.planners[~covered], seed_counts[~covered], strict=True))
    return pl.DataFrame(rows, schema=RANK_INTERVAL_SCHEMA).sort(
        ["median_rank", "ci_low", "planner"], nulls_last=True)
