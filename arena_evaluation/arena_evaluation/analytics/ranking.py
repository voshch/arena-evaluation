"""Ranking Stability and Clustered Bootstrap Procedures (Section 6).

Provides deterministic seed-split rank stability analysis, contextual stability
across operational regimes, and paired cluster bootstrap rank intervals for
evaluating planner ranking concordance.
"""

from __future__ import annotations

import dataclasses
import math
import typing
from typing import Final

import numpy as np
import polars as pl
import scipy.stats

from arena_evaluation.analytics.catalog import _to_lazy

from .stats import cluster_bootstrap_means

__all__ = [
    "CONTEXTUAL_STABILITY_SCHEMA",
    "RANK_INTERVAL_SCHEMA",
    "RANK_SPLITS_SCHEMA",
    "RANK_STABILITY_SCHEMA",
    "ContextualStabilityResult",
    "RankInterval",
    "RankStabilityResult",
    "contextual_rank_stability",
    "paired_cluster_bootstrap_rank_intervals",
    "seed_split_rank_stability",
]

RANK_SPLITS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "split_index": pl.Int64,
        "kendall_tau": pl.Float64,
        "top_k_preserved": pl.Boolean,
    }
)

RANK_STABILITY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "metric": pl.String,
        "planner_col": pl.String,
        "seed_col": pl.String,
        "n_splits": pl.Int64,
        "k": pl.Int64,
        "mean_tau": pl.Float64,
        "std_tau": pl.Float64,
        "top_k_preservation_rate": pl.Float64,
    }
)

CONTEXTUAL_STABILITY_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "arm": pl.String,
        "world": pl.String,
        "crowd_size": pl.String,
        "metric": pl.String,
        "mean_tau": pl.Float64,
        "std_tau": pl.Float64,
        "top_k_preservation_rate": pl.Float64,
    }
)

RANK_INTERVAL_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "planner": pl.String,
        "median_rank": pl.Float64,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
    }
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RankInterval:
    """Confidence interval for planner ordinal rank derived from bootstrap resampling."""

    planner: str
    median_rank: float
    ci_low: float
    ci_high: float

    @classmethod
    def from_dataframe(cls, frame: pl.DataFrame) -> list[RankInterval]:
        """Convert a rank intervals dataframe into structured RankInterval instances."""
        cols = frame.to_dict(as_series=False)
        return [
            cls(
                planner=str(planner),
                median_rank=float(median),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
            )
            for planner, median, ci_low, ci_high in zip(
                cols["planner"],
                cols["median_rank"],
                cols["ci_low"],
                cols["ci_high"],
                strict=True,
            )
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
        cols = frame.to_dict(as_series=False)
        return [
            cls(
                arm=str(arm),
                world=str(world),
                crowd_size=str(crowd),
                metric=str(metric),
                mean_tau=float(mean_tau),
                std_tau=float(std_tau),
                top_k_preservation_rate=float(top_k),
            )
            for arm, world, crowd, metric, mean_tau, std_tau, top_k in zip(
                cols["arm"],
                cols["world"],
                cols["crowd_size"],
                cols["metric"],
                cols["mean_tau"],
                cols["std_tau"],
                cols["top_k_preservation_rate"],
                strict=True,
            )
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

    @property
    def empirical_distribution(self) -> np.ndarray:
        """Empirical distribution of Kendall tau statistics across splits."""
        return self.splits_frame.get_column("kendall_tau").to_numpy()

    def to_frame(self) -> pl.DataFrame:
        """Represent summary metrics as a single-row Polars dataframe."""
        return pl.DataFrame(
            [
                {
                    "metric": self.metric_col,
                    "planner_col": self.planner_col,
                    "seed_col": self.seed_col,
                    "n_splits": self.n_splits,
                    "k": self.k,
                    "mean_tau": self.mean_tau,
                    "std_tau": self.std_tau,
                    "top_k_preservation_rate": self.top_k_preservation_rate,
                }
            ],
            schema=RANK_STABILITY_SCHEMA,
        )


def _build_planner_seed_matrices(
    frame: pl.DataFrame | pl.LazyFrame,
    metric_col: str,
    *,
    planner_col: str,
    seed_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aggregate valid observations into dense ``(n_planners, n_seeds)`` sum and count arrays."""
    lf = _to_lazy(frame)
    agg_df = (
        lf.filter(
            pl.col(metric_col).is_not_null()
            & pl.col(planner_col).is_not_null()
            & pl.col(seed_col).is_not_null()
        )
        .group_by([planner_col, seed_col])
        .agg(
            metric_sum=pl.col(metric_col).cast(pl.Float64).sum(),
            metric_count=pl.col(metric_col).count().cast(pl.Float64),
        )
        .sort([planner_col, seed_col])
        .collect()
    )
    if agg_df.is_empty():
        return None
    planner_arr = agg_df.get_column(planner_col).unique().sort().to_numpy()
    seed_arr = agg_df.get_column(seed_col).unique().sort().to_numpy()
    planner_idx = np.searchsorted(
        planner_arr, agg_df.get_column(planner_col).to_numpy()
    )
    seed_idx = np.searchsorted(seed_arr, agg_df.get_column(seed_col).to_numpy())
    sums = np.zeros((len(planner_arr), len(seed_arr)), dtype=np.float64)
    counts = np.zeros((len(planner_arr), len(seed_arr)), dtype=np.float64)
    sums[planner_idx, seed_idx] = agg_df.get_column("metric_sum").to_numpy()
    counts[planner_idx, seed_idx] = agg_df.get_column("metric_count").to_numpy()
    return planner_arr, sums, counts


def _half_means(
    sums: np.ndarray,
    counts: np.ndarray,
    seed_indices: np.ndarray,
) -> np.ndarray:
    """Compute weighted planner means over a half-sample of seed columns."""
    half_sums = np.sum(sums[:, seed_indices], axis=1)
    half_counts = np.sum(counts[:, seed_indices], axis=1)
    return np.where(half_counts > 0, half_sums / np.maximum(half_counts, 1), 0.0)


def _evaluate_seed_split(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    score_sign: float,
    eff_k: int,
) -> tuple[float, bool]:
    """Compute Kendall's tau and top-k match for one seed split."""
    means_1 = _half_means(sums, counts, idx_1)
    means_2 = _half_means(sums, counts, idx_2)
    ranks_1 = scipy.stats.rankdata(score_sign * means_1, method="average")
    ranks_2 = scipy.stats.rankdata(score_sign * means_2, method="average")
    order_1 = np.argsort(score_sign * means_1)
    order_2 = np.argsort(score_sign * means_2)

    if sums.shape[0] < 2:
        tau_val = 1.0
    else:
        tau_res = typing.cast(
            tuple[float, float], scipy.stats.kendalltau(ranks_1, ranks_2)
        )
        tau_raw = float(tau_res[0])
        if math.isnan(tau_raw):
            tau_val = 1.0 if np.array_equal(ranks_1, ranks_2) else 0.0
        else:
            tau_val = tau_raw

    top_k_idx_1 = order_1[:eff_k]
    top_k_idx_2 = order_2[:eff_k]
    is_preserved = bool(
        set(top_k_idx_1) == set(top_k_idx_2)
        and np.all(ranks_1[top_k_idx_1] == ranks_2[top_k_idx_1])
    )
    return tau_val, is_preserved


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
    """Compute ranking concordance across random seed halvings and top-k preservation rate."""
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    matrices = _build_planner_seed_matrices(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col
    )
    if matrices is None or matrices[1].shape[1] < 2:
        return RankStabilityResult(
            metric_col=metric_col,
            planner_col=planner_col,
            seed_col=seed_col,
            n_splits=0,
            k=k,
            mean_tau=0.0,
            std_tau=0.0,
            top_k_preservation_rate=0.0,
            splits_frame=pl.DataFrame(schema=RANK_SPLITS_SCHEMA),
        )
    planner_arr, sums, counts = matrices
    n_planners, n_seeds = sums.shape
    rng = np.random.default_rng(seed)
    half_size = n_seeds // 2
    eff_k = min(k, n_planners)
    score_sign = -1.0 if descending else 1.0

    taus: list[float] = []
    preserved: list[bool] = []
    for _ in range(n_splits):
        perm = rng.permutation(n_seeds)
        tau_val, is_preserved = _evaluate_seed_split(
            sums,
            counts,
            idx_1=perm[:half_size],
            idx_2=perm[half_size:],
            score_sign=score_sign,
            eff_k=eff_k,
        )
        taus.append(tau_val)
        preserved.append(is_preserved)

    splits_frame = pl.DataFrame(
        {
            "split_index": np.arange(len(taus), dtype=np.int64),
            "kendall_tau": np.array(taus, dtype=np.float64),
            "top_k_preserved": np.array(preserved, dtype=bool),
        },
        schema=RANK_SPLITS_SCHEMA,
    )
    return RankStabilityResult(
        metric_col=metric_col,
        planner_col=planner_col,
        seed_col=seed_col,
        n_splits=len(taus),
        k=eff_k,
        mean_tau=float(np.mean(taus)),
        std_tau=float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0,
        top_k_preservation_rate=float(np.mean(preserved)),
        splits_frame=splits_frame,
    )


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
    lf = _to_lazy(frame)
    df = lf.filter(
        pl.col(metric_col).is_not_null()
        & pl.col(planner_col).is_not_null()
        & pl.col(seed_col).is_not_null()
        & pl.col(arm_col).is_not_null()
        & pl.col(world_col).is_not_null()
        & pl.col(crowd_col).is_not_null()
    ).collect()
    if df.is_empty():
        return pl.DataFrame(schema=CONTEXTUAL_STABILITY_SCHEMA)

    rows: list[dict[str, object]] = []
    for grp in df.partition_by([arm_col, world_col, crowd_col], maintain_order=True):
        res = seed_split_rank_stability(
            grp,
            metric_col,
            planner_col=planner_col,
            seed_col=seed_col,
            n_splits=n_splits,
            k=k,
            descending=descending,
            seed=seed,
        )
        rows.append(
            {
                "arm": str(grp.item(0, arm_col)),
                "world": str(grp.item(0, world_col)),
                "crowd_size": str(grp.item(0, crowd_col)),
                "metric": metric_col,
                "mean_tau": res.mean_tau,
                "std_tau": res.std_tau,
                "top_k_preservation_rate": res.top_k_preservation_rate,
            }
        )
    return pl.DataFrame(rows, schema=CONTEXTUAL_STABILITY_SCHEMA)


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
    """Compute median ranks and paired cluster-bootstrap CIs (cluster unit: ``seed_col``, default ``seed``)."""
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    matrices = _build_planner_seed_matrices(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col
    )
    if matrices is None:
        return pl.DataFrame(schema=RANK_INTERVAL_SCHEMA)
    planner_arr, sums, counts = matrices
    n_planners = len(planner_arr)

    rng = np.random.default_rng(seed)
    boot_means = cluster_bootstrap_means(sums, counts, n_boot=n_bootstrap, rng=rng)
    score_sign = -1.0 if descending else 1.0
    ranks_matrix = np.empty((n_bootstrap, n_planners), dtype=np.float64)
    for b in range(n_bootstrap):
        ranks_matrix[b, :] = scipy.stats.rankdata(
            score_sign * boot_means[:, b], method=rank_method
        )

    alpha = 1.0 - ci
    return pl.DataFrame(
        {
            "planner": planner_arr.astype(str),
            "median_rank": np.median(ranks_matrix, axis=0).astype(np.float64),
            "ci_low": np.quantile(ranks_matrix, alpha / 2.0, axis=0).astype(np.float64),
            "ci_high": np.quantile(ranks_matrix, 1.0 - alpha / 2.0, axis=0).astype(
                np.float64
            ),
        },
        schema=RANK_INTERVAL_SCHEMA,
    ).sort(["median_rank", "ci_low", "planner"])
