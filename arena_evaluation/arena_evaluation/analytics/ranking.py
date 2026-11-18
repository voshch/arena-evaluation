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
        "planners_ranked": pl.Int64,
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
        "planners_ranked": pl.Int64,
        "seeds_ranked": pl.Int64,
        "partially_evaluated": pl.String,
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
        "seeds_ranked": pl.Int64,
    }
)


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
        cols = frame.to_dict(as_series=False)
        return [
            cls(
                planner=str(planner),
                median_rank=None if median is None else float(median),
                ci_low=None if ci_low is None else float(ci_low),
                ci_high=None if ci_high is None else float(ci_high),
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
    planners_ranked: int = 0
    seeds_ranked: int = 0
    partially_evaluated: tuple[str, ...] = ()

    @property
    def restriction(self) -> str:
        """One line naming the planners a seed half could not rank, or ``""`` when every split ranked all of them."""
        if not self.partially_evaluated:
            return ""
        return (
            "a split ranks only the planners with data on both of its seed halves; "
            f"partially evaluated planners={list(self.partially_evaluated)}"
        )

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
                    "planners_ranked": self.planners_ranked,
                    "seeds_ranked": self.seeds_ranked,
                    "partially_evaluated": ",".join(self.partially_evaluated),
                }
            ],
            schema=RANK_STABILITY_SCHEMA,
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PlannerSeedMatrices:
    """Dense ``(planner, seed)`` sums and counts of one metric, plus the planners some seed does not cover."""

    planners: np.ndarray
    seeds: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    partially_evaluated: tuple[str, ...] = ()


def _build_planner_seed_matrices(
    frame: pl.DataFrame | pl.LazyFrame,
    metric_col: str,
    *,
    planner_col: str,
    seed_col: str,
) -> PlannerSeedMatrices | None:
    """Aggregate the observations into dense ``(n_planners, n_seeds)`` sums and counts.

    A cell a planner has no data for keeps a zero count. A seeding half that draws only such cells has
    no observation to rank that planner on, so the ranking products drop the planner from that half
    instead of averaging a fabricated 0.0 into it; the planners some seed does not cover are reported.
    """
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

    fully_covered = np.all(counts > 0.0, axis=1)
    return PlannerSeedMatrices(
        planners=planner_arr,
        seeds=seed_arr,
        sums=sums,
        counts=counts,
        partially_evaluated=tuple(
            str(planner)
            for planner, covered in zip(planner_arr, fully_covered, strict=True)
            if not covered
        ),
    )


def _half_means(
    sums: np.ndarray,
    counts: np.ndarray,
    seed_indices: np.ndarray,
) -> np.ndarray:
    """Compute weighted planner means over a half-sample of seed columns, NULL where the half holds no data."""
    half_sums = np.sum(sums[:, seed_indices], axis=1)
    half_counts = np.sum(counts[:, seed_indices], axis=1)
    return np.where(
        half_counts > 0, half_sums / np.where(half_counts > 0, half_counts, 1.0), np.nan
    )


def _evaluate_seed_split(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    idx_1: np.ndarray,
    idx_2: np.ndarray,
    score_sign: float,
    eff_k: int,
) -> tuple[float, bool, int]:
    """Compute Kendall's tau, top-k match and ranked planner count for one seed split.

    Only the planners both halves observed are ranked; a planner absent from either half has no
    position in that split, so it is dropped from the split rather than scored with a made-up 0.0.
    """
    means_1 = _half_means(sums, counts, idx_1)
    means_2 = _half_means(sums, counts, idx_2)
    ranked = np.isfinite(means_1) & np.isfinite(means_2)
    n_ranked = int(np.count_nonzero(ranked))
    if n_ranked == 0:
        return float("nan"), False, 0
    if n_ranked == 1:
        return 1.0, True, 1

    observed_1 = score_sign * means_1[ranked]
    observed_2 = score_sign * means_2[ranked]
    ranks_1 = scipy.stats.rankdata(observed_1, method="average")
    ranks_2 = scipy.stats.rankdata(observed_2, method="average")
    order_1 = np.argsort(observed_1)
    order_2 = np.argsort(observed_2)

    tau_res = typing.cast(tuple[float, float], scipy.stats.kendalltau(ranks_1, ranks_2))
    tau_raw = float(tau_res[0])
    tau_val = (
        tau_raw
        if not math.isnan(tau_raw)
        else (1.0 if np.array_equal(ranks_1, ranks_2) else 0.0)
    )

    split_k = min(eff_k, n_ranked)
    top_k_idx_1 = order_1[:split_k]
    top_k_idx_2 = order_2[:split_k]
    is_preserved = bool(
        set(top_k_idx_1) == set(top_k_idx_2)
        and np.all(ranks_1[top_k_idx_1] == ranks_2[top_k_idx_1])
    )
    return tau_val, is_preserved, n_ranked


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

    Every split ranks only the planners it observed in both halves (see :func:`_evaluate_seed_split`);
    a planner missing from a half is dropped from that split and named on the result, so a partial
    evaluation can neither fabricate its mean as 0.0 nor drag the other planners' agreement down.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    matrices = _build_planner_seed_matrices(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col
    )
    if matrices is None or matrices.sums.shape[1] < 2:
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
            planners_ranked=0 if matrices is None else int(matrices.sums.shape[0]),
            seeds_ranked=0 if matrices is None else int(matrices.sums.shape[1]),
            partially_evaluated=(
                () if matrices is None else matrices.partially_evaluated
            ),
        )
    sums, counts = matrices.sums, matrices.counts
    n_planners, n_seeds = sums.shape
    rng = np.random.default_rng(seed)
    half_size = n_seeds // 2
    eff_k = min(k, n_planners)
    score_sign = -1.0 if descending else 1.0

    taus: list[float] = []
    preserved: list[bool] = []
    ranked_counts: list[int] = []
    for _ in range(n_splits):
        perm = rng.permutation(n_seeds)
        tau_val, is_preserved, n_ranked = _evaluate_seed_split(
            sums,
            counts,
            idx_1=perm[:half_size],
            idx_2=perm[half_size:],
            score_sign=score_sign,
            eff_k=eff_k,
        )
        taus.append(tau_val)
        preserved.append(is_preserved)
        ranked_counts.append(n_ranked)

    tau_arr = np.array(taus, dtype=np.float64)
    comparable = np.isfinite(tau_arr)
    splits_frame = pl.DataFrame(
        {
            "split_index": np.arange(len(taus), dtype=np.int64),
            "kendall_tau": tau_arr,
            "top_k_preserved": np.array(preserved, dtype=bool),
            "planners_ranked": np.array(ranked_counts, dtype=np.int64),
        },
        schema=RANK_SPLITS_SCHEMA,
    )
    return RankStabilityResult(
        metric_col=metric_col,
        planner_col=planner_col,
        seed_col=seed_col,
        n_splits=len(taus),
        k=eff_k,
        mean_tau=(
            float(tau_arr[comparable].mean()) if comparable.any() else float("nan")
        ),
        std_tau=(
            float(tau_arr[comparable].std(ddof=1))
            if int(np.count_nonzero(comparable)) > 1
            else (0.0 if comparable.any() else float("nan"))
        ),
        top_k_preservation_rate=(
            float(np.array(preserved, dtype=bool)[comparable].mean())
            if comparable.any()
            else float("nan")
        ),
        splits_frame=splits_frame,
        planners_ranked=int(max(ranked_counts)),
        seeds_ranked=n_seeds,
        partially_evaluated=matrices.partially_evaluated,
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
    """Compute median ranks and paired cluster-bootstrap CIs (cluster unit: ``seed_col``, default ``seed``).

    A rank interval compares a planner across every seed, so only the planners observed on all of them are
    ranked; a planner with partial seed coverage keeps its row with a NULL interval instead of a
    confidently narrow one computed from the replicates that happened to draw its few seeds.
    """
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    matrices = _build_planner_seed_matrices(
        frame, metric_col, planner_col=planner_col, seed_col=seed_col
    )
    if matrices is None:
        return pl.DataFrame(schema=RANK_INTERVAL_SCHEMA)
    planner_arr, sums, counts = matrices.planners, matrices.sums, matrices.counts

    rng = np.random.default_rng(seed)
    score_sign = -1.0 if descending else 1.0
    covered = np.all(counts > 0.0, axis=1)
    alpha = 1.0 - ci
    rows: list[dict[str, object]] = []
    if covered.any():
        boot_means = cluster_bootstrap_means(
            sums[covered], counts[covered], n_boot=n_bootstrap, rng=rng
        )
        ranked_planners = int(np.count_nonzero(covered))
        ranks_matrix = np.empty((n_bootstrap, ranked_planners), dtype=np.float64)
        for b in range(n_bootstrap):
            ranks_matrix[b, :] = scipy.stats.rankdata(
                score_sign * boot_means[:, b], method=rank_method
            )
        for rank_index, position in enumerate(np.flatnonzero(covered)):
            column = ranks_matrix[:, rank_index]
            rows.append(
                {
                    "planner": str(planner_arr[position]),
                    "median_rank": float(np.median(column)),
                    "ci_low": float(np.quantile(column, alpha / 2.0)),
                    "ci_high": float(np.quantile(column, 1.0 - alpha / 2.0)),
                    "seeds_ranked": int(np.count_nonzero(counts[position] > 0.0)),
                }
            )
    for position in np.flatnonzero(~covered):
        rows.append(
            {
                "planner": str(planner_arr[position]),
                "median_rank": None,
                "ci_low": None,
                "ci_high": None,
                "seeds_ranked": int(np.count_nonzero(counts[position] > 0.0)),
            }
        )
    return pl.DataFrame(rows, schema=RANK_INTERVAL_SCHEMA).sort(
        ["median_rank", "ci_low", "planner"], nulls_last=True
    )
