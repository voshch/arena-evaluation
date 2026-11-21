"""Analysis store tables: flat index and computation result schemas under ``analysis/``."""

from __future__ import annotations

import polars as pl

from arena_evaluation.storage.registry.table import (
    Column,
    Table,
    TableKind,
)

__all__ = [
    "ClusterBootstrapCIAnalysis",
    "CompositeRankingAnalysis",
    "CorrelationMatrixAnalysis",
    "DistributionBinsAnalysis",
    "EvidenceIndex",
    "GroupSummaryAnalysis",
    "PairedDiffAnalysis",
    "RadarProfileAnalysis",
]


class EvidenceIndex(Table, name="evidence_index", kind=TableKind.ANALYSIS):
    """Flat index of content-addressed evidence records stored at ``analysis/evidence_index.parquet``."""

    key = ("evidence_id",)

    evidence_id = Column(
        pl.String,
        required=True,
        description="Content-addressed evidence identifier (ev-<sha256[:12]>).",
    )
    kind = Column(
        pl.String,
        required=True,
        description="Evidence kind: sql or computation.",
    )
    operation = Column(
        pl.String,
        required=True,
        description="Query name, computation name, or operation label.",
    )
    purpose = Column(
        pl.String,
        description="Human-readable purpose note (excluded from content identity).",
    )
    sql_text = Column(
        pl.String,
        description="SQL statement executed when kind is sql; empty otherwise.",
    )
    snapshot_id = Column(
        pl.String,
        required=True,
        description="Dataset and registry snapshot identifier.",
    )
    params_json = Column(pl.String, required=True, description="Canonical JSON of parameters.")
    row_count = Column(pl.Int64, required=True, description="Number of rows in the recorded frame.")
    schema_signature = Column(pl.String, required=True, description="Sorted column:dtype schema signature.")
    artifact_path = Column(
        pl.String,
        required=True,
        description="Relative path to the evidence Parquet artifact.",
    )
    created_at_ns = Column(pl.Int64, required=True, description="Creation timestamp in nanoseconds.")


class GroupSummaryAnalysis(Table, name="group_summary", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``group_summary`` (preceded by request group columns)."""

    key = ("metric",)
    unique_key = False

    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    n = Column(pl.Int64, required=True, description="Observation count.")
    mean = Column(pl.Float64, unit="metric_unit", description="Group mean.")
    std = Column(pl.Float64, unit="metric_unit", description="Group standard deviation.")
    sem = Column(pl.Float64, unit="metric_unit", description="Standard error of the mean.")
    ci_low = Column(pl.Float64, unit="metric_unit", description="Lower 95% confidence bound.")
    ci_high = Column(pl.Float64, unit="metric_unit", description="Upper 95% confidence bound.")
    err_plus = Column(pl.Float64, unit="metric_unit", description="Upper error-bar delta from mean.")
    err_minus = Column(pl.Float64, unit="metric_unit", description="Lower error-bar delta from mean.")


class DistributionBinsAnalysis(Table, name="distribution_bins", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``distribution_bins`` (preceded by request group columns)."""

    key = ("metric", "bin_index")
    unique_key = False

    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    bin_index = Column(pl.Int64, required=True, description="Zero-based bin index.")
    bin_low = Column(pl.Float64, unit="metric_unit", required=True, description="Lower bin edge.")
    bin_high = Column(pl.Float64, unit="metric_unit", required=True, description="Upper bin edge.")
    bin_center = Column(pl.Float64, unit="metric_unit", required=True, description="Bin midpoint.")
    count = Column(pl.Int64, required=True, description="Observations in bin.")
    density = Column(pl.Float64, unit="1", required=True, description="Normalized bin density.")


class RadarProfileAnalysis(Table, name="radar_profile", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``radar_profile`` (preceded by request group column)."""

    key = ("metric",)
    unique_key = False

    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    direction = Column(pl.String, required=True, description="Optimization direction: higher or lower.")
    raw_mean = Column(pl.Float64, unit="metric_unit", description="Unnormalized group mean.")
    normalized_score = Column(
        pl.Float64,
        unit="1",
        description="Radar-normalized score in [0, 1].",
    )


class CompositeRankingAnalysis(Table, name="composite_ranking", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``composite_ranking`` and ``top_n_ranking``."""

    key = ("metric",)
    unique_key = False

    rank = Column(pl.Int64, description="1-based composite rank (1 is best).")
    composite_score = Column(pl.Float64, unit="1", description="Weighted composite normalized score.")
    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    direction = Column(pl.String, required=True, description="Optimization direction: higher or lower.")
    raw_mean = Column(pl.Float64, unit="metric_unit", description="Unnormalized group mean.")
    normalized_score = Column(pl.Float64, unit="1", description="Min-max normalized metric score in [0, 1].")
    weight = Column(pl.Float64, unit="1", required=True, description="Metric weight.")


class CorrelationMatrixAnalysis(Table, name="correlation_matrix", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``correlation_matrix``."""

    key = ("metric_x", "metric_y")
    unique_key = False

    metric_x = Column(pl.String, required=True, description="First metric name.")
    metric_y = Column(pl.String, required=True, description="Second metric name.")
    pearson_r = Column(pl.Float64, unit="1", description="Pairwise-complete Pearson r.")
    n_pairs = Column(pl.Int64, required=True, description="Number of non-null observation pairs.")


class PairedDiffAnalysis(Table, name="paired_diff", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``paired_diff`` (preceded by planner column)."""

    key = ("metric",)
    unique_key = False

    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    n_pairs = Column(pl.Int64, required=True, description="Number of paired clusters.")
    mean_diff = Column(
        pl.Float64,
        unit="metric_unit",
        description="Mean treatment minus control delta.",
    )
    ci_low = Column(pl.Float64, unit="metric_unit", description="Lower bootstrap CI bound.")
    ci_high = Column(pl.Float64, unit="metric_unit", description="Upper bootstrap CI bound.")
    p_value = Column(pl.Float64, unit="1", description="Two-sided shifted-null p-value.")
    n_clusters = Column(pl.Int64, required=True, description="Cluster count.")


class SeriesProfileAnalysis(Table, name="series_profile", kind=TableKind.ANALYSIS):
    """Aggregated series profile along an x axis (preceded by dynamic group_cols)."""

    unique_key = False

    metric = Column(pl.String, required=True, description="Metric column name.")
    unit = Column(pl.String, description="Physical unit of the metric.")
    x = Column(
        pl.Float64,
        unit="x_unit",
        required=True,
        description="Profile x value or bin centre.",
    )
    n = Column(
        pl.Int64,
        required=True,
        description="Number of non-null observations in cell.",
    )
    value = Column(
        pl.Float64,
        unit="metric_unit",
        description="Cell aggregate value (mean, max, or L_Aeq).",
    )
    err_plus = Column(
        pl.Float64,
        unit="metric_unit",
        description="Upper error bar for reduce='mean' and n >= 2.",
    )
    err_minus = Column(
        pl.Float64,
        unit="metric_unit",
        description="Lower error bar for reduce='mean' and n >= 2.",
    )


class ClusterBootstrapCIAnalysis(Table, name="cluster_bootstrap_ci", kind=TableKind.ANALYSIS):
    """Fixed statistic columns for ``cluster_bootstrap_ci`` (preceded by group columns)."""

    key = ("metric",)
    unique_key = False

    metric = Column(pl.String, required=True, description="Metric name.")
    unit = Column(pl.String, description="Metric unit from dim_metrics.")
    estimate = Column(pl.Float64, unit="metric_unit", description="Cluster-mean point estimate.")
    ci_low = Column(pl.Float64, unit="metric_unit", description="Lower bootstrap CI bound.")
    ci_high = Column(pl.Float64, unit="metric_unit", description="Upper bootstrap CI bound.")
    se = Column(pl.Float64, unit="metric_unit", description="Bootstrap standard error.")
    n_clusters = Column(pl.Int64, required=True, description="Cluster count.")
    n_observations = Column(pl.Int64, required=True, description="Total observation count.")
