"""The storage registry: each stored table declared once, with the schemas derived from it."""

from __future__ import annotations

from arena_evaluation.storage.registry.facts import FACT_KEY_COLUMNS, FACT_TABLES
from arena_evaluation.storage.registry.null_reasons import (
    INVALID_EPISODE_REASONS,
    NULL_REASON_MEANINGS,
    NULL_REASON_PARAMETERISED,
    NullReason,
)
from arena_evaluation.storage.registry.scopes import FactTable, MetricScope
from arena_evaluation.storage.registry.table import (
    METRICS,
    Column,
    Metric,
    MetricStore,
    Table,
    TableKind,
    define_metric,
    defined_metrics,
    registered_tables,
)
from arena_evaluation.storage.registry import dimensions, series, telemetry

__all__ = [
    "FACT_KEY_COLUMNS",
    "FACT_TABLES",
    "INVALID_EPISODE_REASONS",
    "METRICS",
    "NULL_REASON_MEANINGS",
    "NULL_REASON_PARAMETERISED",
    "Column",
    "FactTable",
    "Metric",
    "MetricScope",
    "MetricStore",
    "NullReason",
    "Table",
    "TableKind",
    "define_metric",
    "defined_metrics",
    "dimensions",
    "registered_tables",
    "series",
    "telemetry",
]
