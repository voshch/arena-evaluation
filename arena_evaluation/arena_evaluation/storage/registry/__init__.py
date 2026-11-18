"""The storage registry: each stored table declared once, with the schemas derived from it."""

from __future__ import annotations

from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.fact_io import FACT_KEY_COLUMNS
from arena_evaluation.storage.registry.null_reasons import (
    INVALID_EPISODE_REASONS,
    NULL_REASON_MEANINGS,
    NULL_REASON_PARAMETERISED,
    NullReason,
)
from arena_evaluation.storage.registry.scopes import (
    CHILD_FACT_TABLES,
    SCOPED_FACT_TABLES,
    FactTable,
    MetricScope,
)
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
from arena_evaluation.storage.registry import dimensions, telemetry

__all__ = [
    "CHILD_FACT_TABLES",
    "FACT_KEY_COLUMNS",
    "FACT_TABLES",
    "INVALID_EPISODE_REASONS",
    "METRICS",
    "NULL_REASON_MEANINGS",
    "NULL_REASON_PARAMETERISED",
    "SCOPED_FACT_TABLES",
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
    "telemetry",
]
