"""The metric dictionary: the rows of ``dim_metrics``, built from the metric store and the calculators that publish its metrics."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import (
    load_metric_definitions,
    scope_producers,
    shipped_calculators,
)
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.registry import FACT_TABLES, NullReason, defined_metrics
from arena_evaluation.storage.registry.dimensions import DimMetrics
from arena_evaluation.storage.star_schema import conform


def metric_dictionary(
    calculators: Sequence[type[BaseMetricCalculator]] | None = None,
) -> pl.DataFrame:
    """One row per metric and scope of the shared store, as ``dim_metrics``; the shipped calculators publish them unless others are given."""
    load_metric_definitions()
    publishing = shipped_calculators() if calculators is None else calculators
    directions = _directions(publishing)
    producers = scope_producers(publishing)
    fact_tables = {
        table.table_scope: table.relative_path().stem
        for table in FACT_TABLES.values()
        if table.table_scope is not None
    }
    pairs = [(metric, scope) for metric in defined_metrics() for scope in metric.scopes]
    unpublished = [
        f"{metric.name} at {scope.value}"
        for metric, scope in pairs
        if (metric.name, scope) not in producers
    ]
    if unpublished:
        raise SchemaViolationError(
            f"no calculator publishes the stored metrics {unpublished}"
        )
    columns = {
        "metric": [metric.name for metric, _ in pairs],
        "scope": [scope.value for _, scope in pairs],
        "fact_table": [fact_tables[scope] for _, scope in pairs],
        "dtype": [str(metric.dtype) for metric, _ in pairs],
        "unit": [metric.unit or None for metric, _ in pairs],
        "description": [metric.description for metric, _ in pairs],
        "direction": [directions.get(metric.name) for metric, _ in pairs],
        "producer": [producers[(metric.name, scope)] for metric, scope in pairs],
    }
    return _with_null_reasons(conform(pl.DataFrame(columns), DimMetrics)).sort(
        "metric", "scope"
    )


def _directions(calculators: Sequence[type[BaseMetricCalculator]]) -> dict[str, str]:
    """The better side of each metric a calculator gives one; two calculators giving one metric different sides are refused."""
    declared: dict[str, tuple[str, str]] = {}
    for calculator in calculators:
        for name, direction in calculator.DIRECTIONS.items():
            first_direction, first_owner = declared.setdefault(
                name, (direction.value, calculator.NAME)
            )
            if first_direction != direction.value:
                raise SchemaViolationError(
                    f"metric {name!r} has two directions: {first_direction} from {first_owner} and {direction.value} from {calculator.NAME}"
                )
    return {name: direction for name, (direction, _) in declared.items()}


def _with_null_reasons(frame: pl.DataFrame) -> pl.DataFrame:
    """The frame with the reason of each NULL cell: a unit or a direction that a metric has none of is not applicable."""
    reasons = [
        pl.when(pl.col(column).is_null()).then(
            pl.lit(f"{column}:{NullReason.NOT_APPLICABLE}")
        )
        for column in DimMetrics.null_reason_columns()
    ]
    text = pl.concat_str(reasons, separator=";", ignore_nulls=True)
    return frame.with_columns(null_reason=pl.when(text != "").then(text))
