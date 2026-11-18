"""The metric dictionary: one row per metric and scope of the shared store, published as the dimension ``dim_metrics``."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.processing.metrics.registry import (
    scope_producers,
    shipped_calculators,
)
from arena_evaluation.storage.contracts import enforce_dimension_contract
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    MetricScope,
    NullReason,
    defined_metrics,
)
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES, DimMetrics
from arena_evaluation.storage.registry.table import Metric, TableKind


def _planted_metric(name: str) -> Metric:
    """A metric built outside the shared store, so no other test sees it."""
    return Metric(
        pl.Float64,
        unit="m",
        description=f"planted metric {name}",
        name=name,
        scopes=(MetricScope.EPISODE, MetricScope.ROBOT),
    )


class _HigherAtEpisode(BaseMetricCalculator):
    NAME = "planted_higher_at_episode"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.EPISODE,)}
    DIRECTIONS = {"planted_distance_m": Direction.HIGHER}

    def calculate(self, episode, prior_results):
        return {}


class _HigherAtRobot(BaseMetricCalculator):
    NAME = "planted_higher_at_robot"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.ROBOT,)}
    DIRECTIONS = {"planted_distance_m": Direction.HIGHER}

    def calculate(self, episode, prior_results):
        return {}


class _LowerAtRobot(BaseMetricCalculator):
    NAME = "planted_lower_at_robot"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.ROBOT,)}
    DIRECTIONS = {"planted_distance_m": Direction.LOWER}

    def calculate(self, episode, prior_results):
        return {}


def _store_pairs() -> set[tuple[str, str]]:
    return {
        (metric.name, scope.value)
        for metric in defined_metrics()
        for scope in metric.scopes
    }


def test_the_dictionary_has_one_row_per_metric_and_scope_of_the_store() -> None:
    frame = metric_dictionary()

    assert frame.height == len(_store_pairs())
    assert set(zip(frame["metric"], frame["scope"], strict=True)) == _store_pairs()


def test_a_row_names_the_fact_table_that_stores_the_metric_at_its_scope() -> None:
    frame = metric_dictionary()
    table_of_scope = {
        table.table_scope.value: table
        for table in FACT_TABLES.values()
        if table.table_scope is not None
    }

    for row in frame.iter_rows(named=True):
        table = table_of_scope[row["scope"]]
        assert row["fact_table"] == f"fact_{table.table_name}"
        assert row["metric"] in table.polars_schema()


def test_a_row_carries_the_dtype_unit_and_description_of_the_stored_metric() -> None:
    frame = metric_dictionary()
    rows = {(row["metric"], row["scope"]): row for row in frame.iter_rows(named=True)}

    for metric in defined_metrics():
        for scope in metric.scopes:
            row = rows[(metric.name, scope.value)]
            assert row["dtype"] == str(metric.dtype)
            assert row["unit"] == (metric.unit or None)
            assert row["description"] == metric.description


def test_mar_is_higher_is_better_in_the_dictionary() -> None:
    mar = metric_dictionary().filter(pl.col("metric") == "mar")

    assert mar["direction"].to_list() == ["higher"]


def test_a_metric_without_a_direction_has_a_null_direction_with_its_reason() -> None:
    frame = metric_dictionary()
    directed = {
        name for calculator in shipped_calculators() for name in calculator.DIRECTIONS
    }

    for row in frame.iter_rows(named=True):
        reasons = (row["null_reason"] or "").split(";")
        if row["metric"] in directed:
            assert row["direction"] in {Direction.HIGHER.value, Direction.LOWER.value}
            assert f"direction:{NullReason.NOT_APPLICABLE}" not in reasons
        else:
            assert row["direction"] is None
            assert f"direction:{NullReason.NOT_APPLICABLE}" in reasons


def test_a_metric_without_a_unit_has_a_null_unit_with_its_reason() -> None:
    frame = metric_dictionary()
    unitless = {metric.name for metric in defined_metrics() if not metric.unit}

    assert unitless
    for row in frame.iter_rows(named=True):
        reasons = (row["null_reason"] or "").split(";")
        assert (row["unit"] is None) == (row["metric"] in unitless)
        assert (f"unit:{NullReason.NOT_APPLICABLE}" in reasons) == (
            row["metric"] in unitless
        )


def test_a_row_names_the_calculator_that_publishes_the_metric_at_its_scope() -> None:
    frame = metric_dictionary()
    producers = scope_producers(shipped_calculators())

    for row in frame.iter_rows(named=True):
        assert row["producer"] == producers[(row["metric"], MetricScope(row["scope"]))]


def test_the_dictionary_passes_the_dimension_contract() -> None:
    dimensions = {
        name: pl.DataFrame(schema=table.polars_schema())
        for name, table in DIMENSION_TABLES.items()
    }
    dimensions[DimMetrics.table_name] = metric_dictionary()

    enforce_dimension_contract(dimensions)


def test_the_dictionary_describes_metrics_and_holds_no_metric_value() -> None:
    values = set(DimMetrics.polars_schema()) & {
        metric.name for metric in defined_metrics()
    }

    assert values == set()


def test_two_calculators_giving_one_metric_different_directions_is_an_error() -> None:
    with pytest.raises(
        SchemaViolationError, match="planted_distance_m.*two directions"
    ):
        metric_dictionary((*shipped_calculators(), _HigherAtEpisode, _LowerAtRobot))


def test_two_calculators_giving_one_metric_the_same_direction_are_not_a_conflict() -> (
    None
):
    frame = metric_dictionary(
        (*shipped_calculators(), _HigherAtEpisode, _HigherAtRobot)
    )

    assert frame.height == len(_store_pairs())


def test_a_stored_metric_that_no_calculator_publishes_is_an_error() -> None:
    with pytest.raises(SchemaViolationError, match="no calculator publishes"):
        metric_dictionary(())


def test_two_calculators_publishing_one_metric_at_one_scope_is_an_error() -> None:
    with pytest.raises(SchemaViolationError, match="produced by both"):
        scope_producers((_HigherAtRobot, _LowerAtRobot))


def test_dim_metrics_is_a_dimension_at_the_benchmark_root_keyed_by_metric_and_scope() -> (
    None
):
    assert DimMetrics.table_kind is TableKind.DIMENSION
    assert DimMetrics.relative_path().as_posix() == "dim_metrics.parquet"
    assert DimMetrics.key == ("metric", "scope")
    assert DIMENSION_TABLES["dim_metrics"] is DimMetrics


def test_dim_metrics_schema_is_pinned() -> None:
    assert DimMetrics.polars_schema() == pl.Schema(
        {
            "metric": pl.String,
            "scope": pl.String,
            "fact_table": pl.String,
            "dtype": pl.String,
            "unit": pl.String,
            "description": pl.String,
            "direction": pl.String,
            "producer": pl.String,
            "null_reason": pl.String,
        }
    )
