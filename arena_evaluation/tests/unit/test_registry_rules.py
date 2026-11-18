"""The registry refuses each planted declaration defect when the table class is created, and derives its views."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    Column,
    FactTable,
    MetricScope,
    Table,
    TableKind,
)
from arena_evaluation.storage.registry.table import Metric, MetricStore


def test_a_duplicate_table_name_is_refused() -> None:
    with pytest.raises(SchemaViolationError, match="duplicate table name 'episodes'"):

        class DuplicateEpisodes(
            Table, name="episodes", kind=TableKind.FACT, scope=MetricScope.EPISODE
        ):
            run_id = Column(pl.String, description="Benchmark run of the episode.")


def test_a_misnamed_telemetry_table_is_refused() -> None:
    with pytest.raises(SchemaViolationError, match="misnamed table 'arena_tracks'"):

        class MisnamedTracks(Table, name="arena_tracks", kind=TableKind.TELEMETRY):
            t_ns = Column(pl.Int64, required=True)


def test_a_fact_table_must_carry_the_name_of_a_fact_table() -> None:
    with pytest.raises(SchemaViolationError, match="misnamed table 'robot_facts'"):

        class FactsByAnotherName(Table, name="robot_facts", kind=TableKind.FACT):
            run_id = Column(pl.String, description="Benchmark run of the row.")


def test_a_nested_dtype_is_refused() -> None:
    with pytest.raises(SchemaViolationError, match="nested dtype"):

        class NestedValues(Table, name="dim_nested_values", kind=TableKind.DIMENSION):
            values = Column(
                pl.List(pl.Float64), description="Values of one row.", unit="m"
            )


def test_a_float_column_without_a_unit_is_refused() -> None:
    with pytest.raises(SchemaViolationError, match="needs a unit"):

        class UnitlessSpacing(
            Table, name="dim_unitless_spacing", kind=TableKind.DIMENSION
        ):
            spacing = Column(pl.Float64, description="Spacing between two rows.")


def test_a_key_column_that_is_not_declared_is_refused() -> None:
    with pytest.raises(
        SchemaViolationError, match="key column 'robot_id' is not declared"
    ):

        class KeyedTelemetry(Table, name="robot_keyed", kind=TableKind.TELEMETRY):
            key = ("robot_id",)
            t_ns = Column(pl.Int64, required=True)


def test_a_telemetry_table_without_its_time_column_is_refused() -> None:
    with pytest.raises(SchemaViolationError, match="t_ns is not declared"):

        class TimelessTracks(Table, name="robot_timeless", kind=TableKind.TELEMETRY):
            speed_mps = Column(pl.Float64)


def test_a_column_declared_twice_in_a_hierarchy_is_refused() -> None:
    class RunColumns(Table, abstract=True):
        run_id = Column(pl.String, description="Benchmark run of the row.")

    with pytest.raises(SchemaViolationError, match="column 'run_id' is declared twice"):

        class RepeatedRunColumn(
            RunColumns, name="dim_repeated_run", kind=TableKind.DIMENSION
        ):
            run_id = Column(pl.String, description="Benchmark run of the row.")


def test_a_column_cannot_be_required_and_carry_a_null_reason() -> None:
    with pytest.raises(SchemaViolationError, match="cannot be required"):
        Column(
            pl.Float64,
            unit="m",
            description="Distance.",
            required=True,
            null_reason=True,
        )


def test_a_metric_is_not_declared_as_a_table_attribute() -> None:
    with pytest.raises(SchemaViolationError, match="defined with define_metric"):

        class PlantedMetricAttribute(
            Table, name="dim_planted_attribute", kind=TableKind.DIMENSION
        ):
            planted = Metric(
                pl.Float64,
                unit="m",
                description="Planted distance.",
                name="planted_distance_m",
                scopes=(MetricScope.EPISODE,),
            )


def test_a_metric_is_defined_once() -> None:
    store = MetricStore()
    store.define(
        "planted_twice",
        pl.Float64,
        scopes=(MetricScope.EPISODE,),
        unit="m",
        description="Planted distance.",
    )
    with pytest.raises(
        SchemaViolationError, match="metric 'planted_twice' is defined twice"
    ):
        store.define(
            "planted_twice",
            pl.Float64,
            scopes=(MetricScope.EPISODE,),
            unit="m",
            description="Planted distance.",
        )


def test_a_metric_lists_at_least_one_scope() -> None:
    store = MetricStore()
    with pytest.raises(SchemaViolationError, match="lists no scope"):
        store.define(
            "planted_scopeless_metric",
            pl.Float64,
            scopes=(),
            unit="m",
            description="Planted distance.",
        )
    assert store.all() == ()


def test_a_metric_lists_each_scope_once() -> None:
    with pytest.raises(SchemaViolationError, match="lists a scope twice"):
        MetricStore().define(
            "planted_repeated_scope",
            pl.Float64,
            scopes=(MetricScope.EPISODE, MetricScope.EPISODE),
            unit="m",
            description="Planted distance.",
        )


def test_a_float_metric_needs_a_unit() -> None:
    with pytest.raises(SchemaViolationError, match="needs a unit"):
        MetricStore().define(
            "planted_unitless_metric",
            pl.Float64,
            scopes=(MetricScope.EPISODE,),
            description="Planted distance.",
        )


def test_a_metric_cannot_take_the_name_of_a_column_its_table_declares() -> None:
    with pytest.raises(SchemaViolationError, match="already a declared column"):
        MetricStore().define(
            "result",
            pl.String,
            scopes=(MetricScope.EPISODE,),
            description="Planted verdict.",
            null_reason=True,
        )


def test_arrow_schema_declares_the_writer_types_and_nullability() -> None:
    schema = FACT_TABLES[FactTable.EPISODES].arrow_schema()
    assert schema.field("run_id").type == pa.string()
    assert schema.field("run_id").nullable is False
    assert schema.field("episode_id").type == pa.int64()
    assert schema.field("result").nullable is True
    assert schema.field("null_reason").nullable is True


def test_fact_files_live_in_the_metrics_folder_of_their_episode() -> None:
    assert (
        FACT_TABLES[FactTable.EPISODES].relative_path().as_posix()
        == "metrics/fact_episodes.parquet"
    )
