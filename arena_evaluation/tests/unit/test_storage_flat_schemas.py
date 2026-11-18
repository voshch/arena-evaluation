"""Every registered star-schema, series, and telemetry schema is flat (1NF)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    Table,
    TableKind,
    dimensions,
    registered_tables,
    series,
)
from arena_evaluation.storage.registry.analysis import EvidenceIndex

_NESTED_DTYPES = (pl.List, pl.Struct, pl.Object, pl.Array)


def _registered_schemas() -> list[tuple[str, pl.Schema]]:
    schemas: list[tuple[str, pl.Schema]] = [
        (f"dim:{name}", table.polars_schema())
        for name, table in dimensions.DIMENSION_TABLES.items()
    ]
    schemas.extend(
        (f"fact:{table.table_name}", table.polars_schema())
        for table in FACT_TABLES.values()
    )
    schemas.extend(
        (f"series:{name}", table.polars_schema())
        for name, table in series.SERIES_TABLES.items()
    )
    schemas.extend(
        (f"telemetry:{table.table_name}", table.polars_schema())
        for table in registered_tables()
        if table.table_kind is TableKind.TELEMETRY
    )
    schemas.extend(
        (f"analysis:{table.table_name}", table.polars_schema())
        for table in registered_tables()
        if table.table_kind is TableKind.ANALYSIS
    )
    return schemas


@pytest.mark.parametrize(("name", "schema"), _registered_schemas())
def test_a_registered_schema_has_no_nested_dtype(name: str, schema: pl.Schema) -> None:
    nested = {
        column: dtype
        for column, dtype in schema.items()
        if isinstance(dtype, _NESTED_DTYPES)
    }
    assert not nested, f"{name} has nested columns {nested}"


def test_registered_table_count_matches_registry() -> None:
    assert all(
        isinstance(table, type) and issubclass(table, Table)
        for table in registered_tables()
    )


def test_analysis_tables_live_under_analysis_without_null_reasons(
    tmp_path: pathlib.Path,
) -> None:
    analysis_tables = [
        table for table in registered_tables() if table.table_kind is TableKind.ANALYSIS
    ]
    assert analysis_tables
    for table in analysis_tables:
        assert table.relative_path().parts[0] == "analysis"
        assert not any(
            isinstance(dtype, pl.List) for dtype in table.polars_schema().values()
        )
        assert not any(col.null_reason for col in table._columns.values())

    store = EvidenceStore(benchmark_dir=tmp_path)
    rec = store.record(
        pl.DataFrame({"x": [1, 2]}),
        kind="sql",
        operation="smoke",
        snapshot_id="snap-000000000000",
    )
    assert (tmp_path / EvidenceIndex.relative_path()).is_file()
    scanned = EvidenceIndex.scan(tmp_path).collect()
    assert scanned.height == 1
    assert scanned.item(0, "evidence_id") == rec.evidence_id
