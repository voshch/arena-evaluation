"""Every registered star-schema, series, and telemetry schema is flat (1NF)."""

from __future__ import annotations


import polars as pl
import pytest

from arena_evaluation.storage.registry import (
    FACT_TABLES,
    TableKind,
    dimensions,
    registered_tables,
    series,
)

_NESTED_DTYPES = (pl.List, pl.Struct, pl.Object, pl.Array)


def _registered_schemas() -> list[tuple[str, pl.Schema]]:
    schemas: list[tuple[str, pl.Schema]] = [(f"dim:{name}", table.polars_schema()) for name, table in dimensions.DIMENSION_TABLES.items()]
    schemas.extend((f"fact:{table.table_name}", table.polars_schema()) for table in FACT_TABLES.values())
    schemas.extend((f"series:{name}", table.polars_schema()) for name, table in series.SERIES_TABLES.items())
    schemas.extend((f"telemetry:{table.table_name}", table.polars_schema()) for table in registered_tables() if table.table_kind is TableKind.TELEMETRY)
    schemas.extend((f"analysis:{table.table_name}", table.polars_schema()) for table in registered_tables() if table.table_kind is TableKind.ANALYSIS)
    return schemas


@pytest.mark.parametrize(("name", "schema"), _registered_schemas())
def test_a_registered_schema_has_no_nested_dtype(name: str, schema: pl.Schema) -> None:
    nested = {column: dtype for column, dtype in schema.items() if isinstance(dtype, _NESTED_DTYPES)}
    assert not nested, f"{name} has nested columns {nested}"
