"""The contract of the stored benchmark dimensions: registry schema, unique non-NULL keys, and a reason for every NULL."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from arena_evaluation.storage.exceptions import ContractViolationError
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.table import Table


def enforce_dimension_contract(dimensions: Mapping[str, pl.DataFrame]) -> None:
    """Raise ContractViolationError unless every dimension has the registry schema, unique non-NULL keys and a reason for each NULL."""
    for name, table in DIMENSION_TABLES.items():
        if name not in dimensions:
            raise ContractViolationError(f"{name}: the benchmark has no such dimension")
        frame = dimensions[name]
        expected = table.polars_schema()
        if frame.schema != expected:
            raise ContractViolationError(f"{name}: columns or dtypes differ from the registry: {dict(frame.schema)} != {dict(expected)}")
        _check_keys(name, frame, table)
        _check_null_reasons(name, frame, table)


def _check_keys(name: str, frame: pl.DataFrame, table: type[Table]) -> None:
    keys = list(table.key)
    if frame.height and frame.select(pl.any_horizontal(pl.col(keys).is_null()).any()).item():
        raise ContractViolationError(f"{name}: a key column is NULL")
    duplicated = frame.group_by(keys).len().filter(pl.col("len") > 1)
    if duplicated.height:
        raise ContractViolationError(f"{name}: {duplicated.height} duplicated key(s) on {keys}")


def _check_null_reasons(name: str, frame: pl.DataFrame, table: type[Table]) -> None:
    for column in table.null_reason_columns():
        reasoned = pl.col("null_reason").fill_null("").str.contains(column + ":", literal=True)
        unexplained = frame.filter(pl.col(column).is_null() & ~reasoned)
        if unexplained.height:
            raise ContractViolationError(f"{name}.{column}: {unexplained.height} NULL value(s) without a reason")
