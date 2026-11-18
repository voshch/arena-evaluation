"""The benchmark dimensions as stored: each registry table conformed, stacked across episodes, written once."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Mapping

import polars as pl

from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.table import Table


def empty_dimension(table: type[Table]) -> pl.DataFrame:
    """A frame without rows, with the table's columns and dtypes."""
    return pl.DataFrame(schema=table.polars_schema())


def conform(frame: pl.DataFrame, table: type[Table]) -> pl.DataFrame:
    """The frame with the table's columns in registry order and dtype; a column the frame lacks is NULL."""
    if frame.height == 0:
        return empty_dimension(table)
    columns = [
        pl.col(name).cast(dtype) if name in frame.columns else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in table.polars_schema().items()
    ]
    return frame.select(columns)


def stack(episodes: Iterable[Mapping[str, pl.DataFrame]]) -> dict[str, pl.DataFrame]:
    """Every dimension table with the rows of all episodes, in registry order."""
    parts: dict[str, list[pl.DataFrame]] = {name: [] for name in DIMENSION_TABLES}
    for episode in episodes:
        for name, table in DIMENSION_TABLES.items():
            if name in episode:
                parts[name].append(conform(episode[name], table))
    return {name: pl.concat(parts[name]) if parts[name] else empty_dimension(table) for name, table in DIMENSION_TABLES.items()}


def write_benchmark_dimensions(dimensions: Mapping[str, pl.DataFrame], benchmark_dir: pathlib.Path) -> None:
    """Write each dimension table once, at the benchmark root."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for name, table in DIMENSION_TABLES.items():
        dimensions[name].write_parquet(benchmark_dir / table.relative_path(), compression="zstd")

