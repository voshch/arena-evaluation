"""Streaming sink of the recorded telemetry: rows buffered per table, written as ZSTD Parquet sorted by t_ns."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any, Final

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from arena_evaluation.storage.registry import TableKind, registered_tables

COMPRESSION: Final[str] = "zstd"
FLUSH_ROW_LIMIT: Final[int] = 50_000
_ROBOT_KEY: Final[tuple[str, ...]] = ("env_id", "robot_id")
_ENVIRONMENT_KEY: Final[tuple[str, ...]] = ("env_id",)

_TELEMETRY_TABLES = tuple(table for table in registered_tables() if table.table_kind is TableKind.TELEMETRY)

#: The Arrow schema of every recorded table, derived from its registry class.
TABLE_SCHEMAS: Final[dict[str, pa.Schema]] = {table.table_name: table.arrow_schema() for table in _TELEMETRY_TABLES}
ROBOT_TABLES: Final[tuple[str, ...]] = tuple(table.table_name for table in _TELEMETRY_TABLES if table.key == _ROBOT_KEY)
ENV_TABLES: Final[tuple[str, ...]] = tuple(table.table_name for table in _TELEMETRY_TABLES if table.key == _ENVIRONMENT_KEY)
GLOBAL_TABLES: Final[tuple[str, ...]] = tuple(table.table_name for table in _TELEMETRY_TABLES if not table.key)
_ROW_KEYS: Final[dict[str, tuple[str, ...]]] = {table.table_name: table.row_key for table in _TELEMETRY_TABLES}


class TableSink:
    """Buffers flat rows per table and streams every table into one ZSTD Parquet file, sorted by t_ns."""

    def __init__(self, out_dir: pathlib.Path, *, flush_rows: int = FLUSH_ROW_LIMIT) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._flush_rows = flush_rows
        self._columns: dict[str, dict[str, list[Any]]] = {}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._last_stamp: dict[str, int] = {}
        self._unsorted: set[str] = set()
        self._key_rows: dict[str, dict[tuple[Any, ...], int]] = {}
        self._batch_count: dict[str, int] = {}
        self._pending_rows = 0

    def append(self, table: str, row: Mapping[str, Any]) -> None:
        columns = self._columns.setdefault(table, {name: [] for name in TABLE_SCHEMAS[table].names})
        stamp = row["t_ns"]
        previous = self._last_stamp.get(table)
        if previous is not None and stamp < previous:
            self._unsorted.add(table)
        self._last_stamp[table] = stamp
        key = _ROW_KEYS.get(table, ())
        if key:
            key_value = tuple(row[name] for name in key)
            index = self._key_rows.setdefault(table, {}).get(key_value)
            if index is not None:
                # The declared key is the contract: one row per key, the later record wins.
                for name, values in columns.items():
                    values[index] = row[name]
                return
        for name, values in columns.items():
            values.append(row[name])
        if key:
            self._key_rows[table][key_value] = len(columns["t_ns"]) - 1
        self._pending_rows += 1
        if self._pending_rows >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        for table, columns in self._columns.items():
            if columns["t_ns"]:
                batch = pa.RecordBatch.from_pydict(columns, schema=TABLE_SCHEMAS[table])
                self._writer(table).write_batch(batch)
                self._batch_count[table] = self._batch_count.get(table, 0) + 1
                for values in columns.values():
                    values.clear()
                self._key_rows.setdefault(table, {}).clear()
        self._pending_rows = 0

    def close(self) -> None:
        self.flush()
        for writer in self._writers.values():
            writer.close()
        for table in self._writers:
            key = _ROW_KEYS.get(table, ())
            crosses_batches = bool(key) and self._batch_count.get(table, 0) > 1
            if crosses_batches or table in self._unsorted:
                _finalize(
                    self._out_dir / f"{table}.parquet",
                    key=key,
                    sort=table in self._unsorted,
                )

    def _writer(self, table: str) -> pq.ParquetWriter:
        if table not in self._writers:
            self._writers[table] = pq.ParquetWriter(
                self._out_dir / f"{table}.parquet",
                TABLE_SCHEMAS[table],
                compression=COMPRESSION,
            )
        return self._writers[table]


def _finalize(path: pathlib.Path, *, key: tuple[str, ...], sort: bool) -> None:
    """Keep the later record of each declared key across all batches, then sort by t_ns when the rows came unsorted."""
    frame = pl.read_parquet(path)
    if key:
        frame = frame.unique(subset=list(key), keep="last", maintain_order=True)
    if sort:
        frame = frame.sort("t_ns", maintain_order=True)
    frame.write_parquet(path, compression=COMPRESSION)
