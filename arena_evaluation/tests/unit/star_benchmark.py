"""Canonical Star Schema fixtures for experiment unit tests."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.storage.registry import FACT_TABLES, FactTable, dimensions
from arena_evaluation.storage.star_schema import write_benchmark_dimensions

__all__ = ["read_summary", "write_benchmark"]

_KEYS: dict[str, object] = {
    "run_id": "run",
    "episode_id": 0,
    "robot_id": "robot_0",
    "env_id": "env_0",
}


def _frame(rows: list[dict[str, object]], schema: pl.Schema) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(
        [{name: row.get(name) for name in schema.names()} for row in rows],
        schema=schema,
    )


def write_benchmark(
    directory: pathlib.Path,
    rows: list[dict[str, object]],
    *,
    robot_tracks: list[dict[str, object]] | None = None,
    contacts: list[dict[str, object]] | None = None,
) -> pathlib.Path:
    """Write Star Schema dimension tables at root and per-episode fact/telemetry parquet files."""
    resolved = [{**_KEYS, **row} for row in rows]
    directory.mkdir(parents=True, exist_ok=True)

    dims = {
        name: pl.DataFrame(schema=tbl.polars_schema())
        for name, tbl in dimensions.DIMENSION_TABLES.items()
    }
    dims["dim_episodes"] = _frame(resolved, dimensions.DimEpisodes.polars_schema())
    dims["dim_robots"] = _frame(resolved, dimensions.DimRobots.polars_schema())
    write_benchmark_dimensions(dims, directory)

    contact_rows = [{**_KEYS, **row} for row in (contacts or ())]

    for episode_id in sorted({int(row["episode_id"]) for row in resolved}):
        metrics_dir = directory / "episodes" / f"episode_{episode_id:03d}" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ep_rows = [row for row in resolved if int(row["episode_id"]) == episode_id]
        ep_contacts = [
            row for row in contact_rows if int(row["episode_id"]) == episode_id
        ]
        for fact_enum, fact_table in FACT_TABLES.items():
            table_rows = ep_contacts if fact_enum is FactTable.CONTACTS else ep_rows
            _frame(table_rows, fact_table.polars_schema()).write_parquet(
                metrics_dir / f"fact_{fact_enum.value}.parquet"
            )

    if robot_tracks is not None:
        for episode_id in sorted({int(row["episode_id"]) for row in resolved}):
            samples = [
                sample
                for sample in robot_tracks
                if int(sample["episode_id"]) == episode_id
            ]
            if not samples:
                continue
            path = (
                directory
                / "episodes"
                / f"episode_{episode_id:03d}"
                / "topics"
                / "robot_tracks.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                [
                    {
                        name: value
                        for name, value in sample.items()
                        if name != "episode_id"
                    }
                    for sample in samples
                ]
            ).write_parquet(path)

    return directory


def read_summary(directory: pathlib.Path) -> pl.DataFrame:
    """Return the canonical ``v_episode_summary`` view for a benchmark directory."""
    return BenchmarkCatalog.open(directory).get_table("v_episode_summary").collect()
