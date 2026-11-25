"""The benchmark star schema: conforming dimensions and lazy benchmark-wide scans from the registry."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Mapping
from typing import Final

import polars as pl

from arena_evaluation.storage.atomic import staged_temp_path
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.series import SERIES_TABLES
from arena_evaluation.storage.registry.table import Table, TableKind, registered_tables

__all__ = [
    "conform",
    "empty_dimension",
    "episode_directories",
    "episode_directory_id",
    "has_star_schema",
    "scan_dimensions",
    "scan_fact_views",
    "scan_series_tables",
    "scan_star_schema",
    "scan_telemetry_tables",
    "stack",
    "write_benchmark_dimensions",
]


def empty_dimension(table: type[Table]) -> pl.DataFrame:
    """A frame without rows, with the table's columns and dtypes."""
    return pl.DataFrame(schema=table.polars_schema())


def conform(frame: pl.DataFrame, table: type[Table]) -> pl.DataFrame:
    """The frame with the table's columns in registry order and dtype; a column the frame lacks is NULL."""
    if frame.height == 0:
        return empty_dimension(table)
    columns = [pl.col(name).cast(dtype) if name in frame.columns else pl.lit(None, dtype=dtype).alias(name) for name, dtype in table.polars_schema().items()]
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
    """Write each dimension table once, at the benchmark root, each in one rename."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for name, table in DIMENSION_TABLES.items():
        with staged_temp_path(benchmark_dir / table.relative_path()) as temporary:
            dimensions[name].write_parquet(temporary, compression="zstd")


def episode_directories(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Return sorted episode directories under ``benchmark_dir / 'episodes'``."""
    episodes_root = benchmark_dir / "episodes"
    if not episodes_root.is_dir():
        return ()
    return tuple(sorted(path for path in episodes_root.iterdir() if path.is_dir()))


def episode_directory_id(episode_dir: pathlib.Path) -> int | None:
    """Parse the numeric episode identifier from an ``episode_NNN`` directory name."""
    if not episode_dir.name.startswith("episode_"):
        return None
    suffix = episode_dir.name.removeprefix("episode_")
    return int(suffix) if suffix.isdigit() else None


MERGE_MANIFEST_FILENAME: Final[str] = "merge.yaml"


def has_star_schema(benchmark_dir: pathlib.Path) -> bool:
    """True when ``benchmark_dir`` has its Star Schema root dimension table or merge manifest."""
    return (benchmark_dir / DIMENSION_TABLES["dim_episodes"].relative_path()).is_file() or (benchmark_dir / MERGE_MANIFEST_FILENAME).is_file()


def scan_dimensions(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan every declared dimension table under ``benchmark_dir``."""
    return {name: table.scan(benchmark_dir) if (benchmark_dir / table.relative_path()).is_file() else pl.LazyFrame(schema=table.polars_schema()) for name, table in DIMENSION_TABLES.items()}


def scan_fact_views(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan every declared fact table across all episodes of ``benchmark_dir``."""
    ep_dirs = episode_directories(benchmark_dir)
    frames: dict[str, pl.LazyFrame] = {}
    for table in FACT_TABLES.values():
        fact_name = f"fact_{table.table_name}"
        paths = [ep_dir / table.relative_path() for ep_dir in ep_dirs if (ep_dir / table.relative_path()).is_file()]
        frames[fact_name] = pl.scan_parquet(paths) if paths else pl.LazyFrame(schema=table.polars_schema())
    return frames


def scan_star_schema(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan all declared dimension and fact tables of ``benchmark_dir``."""
    return {**scan_dimensions(benchmark_dir), **scan_fact_views(benchmark_dir)}


def scan_series_tables(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan every declared ``series_*`` table present in ``benchmark_dir``."""
    ep_dirs = episode_directories(benchmark_dir)
    frames: dict[str, pl.LazyFrame] = {}
    for table in SERIES_TABLES.values():
        paths = [ep_dir / table.relative_path() for ep_dir in ep_dirs if (ep_dir / table.relative_path()).is_file()]
        if paths:
            frames[table.table_name] = pl.scan_parquet(paths)
    return frames


def scan_telemetry_tables(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan every declared telemetry table present in ``benchmark_dir`` with ``episode_id``."""
    ep_dirs = episode_directories(benchmark_dir)
    frames: dict[str, pl.LazyFrame] = {}
    for table in registered_tables():
        if table.table_kind is not TableKind.TELEMETRY:
            continue
        episode_frames: list[pl.LazyFrame] = []
        for ep_dir in ep_dirs:
            path = ep_dir / table.relative_path()
            if not path.is_file():
                continue
            ep_id = episode_directory_id(ep_dir)
            episode_frames.append(table.scan(ep_dir).with_columns(episode_id=pl.lit(ep_id, dtype=pl.Int64)))
        if episode_frames:
            frames[table.table_name] = episode_frames[0] if len(episode_frames) == 1 else pl.concat(episode_frames, how="vertical_relaxed")
    return frames
