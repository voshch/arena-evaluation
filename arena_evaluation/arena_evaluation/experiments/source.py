"""Canonical Star Schema access shared by the study analyses."""

from __future__ import annotations

import functools
import pathlib

import polars as pl

from arena_evaluation.analytics.catalog import BenchmarkCatalog

__all__ = ["contacts", "episode_summary", "robot_tracks"]


@functools.lru_cache(maxsize=8)
def _open_catalog(resolved_dir: pathlib.Path) -> BenchmarkCatalog:
    """Open and process-cache one ``BenchmarkCatalog`` per resolved benchmark directory.

    Re-processing a benchmark in the same Python process will continue serving the cached catalog.
    """
    if not (resolved_dir / "dim_episodes.parquet").is_file():
        raise FileNotFoundError(f"no Star Schema tables under {resolved_dir}; run `arena evaluation process --benchmark-dir {resolved_dir}` first")
    return BenchmarkCatalog.open(resolved_dir)


def _catalog(benchmark_dir: pathlib.Path) -> BenchmarkCatalog:
    return _open_catalog(benchmark_dir.resolve())


def contacts(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """The run's canonical ``fact_contacts`` table, one row per contact window."""
    return _catalog(benchmark_dir).get_table("fact_contacts").collect()


def episode_summary(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """The run's canonical ``v_episode_summary`` view, one row per robot episode."""
    return _catalog(benchmark_dir).get_table("v_episode_summary").collect()


def robot_tracks(benchmark_dir: pathlib.Path) -> pl.LazyFrame:
    """The run's canonical ``robot_tracks`` telemetry table, one row per pose sample."""
    return _catalog(benchmark_dir).get_table("robot_tracks")
