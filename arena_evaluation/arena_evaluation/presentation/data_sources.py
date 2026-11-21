"""Declared report data sources: Star Schema tables, canonical views and series views."""

from __future__ import annotations

import functools
from typing import Final

from arena_evaluation.analytics.catalog import CANONICAL_VIEW_NAMES, SERIES_TABLE_PREFIX
from arena_evaluation.storage.registry import FACT_TABLES, dimensions, registered_tables
from arena_evaluation.storage.registry.series import SERIES_TABLES
from arena_evaluation.storage.registry.table import TableKind

__all__ = [
    "EPISODE_SUMMARY_VIEW",
    "REJECTED_DATA_SOURCES",
    "declared_data_sources",
    "declared_series_stems",
    "declared_view_names",
    "series_view_name",
    "validate_data_source",
]

EPISODE_SUMMARY_VIEW: Final[str] = "v_episode_summary"
REJECTED_DATA_SOURCES: Final[frozenset[str]] = frozenset({"metrics", "episode_summary"})


def series_view_name(stem: str) -> str:
    """Return the catalog view name for a published 1NF series table."""
    return f"v_{SERIES_TABLE_PREFIX}{stem}"


@functools.lru_cache(maxsize=1)
def declared_series_stems() -> tuple[str, ...]:
    """Return stems of the 1NF series tables declared in the storage registry."""
    return tuple(sorted(SERIES_TABLES))


@functools.lru_cache(maxsize=1)
def declared_view_names() -> frozenset[str]:
    """Return every catalog view a manifest may name."""
    return frozenset(
        {
            *CANONICAL_VIEW_NAMES,
            *(series_view_name(stem) for stem in declared_series_stems()),
        }
    )


@functools.lru_cache(maxsize=1)
def declared_data_sources() -> frozenset[str]:
    """Return every table or view a manifest may declare."""
    telemetry_names = (tbl.table_name for tbl in registered_tables() if tbl.table_kind is TableKind.TELEMETRY)
    fact_names = (f"fact_{tbl.table_name}" for tbl in FACT_TABLES.values())
    series_names = (tbl.table_name for tbl in SERIES_TABLES.values())
    return frozenset(
        {
            *dimensions.DIMENSION_TABLES.keys(),
            *fact_names,
            *series_names,
            *telemetry_names,
            *declared_view_names(),
        }
    )


def validate_data_source(data_source: str) -> str:
    """Validate a manifest data source against declared tables and views."""
    if not data_source or not data_source.strip():
        raise ValueError("data_source must be a non-empty table or view name.")
    if data_source in REJECTED_DATA_SOURCES:
        raise ValueError(f"legacy data_source {data_source!r} is rejected; use {EPISODE_SUMMARY_VIEW!r} or a declared v_* view.")
    if data_source.startswith("v_") and data_source not in declared_view_names():
        available = ", ".join(sorted(declared_view_names()))
        raise ValueError(f"unknown data_source {data_source!r}: no declared catalog view. Declared views: {available}")
    if data_source.startswith("derived_"):
        return data_source
    if not data_source.startswith("v_") and data_source not in declared_data_sources():
        available = ", ".join(sorted(declared_data_sources()))
        raise ValueError(f"unknown data_source {data_source!r}: not a declared table or view. Declared: {available}")
    return data_source
