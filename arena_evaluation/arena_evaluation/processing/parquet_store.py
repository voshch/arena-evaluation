from __future__ import annotations

import logging
import pathlib

import polars as pl

from arena_evaluation.processing.mcap_reader import load_bundles
from arena_evaluation.storage.schemas import TopicBundle

_log = logging.getLogger(__name__)


class TopicParquetStore:
    """Caches one episode's flat telemetry tables, written by the MCAP reader, and loads them back as TopicBundles."""

    VERSION = 3
    VERSION_FILE = ".cache_version"

    @staticmethod
    def is_current(source_dir: pathlib.Path) -> bool:
        """True when source_dir holds a finished extraction written by this layout."""
        try:
            stamp = (source_dir / TopicParquetStore.VERSION_FILE).read_text().strip()
        except OSError:
            return False
        return stamp == str(TopicParquetStore.VERSION)

    @staticmethod
    def mark_complete(source_dir: pathlib.Path) -> None:
        """Stamp a finished extraction so later runs read it instead of extracting again."""
        (source_dir / TopicParquetStore.VERSION_FILE).write_text(
            f"{TopicParquetStore.VERSION}\n"
        )

    @staticmethod
    def read(source_dir: pathlib.Path) -> dict[str, TopicBundle] | None:
        """TopicBundles over the flat tables of source_dir; None when it holds no robot or a table cannot be read."""
        if not source_dir.is_dir():
            return None
        try:
            return load_bundles(source_dir, {}) or None
        except (OSError, pl.exceptions.PolarsError) as error:
            _log.warning(
                f"{source_dir}: cached telemetry unreadable, extracting again: {error}"
            )
            return None
