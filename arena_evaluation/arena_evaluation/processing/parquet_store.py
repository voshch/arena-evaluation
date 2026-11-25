from __future__ import annotations

import logging
import pathlib

import polars as pl

from arena_evaluation.processing.mcap_reader import load_bundles
from arena_evaluation.storage.atomic import atomic_write_text
from arena_evaluation.storage.layout import NATIVE_LAYOUT_VERSION, registry_schema_hash
from arena_evaluation.storage.schemas import TopicBundle

_log = logging.getLogger(__name__)


class TopicParquetStore:
    """Caches one episode's flat telemetry tables, written by the MCAP reader, and loads them back as TopicBundles."""

    VERSION_FILE = ".cache_version"

    @staticmethod
    def _stamp() -> str:
        """The extraction cache stamp: the native layout version and the registry schema hash."""
        return f"{NATIVE_LAYOUT_VERSION}:{registry_schema_hash()}"

    @staticmethod
    def is_current(source_dir: pathlib.Path) -> bool:
        """True when source_dir holds a finished extraction of this layout and registry schema."""
        try:
            stamp = (source_dir / TopicParquetStore.VERSION_FILE).read_text().strip()
        except OSError:
            return False
        return stamp == TopicParquetStore._stamp()

    @staticmethod
    def mark_complete(source_dir: pathlib.Path) -> None:
        """Atomically stamp a finished extraction so later runs read it instead of extracting again."""
        atomic_write_text(
            source_dir / TopicParquetStore.VERSION_FILE,
            f"{TopicParquetStore._stamp()}\n",
        )

    @staticmethod
    def read(source_dir: pathlib.Path) -> dict[str, TopicBundle] | None:
        """TopicBundles over the flat tables of source_dir; None when it holds no robot or a table cannot be read."""
        if not source_dir.is_dir():
            return None
        try:
            return load_bundles(source_dir) or None
        except (OSError, pl.exceptions.PolarsError) as error:
            _log.warning(f"{source_dir}: cached telemetry unreadable, extracting again: {error}")
            return None
