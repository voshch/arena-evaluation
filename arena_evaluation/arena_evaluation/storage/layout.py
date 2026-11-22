"""The native benchmark layout: its version, its registry schema hash, its stamps and its legacy retirement."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections.abc import Callable
from typing import Final

import yaml

from arena_evaluation.processing.metrics.registry import load_metric_definitions
from arena_evaluation.storage.atomic import atomic_write_text
from arena_evaluation.storage.exceptions import StaleBenchmarkError
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.facts import FACT_TABLES
from arena_evaluation.storage.registry.table import Table, TableKind, registered_tables

__all__ = [
    "NATIVE_LAYOUT_VERSION",
    "check_benchmark_layout",
    "episode_is_complete",
    "registry_schema_hash",
    "retire_legacy_outputs",
    "write_benchmark_stamp",
    "write_episode_stamp",
]

NATIVE_LAYOUT_VERSION: Final[int] = 3  # 3: stored positions and recorded map-frame transforms are in the environment frame.

_BENCHMARK_STAMP_FILE: Final[str] = "benchmark_layout.yaml"
_EPISODE_STAMP_FILE: Final[str] = ".complete"
_LEGACY_DIR_NAME: Final[str] = "legacy"
_LEGACY_BENCHMARK_FILES: Final[tuple[str, ...]] = (
    "metrics.parquet",
    "combined_metrics.parquet",
)
_LAYOUT_HINT: Final[str] = "run the process command with --force-process on this benchmark (old outputs are moved to legacy/, not deleted)"

# This is the one accepted storage -> processing edge.
load_metric_definitions()


def _native_tables() -> tuple[type[Table], ...]:
    """Every table of the native layout: the facts, the dimensions and the recorded telemetry."""
    return (
        *FACT_TABLES.values(),
        *DIMENSION_TABLES.values(),
        *(table for table in registered_tables() if table.table_kind is TableKind.TELEMETRY),
    )


def registry_schema_hash() -> str:
    """The SHA-256 over every native table name and Polars schema, sorted by stored name."""
    entries = sorted((table.file_name().removesuffix(".parquet"), str(table.polars_schema())) for table in _native_tables())
    payload = "\n".join(f"{name}\0{schema}" for name, schema in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_benchmark_stamp(benchmark_dir: pathlib.Path) -> None:
    """Write the benchmark layout stamp in one rename."""
    atomic_write_text(
        benchmark_dir / _BENCHMARK_STAMP_FILE,
        yaml.safe_dump(
            {
                "layout_version": NATIVE_LAYOUT_VERSION,
                "schema_hash": registry_schema_hash(),
            },
            sort_keys=True,
        ),
    )


def write_episode_stamp(metrics_dir: pathlib.Path) -> None:
    """Write the episode completion stamp in one rename, creating the metrics directory."""
    atomic_write_text(
        metrics_dir / _EPISODE_STAMP_FILE,
        json.dumps(
            {
                "layout_version": NATIVE_LAYOUT_VERSION,
                "schema_hash": registry_schema_hash(),
            },
            sort_keys=True,
        ),
    )


def _stamp(
    path: pathlib.Path,
    load: Callable[[str], object],
    errors: tuple[type[Exception], ...],
) -> dict[str, object] | None:
    """The stamp payload, or None when the file is absent, undecodable or not a mapping."""
    try:
        payload = load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) + errors:
        return None
    return payload if isinstance(payload, dict) else None


def _is_current_stamp(payload: dict[str, object]) -> bool:
    """Whether a stamp payload carries the native layout version and the current schema hash."""
    return payload.get("layout_version") == NATIVE_LAYOUT_VERSION and payload.get("schema_hash") == registry_schema_hash()


def episode_is_complete(metrics_dir: pathlib.Path) -> bool:
    """Whether the episode metrics directory carries the current completion stamp; unreadable is incomplete."""
    payload = _stamp(metrics_dir / _EPISODE_STAMP_FILE, json.loads, (json.JSONDecodeError,))
    if payload is None:
        return False
    return _is_current_stamp(payload)


def _episode_directories(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """The ``episodes/episode_*`` folders of a benchmark, sorted."""
    episodes = benchmark_dir / "episodes"
    if not episodes.is_dir():
        return ()
    return tuple(sorted(path for path in episodes.glob("episode_*") if path.is_dir()))


def _episode_fact_files(
    episodes: tuple[pathlib.Path, ...],
) -> tuple[pathlib.Path, ...]:
    """Every stored per-episode fact file, sorted by episode and file name."""
    return tuple(path for episode in episodes for path in sorted((episode / "metrics").glob("fact_*.parquet")))


def _stale(benchmark_dir: pathlib.Path, reason: str) -> StaleBenchmarkError:
    """The refusal of a benchmark whose stored outputs predate the native layout."""
    return StaleBenchmarkError(f"benchmark {benchmark_dir} is stale: {reason}; {_LAYOUT_HINT}.")


def _benchmark_stamp_is_stale(benchmark_dir: pathlib.Path) -> bool:
    """Whether the benchmark stamp exists and does not carry the native layout version and schema hash."""
    stamp_path = benchmark_dir / _BENCHMARK_STAMP_FILE
    if not stamp_path.is_file():
        return False
    payload = _stamp(stamp_path, yaml.safe_load, (yaml.YAMLError,))
    return payload is None or not _is_current_stamp(payload)


def check_benchmark_layout(benchmark_dir: pathlib.Path) -> None:
    """Refuse stored outputs that predate the native layout, before any fact table is read."""
    root = pathlib.Path(benchmark_dir)
    legacy = tuple(root / name for name in _LEGACY_BENCHMARK_FILES if (root / name).is_file())
    if legacy:
        raise _stale(root, f"{legacy[0].name} predates the native layout")
    facts = _episode_fact_files(_episode_directories(root))
    incomplete = tuple(path for path in facts if not episode_is_complete(path.parent))
    if incomplete:
        raise _stale(root, f"{incomplete[0]} has no complete episode stamp")
    if not (root / _BENCHMARK_STAMP_FILE).is_file():
        if facts:
            raise _stale(root, f"{_BENCHMARK_STAMP_FILE} is absent")
        return
    if _benchmark_stamp_is_stale(root):
        raise _stale(root, f"{_BENCHMARK_STAMP_FILE} does not match the native layout")


def _free_sibling(path: pathlib.Path) -> pathlib.Path:
    """The first ``<name>.<n>`` sibling that does not exist."""
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def _move_to_legacy(benchmark_dir: pathlib.Path, source: pathlib.Path) -> pathlib.Path:
    """Move one file under ``legacy/`` at its relative path, keeping any resident file under a numeric suffix."""
    target = benchmark_dir / _LEGACY_DIR_NAME / source.relative_to(benchmark_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        os.replace(target, _free_sibling(target))
    os.replace(source, target)
    return target


def retire_legacy_outputs(benchmark_dir: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Move old benchmark, stale stamp and incomplete-episode outputs under ``legacy/``, never deleting one."""
    root = pathlib.Path(benchmark_dir)
    sources = [root / name for name in _LEGACY_BENCHMARK_FILES if (root / name).is_file()]
    if _benchmark_stamp_is_stale(root):
        sources.append(root / _BENCHMARK_STAMP_FILE)
    sources.extend(path for path in _episode_fact_files(_episode_directories(root)) if not episode_is_complete(path.parent))
    return tuple(_move_to_legacy(root, source) for source in sources)
