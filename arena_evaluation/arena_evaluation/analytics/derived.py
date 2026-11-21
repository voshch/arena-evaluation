"""Derived tables stored under <run>/analysis/derived/ with sidecars and staleness tracking."""

from __future__ import annotations

import datetime
import pathlib
import re
import shutil
from typing import Any, Final

import polars as pl
import yaml

from arena_evaluation.storage import atomic
from arena_evaluation.storage.config_layout import run_analysis_root

DERIVED_DIR_NAME: Final[str] = "derived"
_SAFE_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_VERSIONED_PARQUET: Final[re.Pattern[str]] = re.compile(r"\.v\d+\.parquet$")


def derived_dir(benchmark_dir: pathlib.Path) -> pathlib.Path:
    """Return ``<run>/analysis/derived``."""
    return run_analysis_root(benchmark_dir) / DERIVED_DIR_NAME


def _table_paths(benchmark_dir: pathlib.Path, name: str) -> tuple[str, pathlib.Path, pathlib.Path, pathlib.Path]:
    clean = name.strip()
    if not _SAFE_NAME.match(clean):
        raise ValueError(f"Invalid derived table name {name!r}")
    root = derived_dir(benchmark_dir)
    return clean, root, root / f"{clean}.parquet", root / f"{clean}.yaml"


def _load_sidecar(path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Corrupt derived table sidecar {path}: {exc}") from exc
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Derived table sidecar {path} is not a YAML mapping; fix or remove {path}")


def _discovered(root: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    return sorted(root.glob(pattern)) if root.is_dir() else []


def scan_derived_tables(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan all active derived tables in ``benchmark_dir`` as ``derived_<name>``."""
    return {f"derived_{path.stem}": pl.scan_parquet(path) for path in _discovered(derived_dir(benchmark_dir), "*.parquet") if not _VERSIONED_PARQUET.search(path.name)}


def save_derived_table(
    benchmark_dir: pathlib.Path,
    *,
    name: str,
    frame: pl.DataFrame,
    definition: dict[str, object],
    snapshot_id: str,
    author: str,
    purpose: str = "",
) -> dict[str, Any]:
    """Persist a derived table and its YAML sidecar, archiving the version it replaces.

    The sidecar lock spans the read-archive-write, so each archived ``<name>.v<n>.parquet``
    is named by the version the replaced sidecar carried.
    """
    clean, root, parquet_path, sidecar_path = _table_paths(benchmark_dir, name)
    nested = [column for column, dtype in frame.schema.items() if isinstance(dtype, (pl.List, pl.Struct, pl.Array, pl.Object))]
    if nested:
        raise ValueError(f"Derived table {clean!r} must be flat; the nested column(s) {nested} cannot be persisted. Store one row per element instead (for example a seed column rather than a seed list).")
    author_clean = author.strip()
    if not author_clean:
        raise ValueError(f"Derived table {clean!r} requires a non-empty author")
    root.mkdir(parents=True, exist_ok=True)
    with atomic.file_lock(sidecar_path):
        version = 1
        if sidecar_path.is_file():
            version = int(_load_sidecar(sidecar_path).get("version", 1)) + 1
        if parquet_path.is_file():
            archived = root / f"{clean}.v{max(version - 1, 1)}.parquet"
            with atomic.staged_temp_path(archived) as tmp_archive:
                shutil.copy2(parquet_path, tmp_archive)
        with atomic.staged_temp_path(parquet_path) as tmp_parquet:
            frame.write_parquet(tmp_parquet)
        updated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        sidecar = {
            "name": clean,
            "table_name": f"derived_{clean}",
            "version": version,
            "snapshot_id": snapshot_id,
            "purpose": purpose,
            "author": author_clean,
            "updated_at": updated_at,
            "columns": [{"name": column, "dtype": str(dtype)} for column, dtype in frame.schema.items()],
            "n_rows": frame.height,
            "definition": definition,
        }
        atomic.atomic_write_text(sidecar_path, yaml.safe_dump(sidecar, sort_keys=False))
    return {
        **{key: sidecar[key] for key in ("name", "table_name", "version", "snapshot_id", "n_rows")},
        "path": str(parquet_path),
        "status": "fresh",
    }


def list_derived_tables(benchmark_dir: pathlib.Path, *, current_snapshot_id: str) -> list[dict[str, Any]]:
    """List all derived tables with ``fresh`` or ``stale`` status against ``current_snapshot_id``."""
    entries: list[dict[str, Any]] = []
    for sidecar_path in _discovered(derived_dir(benchmark_dir), "*.yaml"):
        raw = _load_sidecar(sidecar_path)
        stem = sidecar_path.stem
        snapshot_id = str(raw.get("snapshot_id") or "")
        entries.append(
            {
                "name": str(raw.get("name") or stem),
                "table_name": str(raw.get("table_name") or f"derived_{stem}"),
                "version": int(raw.get("version") or 1),
                "snapshot_id": snapshot_id,
                "status": "fresh" if snapshot_id == current_snapshot_id else "stale",
                "n_rows": int(raw.get("n_rows") or 0),
                "purpose": str(raw.get("purpose") or ""),
                "author": str(raw.get("author") or ""),
            }
        )
    return entries


def refresh_derived_table(benchmark_dir: pathlib.Path, name: str) -> dict[str, Any]:
    """Recompute a derived table from its stored definition against the current catalog."""
    from arena_evaluation.analytics.analyses import run_procedure
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.sql_engine import PolarsSQLEngine

    clean, _root, _parquet_path, sidecar_path = _table_paths(benchmark_dir, name)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Derived table {clean!r} not found")
    raw = _load_sidecar(sidecar_path)
    definition = raw.get("definition") or {}
    kind = str(definition.get("kind") or "")
    catalog = BenchmarkCatalog.open(benchmark_dir)
    summary_frame: pl.DataFrame | None = None
    match kind:
        case "sql":
            frame = PolarsSQLEngine(catalog).query(str(definition.get("sql") or ""))
        case "procedure" | "procedure_summary":
            procedure = str(definition.get("procedure") or "")
            result = run_procedure(procedure, catalog, definition.get("params") or {})
            frame = result.summary if kind == "procedure_summary" else result.frame
            summary_frame = result.summary if kind == "procedure" else None
        case _:
            raise ValueError(f"Cannot refresh derived table {clean!r} of kind {kind!r}")
    author_val = str(raw.get("author") or "").strip()
    if not author_val:
        raise ValueError(f"Derived table sidecar {sidecar_path} is missing 'author'")
    purpose = str(raw.get("purpose") or "")
    info = save_derived_table(
        benchmark_dir,
        name=clean,
        frame=frame,
        definition=definition,
        snapshot_id=catalog.snapshot_id,
        purpose=purpose,
        author=author_val,
    )
    summary_sidecar = derived_dir(benchmark_dir) / f"{clean}_summary.yaml"
    if summary_frame is not None and summary_sidecar.is_file():
        save_derived_table(
            benchmark_dir,
            name=f"{clean}_summary",
            frame=summary_frame,
            definition={**definition, "kind": "procedure_summary"},
            snapshot_id=catalog.snapshot_id,
            purpose=purpose,
            author=author_val,
        )
    return info
