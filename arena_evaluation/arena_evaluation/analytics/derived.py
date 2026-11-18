"""Derived tables stored under <run>/analysis/derived/ with sidecars and staleness tracking."""

from __future__ import annotations

import datetime
import pathlib
import re
import shutil
from typing import Any, Final

import polars as pl
import yaml

from arena_evaluation.storage.atomic import (
    atomic_write_text,
    file_lock,
    staged_temp_path,
)

from arena_evaluation.storage.config_layout import run_analysis_root

DERIVED_DIR_NAME: Final[str] = "derived"
_SAFE_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_VERSIONED_PARQUET: Final[re.Pattern[str]] = re.compile(r"\.v\d+\.parquet$")


def derived_dir(benchmark_dir: pathlib.Path) -> pathlib.Path:
    """Return ``<run>/analysis/derived``."""
    return run_analysis_root(benchmark_dir) / DERIVED_DIR_NAME


def _validate_name(name: str) -> str:
    clean = name.strip()
    if not _SAFE_NAME.match(clean):
        raise ValueError(f"Invalid derived table name {name!r}")
    return clean


def scan_derived_tables(benchmark_dir: pathlib.Path) -> dict[str, pl.LazyFrame]:
    """Lazily scan all active derived tables in ``benchmark_dir`` as ``derived_<name>``."""
    root = derived_dir(benchmark_dir)
    if not root.is_dir():
        return {}
    tables: dict[str, pl.LazyFrame] = {}
    for path in sorted(root.glob("*.parquet")):
        if _VERSIONED_PARQUET.search(path.name):
            continue
        tables[f"derived_{path.stem}"] = pl.scan_parquet(path)
    return tables


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
    """Persist a derived table and its YAML sidecar, archiving version 1 on first replace."""
    clean = _validate_name(name)
    author_clean = author.strip()
    if not author_clean:
        raise ValueError(f"Derived table {clean!r} requires a non-empty author")
    root = derived_dir(benchmark_dir)
    root.mkdir(parents=True, exist_ok=True)
    parquet_path = root / f"{clean}.parquet"
    sidecar_path = root / f"{clean}.yaml"

    with file_lock(sidecar_path):
        version = 1
        if sidecar_path.is_file():
            try:
                existing = yaml.safe_load(sidecar_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Corrupt derived table sidecar {sidecar_path}: {exc}"
                ) from exc
            if not isinstance(existing, dict):
                raise ValueError(
                    f"Derived table sidecar {sidecar_path} is not a YAML mapping; fix or remove {sidecar_path}"
                )
            version = int(existing.get("version", 1)) + 1
        if parquet_path.is_file():
            v1_path = root / f"{clean}.v1.parquet"
            if not v1_path.is_file():
                with staged_temp_path(v1_path) as tmp_v1:
                    shutil.copy2(parquet_path, tmp_v1)

        with staged_temp_path(parquet_path) as tmp_parquet:
            frame.write_parquet(tmp_parquet)
        sidecar = {
            "name": clean,
            "table_name": f"derived_{clean}",
            "version": version,
            "snapshot_id": snapshot_id,
            "purpose": purpose,
            "author": author_clean,
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "columns": [
                {"name": col, "dtype": str(dtype)}
                for col, dtype in frame.schema.items()
            ],
            "n_rows": frame.height,
            "definition": definition,
        }
        atomic_write_text(sidecar_path, yaml.safe_dump(sidecar, sort_keys=False))
    return {
        "name": clean,
        "table_name": f"derived_{clean}",
        "version": version,
        "snapshot_id": snapshot_id,
        "n_rows": frame.height,
        "path": str(parquet_path),
        "status": "fresh",
    }


def list_derived_tables(
    benchmark_dir: pathlib.Path, *, current_snapshot_id: str
) -> list[dict[str, Any]]:
    """List all derived tables with ``fresh`` or ``stale`` status against ``current_snapshot_id``."""
    root = derived_dir(benchmark_dir)
    if not root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for sidecar_path in sorted(root.glob("*.yaml")):
        raw = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Derived table sidecar {sidecar_path} is not a YAML mapping; "
                f"fix or remove {sidecar_path}"
            )
        snap = str(raw.get("snapshot_id") or "")
        entries.append(
            {
                "name": str(raw.get("name") or sidecar_path.stem),
                "table_name": str(
                    raw.get("table_name") or f"derived_{sidecar_path.stem}"
                ),
                "version": int(raw.get("version") or 1),
                "snapshot_id": snap,
                "status": "fresh" if snap == current_snapshot_id else "stale",
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

    clean = _validate_name(name)
    sidecar_path = derived_dir(benchmark_dir) / f"{clean}.yaml"
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Derived table {clean!r} not found")
    raw = yaml.safe_load(sidecar_path.read_text(encoding="utf-8")) or {}
    definition = raw.get("definition") or {}
    kind = str(definition.get("kind") or "")
    catalog = BenchmarkCatalog.open(benchmark_dir)

    summary_frame: pl.DataFrame | None = None
    if kind == "sql":
        sql_text = str(definition.get("sql") or "")
        frame = PolarsSQLEngine(catalog).query(sql_text)
    elif kind in {"procedure", "procedure_summary"}:
        proc = str(definition.get("procedure") or "")
        params = definition.get("params") or {}
        result = run_procedure(proc, catalog, params)
        if kind == "procedure_summary":
            frame = result.summary
        else:
            frame = result.frame
            summary_frame = result.summary
    else:
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
