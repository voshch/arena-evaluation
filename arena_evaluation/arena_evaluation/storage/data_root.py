from __future__ import annotations

import datetime
import logging
import os
import pathlib
from collections.abc import Callable, Mapping

import yaml
from ament_index_python.packages import get_package_share_directory

_log = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.yaml"
_CREATED_AT_KEY = "created_at"
_DATA_DIR_VARIABLE = "ARENA_DATA_DIR"
_PACKAGE_NAME = "arena_evaluation"


def benchmarks_root(
    *,
    environ: Mapping[str, str] = os.environ,
    share_directory: Callable[[str], str] = get_package_share_directory,
) -> pathlib.Path:
    """``$ARENA_DATA_DIR/benchmarks``, or the package data folder when the variable is unset or empty."""
    data_dir = environ.get(_DATA_DIR_VARIABLE)
    if data_dir:
        return pathlib.Path(data_dir) / "benchmarks"
    return pathlib.Path(share_directory(_PACKAGE_NAME)) / "data"


def _manifest_created_at(run_dir: pathlib.Path) -> str | None:
    """The run manifest's created_at as a sortable string, None for a folder that is not a run."""
    manifest_path = run_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        _log.warning("%s: unreadable manifest, not a run: %r", manifest_path, error)
        return None
    if not isinstance(payload, dict):
        return None
    created = payload.get(_CREATED_AT_KEY)
    if isinstance(created, datetime.datetime):
        aware = created if created.tzinfo is not None else created.replace(tzinfo=datetime.UTC)
        return aware.astimezone(datetime.UTC).isoformat()
    if isinstance(created, str):
        return created
    return None


def latest_benchmark(
    root: pathlib.Path | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    share_directory: Callable[[str], str] = get_package_share_directory,
) -> pathlib.Path | None:
    """The run folder with the newest manifest ``created_at``, addressed by its folder name."""
    resolved_root = root or benchmarks_root(environ=environ, share_directory=share_directory)
    if not resolved_root.is_dir():
        return None
    candidates = [(created_at, path.name, path) for path in resolved_root.iterdir() if path.is_dir() and (created_at := _manifest_created_at(path)) is not None]
    if not candidates:
        return None
    return max(candidates)[2]
