"""Canonical configuration and run-directory layout helpers."""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
from typing import Final

_log = logging.getLogger(__name__)

CONFIGS_DIR_NAME: Final[str] = "configs"
ANALYSIS_DIR_NAME: Final[str] = "analysis"
REPORTS_DIR_NAME: Final[str] = "reports"
EVIDENCE_DIR_NAME: Final[str] = "evidence"


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigKind:
    """Folder and entry filename for a single configuration kind."""

    name: str
    folder: str
    filename: str


CONFIG_KINDS: Final[dict[str, ConfigKind]] = {
    "suite": ConfigKind(name="suite", folder="suites", filename="suite.yaml"),
    "contest": ConfigKind(name="contest", folder="contests", filename="contest.yaml"),
    "report": ConfigKind(name="report", folder="reports", filename="report.yaml"),
    "plot": ConfigKind(name="plot", folder="plots", filename="plot.yaml"),
    "query": ConfigKind(name="query", folder="queries", filename="query.yaml"),
}


def get_config_kind(kind: str) -> ConfigKind:
    """Return the ConfigKind spec for a registered kind name."""
    if kind not in CONFIG_KINDS:
        valid = ", ".join(sorted(CONFIG_KINDS))
        raise KeyError(f"Unknown config kind {kind!r}. Valid kinds: {valid}.")
    return CONFIG_KINDS[kind]


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigAsset:
    """A single configuration file at a config root, unlike a bundled ConfigKind."""

    name: str
    filename: str


#: The flat configuration assets resolved through all_config_roots(). ``palette`` is the
#: asset that lives directly under ``configs/`` and is resolved through :func:`find_config_asset`.
CONFIG_ASSETS: Final[dict[str, ConfigAsset]] = {
    "palette": ConfigAsset(name="palette", filename="color_palette.yaml"),
}


def get_config_asset(kind: str) -> ConfigAsset:
    """Return the ConfigAsset spec for a registered flat config kind."""
    if kind not in CONFIG_ASSETS:
        valid = ", ".join(sorted(CONFIG_ASSETS))
        raise KeyError(f"Unknown config asset {kind!r}. Valid assets: {valid}.")
    return CONFIG_ASSETS[kind]


def share_dir() -> pathlib.Path | None:
    """Return the installed share directory of arena_evaluation, or None when it is not installed."""
    if not os.environ.get("AMENT_PREFIX_PATH"):
        return None
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return None
    try:
        return pathlib.Path(get_package_share_directory("arena_evaluation"))
    except PackageNotFoundError:
        return None


def source_tree_dir() -> pathlib.Path | None:
    """Return the arena_evaluation package root in a source checkout, or None."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / CONFIGS_DIR_NAME / CONFIG_KINDS["report"].folder).is_dir():
            return parent
    return None


def package_config_roots() -> list[pathlib.Path]:
    """Return deduplicated package config roots: the installed share directory, then the source tree."""
    installed = share_dir()
    source = source_tree_dir()
    if installed is None and source is not None:
        _log.warning(
            "arena_evaluation is not installed; reading package configs from the source tree at %s",
            source,
        )
    roots: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()
    for base in (installed, source):
        if base is None:
            continue
        cfg_root = (base / CONFIGS_DIR_NAME).resolve()
        if cfg_root.is_dir() and cfg_root not in seen:
            seen.add(cfg_root)
            roots.append(cfg_root)
    return roots


def all_config_roots(
    imported_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Return config roots in precedence order: imported root (if any), share dir, source tree."""
    roots: list[pathlib.Path] = []
    if imported_dir is not None:
        roots.append(imported_dir)
    for root in package_config_roots():
        if root not in roots:
            roots.append(root)
    return roots


def find_config_asset(kind: str) -> pathlib.Path | None:
    """Resolve a flat config asset through :func:`all_config_roots`, warning when no root carries it."""
    spec = get_config_asset(kind)
    roots = all_config_roots()
    for root in roots:
        candidate = root / spec.filename
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(root) for root in roots) or "(no config root)"
    _log.warning(
        "config asset %r (%s) is absent from every config root: %s; the consumer falls back to its built-in default",
        kind,
        spec.filename,
        searched,
    )
    return None


def run_analysis_root(run_dir: pathlib.Path) -> pathlib.Path:
    """Return the run-local declaration root ``<run>/analysis``."""
    return run_dir / ANALYSIS_DIR_NAME


def run_evidence_dir(run_dir: pathlib.Path) -> pathlib.Path:
    """Return the run-local evidence artifact directory ``<run>/analysis/evidence``."""
    return run_analysis_root(run_dir) / EVIDENCE_DIR_NAME


def bundle_path(root: pathlib.Path, kind: str, name: str) -> pathlib.Path:
    """Return ``<root>/<kind>s/<name>/<kind>.yaml`` for a config bundle."""
    spec = get_config_kind(kind)
    return root / spec.folder / name / spec.filename


def bundle_config_file(path: pathlib.Path, kind: str) -> pathlib.Path:
    """Return the yaml a config resolver returned: the file itself or the bundle directory's yaml."""
    if path.is_dir():
        return path / get_config_kind(kind).filename
    return path


def run_reports_root(output_root: pathlib.Path) -> pathlib.Path:
    """Return ``<output_root>/reports`` where rendered report directories live."""
    return output_root / REPORTS_DIR_NAME


def run_report_output_dir(output_root: pathlib.Path, report_name: str) -> pathlib.Path:
    """Return ``<output_root>/reports/<report_name>`` for rendered report artifacts."""
    return run_reports_root(output_root) / report_name


def find_rendered_reports(output_root: pathlib.Path) -> list[pathlib.Path]:
    """Return all rendered ``<output_root>/reports/<name>/report.html`` paths."""
    reports_dir = run_reports_root(output_root)
    if not reports_dir.is_dir():
        return []
    return sorted(entry / "report.html" for entry in reports_dir.iterdir() if entry.is_dir() and (entry / "report.html").is_file())


def find_rendered_plots(output_root: pathlib.Path) -> list[pathlib.Path]:
    """Return all rendered ``<output_root>/reports/<name>/plots/*.png`` paths."""
    reports_dir = run_reports_root(output_root)
    if not reports_dir.is_dir():
        return []
    return sorted(plot for entry in reports_dir.iterdir() if entry.is_dir() and (entry / "plots").is_dir() for plot in (entry / "plots").glob("*.png"))


def configs_readme_path() -> pathlib.Path | None:
    """Return the unified ``configs/README.md`` path if present."""
    for root in package_config_roots():
        cand = root / "README.md"
        if cand.is_file():
            return cand
    return None
