"""Deep declaration lint: resolve every plot binding and report a per-plot verdict.

The shallow lint in :mod:`arena_evaluation.presentation.lint` validates schemas and
table names. This module closes the gap that let 152 of 297 shipped plots report
``valid`` while their data binding could not produce a frame: it executes
``data.query`` and ``data.stored_query``, runs ``data.computation`` with its
``params.metrics``, and checks every channel against the resolved frame.

The resolver is built with ``evidence_store=None`` on purpose: declaration lint is a
read-only tool and must never write evidence artifacts. Plots bound to an
``evidence_id`` are reported as *not checked* instead, because reading them needs a
pinned artifact rather than a declaration.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Final

import polars as pl
import polars.exceptions
import pydantic
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest
from arena_evaluation.storage.config_layout import (
    CONFIG_KINDS,
    package_config_roots,
    run_analysis_root,
)
from arena_evaluation.storage.registry import TableKind, registered_tables

__all__ = ["DeepLintReport", "PlotVerdict", "deep_lint_declarations"]

_RESOLVE_ERRORS: Final[tuple[type[Exception], ...]] = (
    KeyError,
    TypeError,
    ValueError,
    OSError,
    RuntimeError,
    pydantic.ValidationError,
    polars.exceptions.PolarsError,
)

_SCALAR_CHANNELS: Final[tuple[str, ...]] = (
    "x",
    "y",
    "color",
    "facet_col",
    "facet_row",
    "size",
    "symbol",
)


@dataclasses.dataclass(frozen=True, slots=True)
class PlotVerdict:
    """One plot declaration's deep binding verdict."""

    name: str
    passed: bool
    checked: bool
    reason: str
    row_count: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class DeepLintReport:
    """Deep binding results for every plot and manifest plot declaration."""

    valid: bool
    plots: tuple[PlotVerdict, ...]

    @property
    def problems(self) -> tuple[str, ...]:
        """Failed verdicts rendered as ``<label>: <reason>`` lines."""
        return tuple(
            f"{verdict.name}: {verdict.reason}"
            for verdict in self.plots
            if not verdict.passed
        )

    @property
    def checked(self) -> int:
        """Number of plots whose binding was actually resolved."""
        return sum(1 for verdict in self.plots if verdict.checked)


def _lint_catalog(benchmark_dir: pathlib.Path | None) -> BenchmarkCatalog:
    """Open the run catalog, or register every known table's schema when there is no run."""
    catalog = (
        BenchmarkCatalog.open(benchmark_dir)
        if benchmark_dir is not None
        else BenchmarkCatalog()
    )
    if benchmark_dir is not None:
        return catalog
    existing = set(catalog.table_names())
    for tbl in registered_tables():
        table_name = (
            f"fact_{tbl.table_name}"
            if tbl.table_kind is TableKind.FACT
            and not tbl.table_name.startswith("fact_")
            else tbl.table_name
        )
        if table_name not in existing:
            schema = dict(tbl.polars_schema())
            if tbl.table_kind is TableKind.TELEMETRY and "episode_id" not in schema:
                schema["episode_id"] = pl.Int64
            catalog.register_table(table_name, pl.LazyFrame(schema=schema))
    catalog._register_series_views()
    return catalog


def _bundle_files(
    kind: str, roots: tuple[pathlib.Path, ...]
) -> list[tuple[str, pathlib.Path]]:
    """Return ``(name, yaml_path)`` for every bundle of ``kind`` under ``roots``."""
    spec = CONFIG_KINDS[kind]
    seen: dict[str, pathlib.Path] = {}
    for root in roots:
        folder = root / spec.folder
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            candidate = entry / spec.filename
            if entry.is_dir() and candidate.is_file() and entry.name not in seen:
                seen[entry.name] = candidate
    return [(name, seen[name]) for name in sorted(seen)]


def _resolver(benchmark_dir: pathlib.Path | None) -> DataResolver:
    catalog = _lint_catalog(benchmark_dir)
    return DataResolver(
        catalog=catalog,
        sql_engine=PolarsSQLEngine(catalog),
        evidence_store=None,
        query_catalog=QueryCatalog(run_dir=benchmark_dir),
        benchmark_dir=None,
        recipe_dirs=(run_analysis_root(benchmark_dir),)
        if benchmark_dir is not None
        else (),
    )


def _channel_columns(spec: PlotSpec) -> list[tuple[str, str]]:
    """Return ``(channel, column)`` pairs the renderer reads from the resolved frame."""
    pairs: list[tuple[str, str]] = []
    for channel, value in spec.channels.model_dump(exclude_none=True).items():
        if channel in ("split", "hierarchy"):
            continue
        if channel not in _SCALAR_CHANNELS:
            continue
        if isinstance(value, str) and value:
            pairs.append((channel, value))
        elif isinstance(value, list):
            pairs.extend((channel, str(item)) for item in value if item)
    return pairs


def _lint_plot(
    label: str,
    spec: PlotSpec,
    resolver: DataResolver,
) -> PlotVerdict:
    """Resolve one plot's data binding and channels into a verdict."""
    if spec.data is not None and spec.data.evidence_id:
        return PlotVerdict(
            name=label,
            passed=True,
            checked=False,
            reason="evidence binding not checked by declaration lint",
            row_count=None,
        )
    try:
        resolved = resolver.resolve_spec(spec)
    except _RESOLVE_ERRORS as exc:
        return PlotVerdict(
            name=label,
            passed=False,
            checked=True,
            reason=f"cannot resolve recipe: {exc}",
            row_count=None,
        )
    try:
        frame, _diagnostics = resolver.resolve(resolved)
    except _RESOLVE_ERRORS as exc:
        return PlotVerdict(
            name=label,
            passed=False,
            checked=True,
            reason=f"cannot bind data: {exc}",
            row_count=None,
        )

    missing = [
        f"{channel}={column!r}"
        for channel, column in _channel_columns(resolved)
        if column not in frame.columns
    ]
    if missing:
        return PlotVerdict(
            name=label,
            passed=False,
            checked=True,
            reason=(
                "channel column(s) missing from the resolved frame: "
                + ", ".join(missing)
            ),
            row_count=frame.height,
        )
    return PlotVerdict(
        name=label,
        passed=True,
        checked=True,
        reason="",
        row_count=frame.height,
    )


def _load_plot_spec(path: pathlib.Path) -> tuple[PlotSpec | None, str]:
    """Load and validate one plot declaration, returning the spec or the reason it failed."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"cannot read declaration: {exc}"
    try:
        return PlotSpec.model_validate(raw), ""
    except (pydantic.ValidationError, ValueError) as exc:
        return None, f"invalid PlotSpec: {exc}"


def _load_manifest(path: pathlib.Path) -> tuple[VizManifest | None, str]:
    """Load and validate one report manifest, returning the manifest or the reason it failed."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"cannot read manifest: {exc}"
    try:
        return VizManifest.model_validate(raw), ""
    except (pydantic.ValidationError, ValueError) as exc:
        return None, f"invalid VizManifest: {exc}"


def deep_lint_declarations(
    benchmark_dir: pathlib.Path | None = None,
    *,
    extra_roots: tuple[pathlib.Path, ...] = (),
) -> DeepLintReport:
    """Resolve every shipped and run-local plot and manifest plot; fail on any broken binding."""
    roots: tuple[pathlib.Path, ...] = (
        *package_config_roots(),
        *extra_roots,
        *((run_analysis_root(benchmark_dir),) if benchmark_dir is not None else ()),
    )
    resolver = _resolver(benchmark_dir)
    verdicts: list[PlotVerdict] = []

    for name, path in _bundle_files("plot", roots):
        spec, reason = _load_plot_spec(path)
        verdicts.append(
            PlotVerdict(
                name=f"plot {name!r}",
                passed=False,
                checked=True,
                reason=reason,
                row_count=None,
            )
            if spec is None
            else _lint_plot(f"plot {name!r}", spec, resolver)
        )

    for name, path in _bundle_files("report", roots):
        manifest, reason = _load_manifest(path)
        if manifest is None:
            verdicts.append(
                PlotVerdict(
                    name=f"report {name!r}",
                    passed=False,
                    checked=True,
                    reason=reason,
                    row_count=None,
                )
            )
            continue
        for plot in manifest.plots:
            verdicts.append(
                _lint_plot(f"report {name!r} plot {plot.id!r}", plot, resolver)
            )

    return DeepLintReport(
        valid=all(verdict.passed for verdict in verdicts),
        plots=tuple(verdicts),
    )
