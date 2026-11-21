"""Deep declaration lint: resolve every plot binding and report a per-plot verdict."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator
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
    "y2",
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

    @classmethod
    def failed(
        cls,
        name: str,
        reason: str,
        *,
        checked: bool = True,
        row_count: int | None = None,
    ) -> PlotVerdict:
        """A verdict for a binding that could not be produced."""
        return cls(name, False, checked, reason, row_count)

    @classmethod
    def ok(
        cls,
        name: str,
        row_count: int | None,
        *,
        checked: bool = True,
        reason: str = "",
    ) -> PlotVerdict:
        """A verdict for a binding that resolved to a frame."""
        return cls(name, True, checked, reason, row_count)


@dataclasses.dataclass(frozen=True, slots=True)
class DeepLintReport:
    """Deep binding results for every plot and manifest plot declaration."""

    valid: bool
    plots: tuple[PlotVerdict, ...]

    @property
    def problems(self) -> tuple[str, ...]:
        """Failed verdicts rendered as ``<label>: <reason>`` lines."""
        return tuple(f"{verdict.name}: {verdict.reason}" for verdict in self.plots if not verdict.passed)

    @property
    def checked(self) -> int:
        """Number of plots whose binding was actually resolved."""
        return sum(1 for verdict in self.plots if verdict.checked)


def _resolver(benchmark_dir: pathlib.Path | None) -> DataResolver:
    """Open the run catalog, or register every known table's schema when there is no run."""
    if benchmark_dir is not None:
        catalog = BenchmarkCatalog.open(benchmark_dir)
    else:
        catalog = BenchmarkCatalog()
        existing = set(catalog.table_names())
        for tbl in registered_tables():
            table_name = f"fact_{tbl.table_name}" if tbl.table_kind is TableKind.FACT and not tbl.table_name.startswith("fact_") else tbl.table_name
            if table_name in existing:
                continue
            schema = dict(tbl.polars_schema())
            if tbl.table_kind is TableKind.TELEMETRY and "episode_id" not in schema:
                schema["episode_id"] = pl.Int64
            catalog.register_table(table_name, pl.LazyFrame(schema=schema))
        catalog._register_series_views()
    return DataResolver(
        catalog=catalog,
        sql_engine=PolarsSQLEngine(catalog),
        evidence_store=None,
        query_catalog=QueryCatalog(run_dir=benchmark_dir),
        benchmark_dir=None,
        recipe_dirs=(run_analysis_root(benchmark_dir),) if benchmark_dir is not None else (),
    )


def _bundle_files(kind: str, roots: tuple[pathlib.Path, ...]) -> list[tuple[str, pathlib.Path]]:
    """Return ``(name, yaml_path)`` for every bundle of ``kind`` under ``roots``."""
    spec = CONFIG_KINDS[kind]
    seen: dict[str, pathlib.Path] = {}
    for root in roots:
        folder = root / spec.folder
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            candidate = entry / spec.filename
            if entry.is_dir() and candidate.is_file():
                seen.setdefault(entry.name, candidate)
    return sorted(seen.items())


def _channel_columns(spec: PlotSpec) -> list[tuple[str, str]]:
    """Return ``(channel, column)`` pairs the renderer reads from the resolved frame."""
    pairs: list[tuple[str, str]] = []
    for channel, value in spec.channels.model_dump(exclude_none=True).items():
        if channel not in _SCALAR_CHANNELS:
            continue
        columns = (value,) if isinstance(value, str) else value
        pairs.extend((channel, str(column)) for column in columns if column)
    return pairs


def _lint_plot(
    label: str,
    spec: PlotSpec,
    resolver: DataResolver,
) -> PlotVerdict:
    """Resolve one plot's data binding and channels into a verdict."""
    if spec.data is not None and spec.data.evidence_id:
        return PlotVerdict.ok(
            label,
            None,
            checked=False,
            reason="evidence binding not checked by declaration lint",
        )
    try:
        resolved = resolver.resolve_spec(spec)
    except _RESOLVE_ERRORS as exc:
        return PlotVerdict.failed(label, f"cannot resolve recipe: {exc}")
    try:
        frame, _diagnostics = resolver.resolve(resolved)
    except _RESOLVE_ERRORS as exc:
        return PlotVerdict.failed(label, f"cannot bind data: {exc}")

    missing = [f"{channel}={column!r}" for channel, column in _channel_columns(resolved) if column not in frame.columns]
    if missing:
        return PlotVerdict.failed(
            label,
            "channel column(s) missing from the resolved frame: " + ", ".join(missing),
            row_count=frame.height,
        )
    return PlotVerdict.ok(label, frame.height)


def _load_declaration(model: type[PlotSpec | VizManifest], path: pathlib.Path) -> tuple[PlotSpec | VizManifest | None, str]:
    """Load and validate one declaration, returning it or the reason it failed."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"cannot read declaration: {exc}"
    try:
        return model.model_validate(raw), ""
    except (pydantic.ValidationError, ValueError) as exc:
        return None, f"invalid {model.__name__}: {exc}"


def _declaration_verdicts(
    kind: str,
    roots: tuple[pathlib.Path, ...],
    resolver: DataResolver,
) -> Iterator[PlotVerdict]:
    """One verdict per plot declaration, or per plot inside a report manifest."""
    model = PlotSpec if kind == "plot" else VizManifest
    for name, path in _bundle_files(kind, roots):
        declaration, reason = _load_declaration(model, path)
        label = f"{kind} {name!r}"
        if declaration is None:
            yield PlotVerdict.failed(label, reason)
        elif isinstance(declaration, PlotSpec):
            yield _lint_plot(label, declaration, resolver)
        else:
            yield from (_lint_plot(f"{label} plot {plot.id!r}", plot, resolver) for plot in declaration.plots)


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
    verdicts = [
        *_declaration_verdicts("plot", roots, resolver),
        *_declaration_verdicts("report", roots, resolver),
    ]
    return DeepLintReport(
        valid=all(verdict.passed for verdict in verdicts),
        plots=tuple(verdicts),
    )
