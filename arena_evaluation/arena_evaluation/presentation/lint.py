"""Declaration lint for stored queries, plot recipes, and report manifests."""

from __future__ import annotations

import dataclasses
import pathlib

import polars as pl
import polars.exceptions
import pydantic
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY
from arena_evaluation.presentation.resolver import DataResolver, find_recipe_path
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest
from arena_evaluation.storage.config_layout import (
    CONFIG_KINDS,
    package_config_roots,
    run_analysis_root,
)
from arena_evaluation.storage.registry import TableKind, registered_tables

__all__ = ["LintReport", "lint_declarations", "lint_manifest_bindings"]


@dataclasses.dataclass(frozen=True, slots=True)
class LintReport:
    """Result of linting stored queries, plot recipes, and report manifests."""

    valid: bool
    problems: tuple[str, ...]
    checked_queries: int
    checked_plots: int
    checked_manifests: int
    bound_plots: int = 0
    unbound: tuple[str, ...] = ()

    @property
    def schema_valid(self) -> bool:
        """Whether every declaration parsed and named only known types, sources and channels."""
        return self.valid


def _collect_bundle_files(
    kind: str, extra_roots: tuple[pathlib.Path, ...]
) -> list[pathlib.Path]:
    spec = CONFIG_KINDS[kind]
    seen: dict[str, pathlib.Path] = {}
    for root in (*package_config_roots(), *extra_roots):
        folder = root / spec.folder
        if not folder.is_dir():
            continue
        for entry in sorted(folder.iterdir()):
            target = entry / spec.filename
            if entry.is_dir() and target.is_file():
                seen[entry.name] = target
    return [seen[name] for name in sorted(seen)]


def _validate_plot_spec(
    label: str,
    spec: PlotSpec,
    catalog: BenchmarkCatalog,
    extra_recipe_dirs: tuple[pathlib.Path, ...],
) -> list[str]:
    problems: list[str] = []
    if spec.type and spec.type not in RENDERER_REGISTRY:
        problems.append(f"{label}: unknown plot type {spec.type!r}")
    if spec.uses:
        if find_recipe_path(spec.uses, extra_dirs=extra_recipe_dirs) is None:
            problems.append(f"{label}: referenced recipe {spec.uses!r} not found")
    if spec.data is not None and spec.data.source:
        if spec.data.source not in catalog.table_names():
            problems.append(f"{label}: unknown data source table {spec.data.source!r}")
        else:
            try:
                cols = set(catalog.get_table(spec.data.source).collect_schema().names())
            except (polars.exceptions.PolarsError, KeyError, ValueError) as exc:
                problems.append(f"{label}: cannot resolve {spec.data.source!r} ({exc})")
                return problems
            for channel, field in spec.channels.model_dump(exclude_none=True).items():
                if isinstance(field, str) and field not in cols and not spec.data.query:
                    problems.append(
                        f"{label}: channel {channel}={field!r} missing from {spec.data.source!r}"
                    )
    return problems


def _lint_catalog(benchmark_dir: pathlib.Path | None) -> BenchmarkCatalog:
    catalog = (
        BenchmarkCatalog.open(benchmark_dir)
        if benchmark_dir is not None
        else BenchmarkCatalog()
    )
    existing = set(catalog.table_names())
    for tbl in registered_tables():
        tname = (
            f"fact_{tbl.table_name}"
            if tbl.table_kind is TableKind.FACT
            and not tbl.table_name.startswith("fact_")
            else tbl.table_name
        )
        if tname not in existing:
            schema = dict(tbl.polars_schema())
            if tbl.table_kind is TableKind.TELEMETRY and "episode_id" not in schema:
                schema["episode_id"] = pl.Int64
            catalog.register_table(tname, pl.LazyFrame(schema=schema))
    catalog._register_series_views()  # agent-contracts: ignore AC002 - lazy view registration
    return catalog


def _dummy_params(parameters: object) -> dict[str, object]:
    if not isinstance(parameters, dict):
        return {}
    params: dict[str, object] = {}
    for pname, spec in parameters.items():
        if isinstance(spec, dict):
            if spec.get("default") is not None:
                params[str(pname)] = spec["default"]
            elif spec.get("type") in ("int", "integer"):
                params[str(pname)] = 0
            elif spec.get("type") in ("float", "number"):
                params[str(pname)] = 0.0
            elif spec.get("type") in ("bool", "boolean"):
                params[str(pname)] = False
            else:
                params[str(pname)] = ""
        else:
            params[str(pname)] = ""
    return params


def _binding_problems(
    name: str,
    manifest: VizManifest,
    catalog: BenchmarkCatalog,
    benchmark_dir: pathlib.Path,
) -> tuple[list[str], int]:
    """Resolve every manifest plot against the real catalog; return the unbound ones."""
    resolver = DataResolver(catalog=catalog, benchmark_dir=benchmark_dir)
    problems: list[str] = []
    bound = 0
    for plot in manifest.plots:
        try:
            spec = resolver.resolve_spec(plot)
            frame, diagnostics = resolver.resolve(
                spec, default_source=manifest.data_source
            )
        except Exception as exc:  # noqa: BLE001 - every resolver failure is a lint finding
            problems.append(f"report {name!r} plot {plot.id!r}: cannot bind ({exc})")
            continue
        if frame.height == 0 or diagnostics.status == "FAILED":
            detail = "; ".join(diagnostics.issues) or "resolved to 0 rows"
            problems.append(f"report {name!r} plot {plot.id!r}: unbound ({detail})")
            continue
        bound += 1
    return problems, bound


def lint_manifest_bindings(
    manifest: VizManifest, benchmark_dir: pathlib.Path
) -> tuple[str, ...]:
    """Resolve every plot of one manifest against a run; return the unbound-plot findings."""
    catalog = _lint_catalog(benchmark_dir)
    problems, _ = _binding_problems(
        manifest.name or "inline", manifest, catalog, benchmark_dir
    )
    return tuple(problems)


def lint_declarations(benchmark_dir: pathlib.Path | None = None) -> LintReport:
    """Validate stored queries, plot recipes and report manifests; bind them when a run is given."""
    catalog = _lint_catalog(benchmark_dir)
    engine = PolarsSQLEngine(catalog)
    extra_roots = (
        (run_analysis_root(benchmark_dir),) if benchmark_dir is not None else ()
    )
    extra_recipe_dirs = extra_roots
    problems: list[str] = []
    unbound: list[str] = []
    bound_plots = 0

    q_catalog = QueryCatalog(run_dir=benchmark_dir)
    queries = q_catalog.list_queries()
    for q in queries:
        if not q.sql.strip():
            problems.append(f"query {q.name!r}: empty SQL body")
            continue
        try:
            rendered_sql = q.render_sql(params=_dummy_params(dict(q.parameters)))
            engine.query(rendered_sql, limit=1)
        except (polars.exceptions.PolarsError, KeyError, ValueError, OSError) as exc:
            problems.append(f"query {q.name!r}: {exc}")

    plot_files = _collect_bundle_files("plot", extra_roots)
    for path in plot_files:
        name = path.parent.name
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            spec = PlotSpec.model_validate(raw)
        except (yaml.YAMLError, pydantic.ValidationError, ValueError, OSError) as exc:
            problems.append(f"plot {name!r}: invalid PlotSpec ({exc})")
            continue
        problems.extend(
            _validate_plot_spec(f"plot {name!r}", spec, catalog, extra_recipe_dirs)
        )

    manifest_files = _collect_bundle_files("report", extra_roots)
    for path in manifest_files:
        name = path.parent.name
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            manifest = VizManifest.model_validate(raw)
        except (yaml.YAMLError, pydantic.ValidationError, ValueError, OSError) as exc:
            problems.append(f"report {name!r}: invalid VizManifest ({exc})")
            continue
        for plot in manifest.plots:
            problems.extend(
                _validate_plot_spec(
                    f"report {name!r} plot {plot.id!r}",
                    plot,
                    catalog,
                    extra_recipe_dirs,
                )
            )
        if benchmark_dir is not None:
            binding_problems, bound = _binding_problems(
                name, manifest, catalog, benchmark_dir
            )
            unbound.extend(binding_problems)
            bound_plots += bound

    return LintReport(
        valid=not problems,
        problems=tuple(problems),
        checked_queries=len(queries),
        checked_plots=len(plot_files),
        checked_manifests=len(manifest_files),
        bound_plots=bound_plots,
        unbound=tuple(unbound),
    )
