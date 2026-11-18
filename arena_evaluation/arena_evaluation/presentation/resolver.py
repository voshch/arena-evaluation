"""Data and recipe resolver for the declarative presentation engine."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

import polars as pl
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import run_computation
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.checks import DiagnosticInspector, PlotDiagnostics
from arena_evaluation.presentation.predicate import compile_where_predicate
from arena_evaluation.presentation.viz_manifest import PlotSpec
from arena_evaluation.storage.config_layout import (
    bundle_path,
    package_config_roots,
    run_analysis_root,
)

__all__ = [
    "DataResolver",
    "PlotDiagnostics",
    "deep_merge",
    "find_recipe_path",
    "load_recipe",
    "recipe_search_dirs",
]


def recipe_search_dirs() -> tuple[pathlib.Path, ...]:
    """Return the default package config roots for plot declaration bundles."""
    return tuple(package_config_roots())


def find_recipe_path(
    name: str, extra_dirs: Sequence[pathlib.Path] | None = None
) -> pathlib.Path | None:
    """Locate a plot bundle ``<root>/plots/<name>/plot.yaml`` in the configured roots."""
    stem = name.removesuffix(".yaml").removesuffix(".yml")
    for root in (*(extra_dirs or ()), *recipe_search_dirs()):
        if not root.is_dir():
            continue
        bundled = bundle_path(root, "plot", stem)
        if bundled.is_file():
            return bundled
    return None


def load_recipe(
    name: str, extra_dirs: Sequence[pathlib.Path] | None = None
) -> dict[str, object]:
    """Load a recipe dictionary from its YAML declaration."""
    path = find_recipe_path(name, extra_dirs=extra_dirs)
    if path is None:
        raise FileNotFoundError(f"Plot recipe '{name}' not found in search paths.")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"Recipe '{name}' must be a YAML mapping, got {type(data).__name__}."
        )
    return data


def deep_merge(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    """Recursively merge overrides into a base mapping without mutating either input."""
    result: dict[str, object] = dict(base)
    for key, value in overrides.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        elif value is not None:
            result[key] = value
    return result


class DataResolver:
    """Resolves PlotSpec data bindings against Star Schema, SQL queries, computations, or EvidenceStore."""

    def __init__(
        self,
        *,
        catalog: BenchmarkCatalog | None = None,
        sql_engine: PolarsSQLEngine | None = None,
        evidence_store: EvidenceStore | None = None,
        query_catalog: QueryCatalog | None = None,
        benchmark_dir: pathlib.Path | None = None,
        recipe_dirs: Sequence[pathlib.Path] | None = None,
    ) -> None:
        self.benchmark_dir = (
            pathlib.Path(benchmark_dir) if benchmark_dir is not None else None
        )
        self.recipe_dirs: list[pathlib.Path] = (
            list(recipe_dirs) if recipe_dirs is not None else []
        )
        if self.benchmark_dir is not None:
            self.recipe_dirs.append(run_analysis_root(self.benchmark_dir))

        self.catalog = (
            catalog
            if catalog is not None
            else (
                BenchmarkCatalog.open(self.benchmark_dir)
                if self.benchmark_dir is not None
                else None
            )
        )
        self.sql_engine = (
            sql_engine
            if sql_engine is not None
            else (PolarsSQLEngine(self.catalog) if self.catalog is not None else None)
        )
        self.evidence_store = (
            evidence_store
            if evidence_store is not None
            else (
                EvidenceStore(self.benchmark_dir)
                if self.benchmark_dir is not None
                else None
            )
        )
        self.query_catalog = (
            query_catalog
            if query_catalog is not None
            else QueryCatalog(run_dir=self.benchmark_dir)
        )

    def resolve_spec(self, spec: PlotSpec) -> PlotSpec:
        """Resolve ``uses: <recipe>`` reference and return the merged PlotSpec."""
        if not spec.uses:
            return spec

        recipe_raw = load_recipe(spec.uses, extra_dirs=self.recipe_dirs)
        user_overrides = spec.model_dump(exclude_unset=True)
        user_overrides.pop("uses", None)

        merged = deep_merge(recipe_raw, user_overrides)
        if not merged.get("id"):
            merged["id"] = spec.id or spec.uses
        return PlotSpec.model_validate(merged)

    def resolve(
        self,
        spec: PlotSpec,
        *,
        default_source: str | None = None,
    ) -> tuple[pl.DataFrame, PlotDiagnostics]:
        """Resolve data frame and technical diagnostics for a PlotSpec."""
        resolved_spec = self.resolve_spec(spec)
        df, evidence_id, snapshot_id = self._fetch_frame(
            resolved_spec,
            default_source=default_source,
        )
        df = self._apply_slice(resolved_spec, df)
        inspected = DiagnosticInspector.inspect(df, resolved_spec)
        diagnostics = PlotDiagnostics(
            status=inspected.status,
            issues=inspected.issues,
            row_count=inspected.row_count,
            evidence_id=evidence_id,
            snapshot_id=snapshot_id,
        )
        return df, diagnostics

    def _fetch_frame(
        self,
        spec: PlotSpec,
        *,
        default_source: str | None,
    ) -> tuple[pl.DataFrame, str, str]:
        data_spec = spec.data
        snapshot_id = self.catalog.snapshot_id if self.catalog is not None else ""
        evidence_id = ""

        if data_spec is not None and data_spec.computation:
            if self.catalog is None:
                raise RuntimeError(
                    f"Cannot run computation for plot '{spec.id}': catalog not initialized."
                )
            params = dict(data_spec.params or {})
            if default_source and "source" not in params:
                params["source"] = default_source
            result = run_computation(
                data_spec.computation,
                self.catalog,
                params,
                evidence_store=self.evidence_store,
                purpose=spec.id,
            )
            df = result.summary
            evidence_id = result.evidence_id
            snapshot_id = result.snapshot_id
        elif data_spec is not None and data_spec.stored_query:
            if self.sql_engine is None:
                raise RuntimeError(
                    f"Cannot execute stored query for plot '{spec.id}': SQL engine not initialized."
                )
            df = self.query_catalog.execute(
                data_spec.stored_query, self.sql_engine, params=data_spec.params
            )
        elif data_spec is not None and data_spec.query:
            if self.sql_engine is None:
                raise RuntimeError(
                    f"Cannot execute SQL query for plot '{spec.id}': SQL engine not initialized."
                )
            df = self.sql_engine.query(data_spec.query)
        elif data_spec is not None and data_spec.evidence_id:
            if self.evidence_store is None:
                raise RuntimeError(
                    f"Cannot fetch evidence '{data_spec.evidence_id}' for plot '{spec.id}': EvidenceStore not initialized."
                )
            df = self.evidence_store.read_frame(data_spec.evidence_id)
            evidence_id = data_spec.evidence_id
        else:
            table_source = (
                data_spec.source
                if (data_spec is not None and data_spec.source)
                else default_source
            )
            if not table_source:
                raise ValueError(
                    f"Cannot resolve data for plot '{spec.id}': no source, query, stored_query, computation, or evidence_id provided."
                )
            if self.catalog is None:
                raise RuntimeError(
                    f"Cannot fetch table '{table_source}' for plot '{spec.id}': catalog not initialized."
                )
            df = self.catalog.get_table(table_source).collect()

        return (
            self._apply_where_eager(df, data_spec.where if data_spec is not None else None),
            evidence_id,
            snapshot_id,
        )

    @staticmethod
    def _apply_where_eager(df: pl.DataFrame, where: str | None) -> pl.DataFrame:
        if not where:
            return df
        return df.lazy().filter(compile_where_predicate(where, df.schema)).collect()

    @staticmethod
    def _apply_slice(spec: PlotSpec, df: pl.DataFrame) -> pl.DataFrame:
        if spec.slice is None:
            return df

        res_df = df
        schema = res_df.schema
        for field_name, value in (
            ("episode_id", spec.slice.episode_id),
            ("encounter_id", spec.slice.encounter_id),
            ("contact_id", spec.slice.contact_id),
        ):
            if value is None:
                continue
            if field_name not in schema:
                raise ValueError(
                    f"Plot '{spec.id}': slice '{field_name}' requires missing column '{field_name}'."
                )
            res_df = res_df.filter(pl.col(field_name) == value)
        return res_df
