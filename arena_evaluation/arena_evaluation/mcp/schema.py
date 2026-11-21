"""Self-describing lake schema, multi-run merge, and declaration lint MCP tools."""

from __future__ import annotations

import json
import pathlib
from typing import Annotated, Any

import pydantic
from arena_mcp.core import ToolError

from .base import OptionalRunId, RunId, data_root, domain, run_path

_VIEW_DESCRIPTIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "v_episode_summary": (
        "episode",
        "Joined episode, robot, and crowd summary with scalar performance, energy, acoustic, and safety metrics.",
        ("run_id", "episode_id", "robot_id"),
    ),
    "v_planner_scorecard": (
        "planner",
        "Per-planner aggregated success, efficiency, energy, and safety scorecard.",
        ("world", "scenario", "planner", "robot_model"),
    ),
    "v_contacts": (
        "contact",
        "Contact events joined with closing velocities and attribution shares.",
        ("run_id", "episode_id", "robot_id", "contact_id"),
    ),
    "v_contact_summary": (
        "contact",
        "Per-planner contact attribution summary.",
        ("planner",),
    ),
    "v_encounters": (
        "encounter",
        "Pedestrian encounter events joined with episode context.",
        ("run_id", "episode_id", "robot_id", "encounter_id"),
    ),
    "v_pedestrian_summary": (
        "pedestrian",
        "Per-pedestrian summary joined with episode metadata.",
        ("run_id", "episode_id", "ped_id"),
    ),
}


def _schema_metadata(
    catalog: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Table and column metadata from storage.registry, overlaid with the catalog's dim_metrics."""
    from arena_evaluation.storage.registry import defined_metrics, registered_tables

    table_meta: dict[str, dict[str, Any]] = {}
    col_meta: dict[str, dict[str, str]] = {}

    for table in registered_tables():
        table_name = table.table_name
        catalog_name = f"fact_{table_name}" if table.table_kind.value == "fact" and not table_name.startswith("fact_") else table_name
        table_meta[catalog_name] = {
            "kind": table.table_kind.value,
            "scope": table.table_scope.value if table.table_scope is not None else table.table_kind.value,
            "keys": list(table.key),
            "meaning": (table.__doc__ or "").strip().splitlines()[0] if table.__doc__ else "",
        }
        for column_name in table.polars_schema().names():
            column = table.column(column_name)
            col_meta.setdefault(column_name, {"unit": column.unit, "meaning": column.description})

    for metric in defined_metrics():
        col_meta[metric.name] = {
            "unit": metric.unit,
            "meaning": metric.description,
        }

    if "dim_metrics" in catalog.table_names():
        for row in catalog.get_table("dim_metrics").collect().to_dicts():
            name = str(row.get("metric") or "")
            if not name:
                continue
            existing = col_meta.get(name, {})
            col_meta[name] = {
                "unit": str(row.get("unit") or "") or existing.get("unit", ""),
                "meaning": str(row.get("description") or "") or existing.get("meaning", ""),
                "direction": str(row.get("direction") or ""),
            }
    return table_meta, col_meta


def _table_metadata(name: str, table_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The registry metadata of ``name``, or its declared view/derived description."""
    known = table_meta.get(name)
    if known is not None:
        return known
    if name in _VIEW_DESCRIPTIONS:
        scope, meaning, keys = _VIEW_DESCRIPTIONS[name]
        return {
            "kind": "view",
            "scope": scope,
            "keys": list(keys),
            "meaning": meaning,
        }
    return {
        "kind": "view" if name.startswith("v_") else "derived",
        "scope": "derived",
        "keys": [],
        "meaning": "",
    }


def _open_catalog(run_id: str | None) -> tuple[pathlib.Path | None, Any]:
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.storage.star_schema import has_star_schema

    if run_id is None:
        return None, BenchmarkCatalog()
    path = run_path(run_id)
    if not has_star_schema(path):
        raise ToolError(
            f"run {run_id!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )
    return path, BenchmarkCatalog.open(path)


def _column_details(lf: Any, col_meta: dict[str, dict[str, str]], keys: list[str]) -> tuple[int, list[dict[str, Any]]]:
    import polars as pl

    schema = lf.collect_schema()
    names = schema.names()
    if not names:
        return 0, []

    exprs: list[pl.Expr] = [pl.len().alias("__row_count")]
    for name in names:
        dtype = schema[name]
        exprs.append(pl.col(name).null_count().alias(f"__null__{name}"))
        if dtype.is_numeric():
            exprs.append(pl.col(name).cast(pl.Float64).min().alias(f"__min__{name}"))
            exprs.append(pl.col(name).cast(pl.Float64).max().alias(f"__max__{name}"))
        elif dtype in (pl.String, pl.Boolean, pl.Categorical) or isinstance(dtype, pl.Enum):
            exprs.append(pl.col(name).drop_nulls().n_unique().alias(f"__uniq__{name}"))

    stats_row = lf.select(exprs).collect().row(0, named=True)
    row_count = int(stats_row["__row_count"])
    key_set = set(keys)

    columns: list[dict[str, Any]] = []
    for name in names:
        dtype = schema[name]
        meta = col_meta.get(name, {})
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(dtype),
            "unit": meta.get("unit", ""),
            "meaning": meta.get("meaning", ""),
            "direction": meta.get("direction", ""),
            "null_count": int(stats_row[f"__null__{name}"]),
            "is_join_key": name in key_set,
        }
        if f"__min__{name}" in stats_row:
            entry["min"] = stats_row[f"__min__{name}"]
            entry["max"] = stats_row[f"__max__{name}"]
        if f"__uniq__{name}" in stats_row:
            entry["n_unique"] = int(stats_row[f"__uniq__{name}"])
        columns.append(entry)
    return row_count, columns


@domain.tool(read_only=True)
def evaluation_schema(
    run_id: OptionalRunId = None,
    table: Annotated[
        str | None,
        pydantic.Field(description="Table or view name to inspect in detail; omit for overview"),
    ] = None,
) -> dict[str, Any]:
    """Describe every table in the lake, or one table's columns with units, meanings, null counts, and ranges."""
    import polars as pl

    path, catalog = _open_catalog(run_id)
    table_meta, col_meta = _schema_metadata(catalog)

    available = list(catalog.table_names())
    if table is not None:
        if table not in available:
            raise ToolError(
                f"unknown table {table!r}",
                hint=f"available tables: {', '.join(available)}",
            )
        tmeta = _table_metadata(table, table_meta)
        row_count, columns = _column_details(catalog.get_table(table), col_meta, tmeta["keys"])
        sample_cols = ", ".join(column["name"] for column in columns[:5]) or "*"
        return {
            "run_id": path.name if path is not None else None,
            "snapshot_id": catalog.snapshot_id,
            "table": table,
            "kind": tmeta["kind"],
            "scope": tmeta["scope"],
            "keys": tmeta["keys"],
            "join_keys": tmeta["keys"],
            "row_count": row_count,
            "meaning": tmeta["meaning"],
            "example_query": f"SELECT {sample_cols} FROM {table} LIMIT 10",
            "columns": columns,
        }

    tables_out: list[dict[str, Any]] = []
    for name in available:
        lf = catalog.get_table(name)
        try:
            row_count: int | None = int(lf.select(pl.len()).collect().item())
        except pl.exceptions.PolarsError:
            row_count = None
        tmeta = _table_metadata(name, table_meta)
        tables_out.append(
            {
                "name": name,
                "kind": tmeta["kind"],
                "scope": tmeta["scope"],
                "keys": tmeta["keys"],
                "row_count": row_count,
                "meaning": tmeta["meaning"],
            }
        )

    return {
        "run_id": path.name if path is not None else None,
        "snapshot_id": catalog.snapshot_id,
        "n_tables": len(tables_out),
        "tables": tables_out,
    }


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_merge_runs(
    run_ids: Annotated[list[str], pydantic.Field(description="Source benchmark run IDs to merge")],
    name: Annotated[str, pydantic.Field(description="Name for the merged benchmark folder")],
    labels: Annotated[
        dict[str, dict[str, str]] | None,
        pydantic.Field(description="Optional per-run label mapping, e.g. {'run_a': {'simulator': 'isaac'}}"),
    ] = None,
) -> dict[str, Any]:
    """Create a merged benchmark referencing source runs and per-source labels without copying episode files."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.merge import create_merged_benchmark

    benchmarks = data_root()
    try:
        merged_dir = create_merged_benchmark(benchmarks, run_ids=run_ids, name=name, labels=labels)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise ToolError(
            str(exc),
            hint="ensure source runs exist with Star Schema tables and target name is unused",
        ) from exc

    catalog = BenchmarkCatalog.open(merged_dir)
    return {
        "name": name,
        "path": str(merged_dir),
        "snapshot_id": catalog.snapshot_id,
        "sources": run_ids,
    }


@domain.tool(read_only=True)
def evaluation_lint_declarations(
    run_id: OptionalRunId = None,
    deep: Annotated[
        bool,
        pydantic.Field(description="resolve every plot's query, stored query, computation, and channels, not only its schema"),
    ] = False,
) -> dict[str, Any]:
    """Validate every stored query, plot recipe, and report manifest against the catalog and registry.

    With ``deep=True`` each plot's data binding is executed and every channel is resolved
    against the produced frame, so ``valid`` is false when a shipped or run-local plot
    cannot bind (the shallow check misses those). ``plots`` carries one verdict per
    declaration with the reason.
    """
    from arena_evaluation.presentation.lint import lint_declarations

    from .lint import deep_lint_declarations

    path = run_path(run_id) if run_id is not None else None
    report = lint_declarations(path)
    deep_report = deep_lint_declarations(path) if deep else None
    problems = [*report.problems, *(deep_report.problems if deep_report else ())]
    return {
        "run_id": path.name if path is not None else None,
        "valid": report.valid and (deep_report.valid if deep_report else True),
        "problems": problems,
        "checked_queries": report.checked_queries,
        "checked_plots": report.checked_plots,
        "checked_manifests": report.checked_manifests,
        "deep": deep,
        "deep_checked_plots": deep_report.checked if deep_report else 0,
        "plots": [
            {
                "name": verdict.name,
                "passed": verdict.passed,
                "checked": verdict.checked,
                "reason": verdict.reason,
                "row_count": verdict.row_count,
            }
            for verdict in (deep_report.plots if deep_report else ())
        ],
    }


@domain.resource("arena://evaluation/runs/{run_id}/schema")
def run_schema_resource(run_id: RunId) -> str:
    """Self-describing Star Schema overview for one benchmark run."""
    return json.dumps(evaluation_schema(run_id=run_id))
