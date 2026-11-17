"""MCP Query Gateway and Evidence Store Tools (Layer 4).

Exposes OLAP SQL query execution with a capped inline preview, transient vs. pinned
evidence persistence, stored query declaration management, and evidence index querying.
"""

from __future__ import annotations

import difflib
import re
from typing import Annotated, Any, Final

import polars as pl
import polars.exceptions
import pydantic
from arena_mcp.core import ToolError

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog, QueryDeclaration
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine, validate_readonly_sql
from arena_evaluation.storage.config_layout import run_analysis_root
from arena_evaluation.storage.star_schema import has_star_schema

from .base import (
    OptionalRunId,
    RunId,
    append_audit_event,
    component,
    domain,
    run_path,
)

__all__ = [
    "evaluation_evidence",
    "evaluation_queries",
    "evaluation_query",
    "evaluation_query_save",
]

PREVIEW_ROWS: Final[int] = 200
_SHOWN_LIMIT: Final[int] = 20
# The engine, the validator and the declaration renderer all raise from this family.
_QUERY_FAILURES: Final[tuple[type[BaseException], ...]] = (
    ValueError,
    KeyError,
    OSError,
    RuntimeError,
    polars.exceptions.PolarsError,
)
_TABLE_MISS = re.compile(r"relation '([^']+)' was not found")
_COLUMN_MISS = re.compile(r'unable to find column "([^"]+)"')
_VALID_COLUMNS = re.compile(r"valid columns: \[(.*?)\]", re.DOTALL)
_FROM_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _closest(missing: str, candidates: tuple[str, ...] | list[str]) -> str:
    """A ``; closest: ...`` suffix naming the near-miss candidates for a missing name."""
    matches = difflib.get_close_matches(missing, candidates, n=5, cutoff=0.5)
    return f"; closest: {', '.join(matches)}" if matches else ""


def _shown(names: tuple[str, ...] | list[str]) -> str:
    ordered = sorted(names)
    return ", ".join(ordered[:_SHOWN_LIMIT]) + (
        " ..." if len(ordered) > _SHOWN_LIMIT else ""
    )


def _column_candidates(catalog: BenchmarkCatalog, message: str, sql: str) -> list[str]:
    """The columns the failed query could have named: the engine's own list, else the FROM table's, else every table's."""
    valid = _VALID_COLUMNS.search(message)
    if valid is not None:
        return [name.strip().strip('"') for name in valid.group(1).split(",")]
    source = _FROM_TABLE.search(sql)
    if source is not None and source.group(1) in catalog.table_names():
        return catalog.get_table(source.group(1)).collect_schema().names()
    return sorted(
        {
            name
            for table in catalog.table_names()
            for name in catalog.get_table(table).collect_schema().names()
        }
    )


def _query_error(exc: Exception, catalog: BenchmarkCatalog, sql: str) -> ToolError:
    """A ToolError naming the closest valid table or column when the SQL engine reports one."""
    message = str(exc)
    table = _TABLE_MISS.search(message)
    if table is not None:
        missing = table.group(1)
        return ToolError(
            f"unknown table {missing!r}{_closest(missing, catalog.table_names())}",
            hint=f"tables: {_shown(catalog.table_names())}; evaluation_summarize shows the episode summary columns",
        )
    column = _COLUMN_MISS.search(message)
    if column is not None:
        missing = column.group(1)
        return ToolError(
            f"unknown column {missing!r}{_closest(missing, _column_candidates(catalog, message, sql))}",
            hint="evaluation_summarize lists the episode summary columns; evaluation_slice previews one table's rows",
        )
    return ToolError(
        f"Query execution failed: {exc}",
        hint="evaluation_summarize lists the run's summary columns",
    )


def _render(
    decl: QueryDeclaration, params: dict[str, Any] | None, error: str, hint: str
) -> str:
    """Render one stored or ad-hoc declaration, mapping parameter failures to a hinted ToolError."""
    try:
        return decl.render_sql(params)
    except ValueError as exc:
        raise ToolError(f"{error}: {exc}", hint=hint) from exc


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_query(
    run_id: OptionalRunId = None,
    query: Annotated[
        str | None, pydantic.Field(description="Ad-hoc read-only SQL query to execute")
    ] = None,
    *,
    stored_query: Annotated[
        str | None,
        pydantic.Field(description="Name of stored query declaration to execute"),
    ] = None,
    params: Annotated[
        dict[str, Any] | None,
        pydantic.Field(description="Parameters for stored query or SQL placeholders"),
    ] = None,
    pin_evidence: Annotated[
        bool,
        pydantic.Field(
            description="Persist the full query result as content-addressed evidence"
        ),
    ] = False,
    purpose: Annotated[
        str,
        pydantic.Field(
            description="Hypothesis or purpose for pinning evidence, a non-empty purpose pins too"
        ),
    ] = "",
    limit: Annotated[
        int,
        pydantic.Field(
            description="rows of the inline preview, the full result stays in pinned evidence",
            ge=1,
        ),
    ] = PREVIEW_ROWS,
    save_as: Annotated[
        str | None,
        pydantic.Field(
            description="Optional derived table name to persist under analysis/derived/<name>.parquet"
        ),
    ] = None,
) -> dict[str, Any]:
    """Execute an ad-hoc or stored analytical SQL query and return a capped preview: row_count, returned_rows, truncated, columns and data; pin_evidence=true pins the full result."""
    if query is None and stored_query is None:
        raise ToolError(
            "Must provide either 'query' or 'stored_query'",
            hint="evaluation_queries lists stored queries",
        )
    if query is not None and stored_query is not None:
        raise ToolError(
            "Cannot provide both 'query' and 'stored_query'",
            hint="pass exactly one of query or stored_query; evaluation_queries lists stored queries",
        )

    benchmark_path = run_path(run_id)
    if not has_star_schema(benchmark_path):
        raise ToolError(
            f"run {benchmark_path.name!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )
    catalog = BenchmarkCatalog.open(benchmark_path)
    engine = PolarsSQLEngine(catalog)

    decl: QueryDeclaration | None
    if stored_query is not None:
        try:
            decl = QueryCatalog(run_dir=benchmark_path).get_query(stored_query)
        except KeyError as exc:
            raise ToolError(
                f"Stored query {stored_query!r} not found",
                hint="evaluation_queries lists available queries",
            ) from exc
        operation = f"stored_query:{stored_query}"
        render_error = f"Invalid parameters for {stored_query!r}"
        render_hint = "evaluation_queries shows the declaration's parameters and defaults"
    else:
        clean_query = query.strip() if query is not None else ""
        operation = f"sql:{clean_query}"
        render_error = "Invalid query parameters"
        render_hint = (
            "pass :placeholder names as params, or drop params for SQL without placeholders"
        )
        if params:
            decl = QueryDeclaration(
                name="ad_hoc", description="", sql=clean_query, parameters=params
            )
        else:
            decl = None
            try:
                sql_to_run = validate_readonly_sql(clean_query)
            except _QUERY_FAILURES as exc:
                raise ToolError(
                    f"Invalid SQL query: {exc}",
                    hint="only one read-only SELECT or WITH ... SELECT statement is accepted",
                ) from exc

    if decl is not None:
        sql_to_run = _render(decl, params, render_error, render_hint)

    try:
        plan = engine.query_lazy(sql_to_run)
        row_count = plan.select(pl.len()).collect().item()
        preview = plan.head(limit).collect()
    except _QUERY_FAILURES as exc:
        raise _query_error(exc, catalog, sql_to_run) from exc

    returned_rows = preview.height
    truncated = row_count > returned_rows
    should_pin = (pin_evidence or bool(purpose.strip())) and row_count > 0
    frame: pl.DataFrame | None = None
    if should_pin:
        try:
            frame = plan.collect()
        except _QUERY_FAILURES as exc:
            raise _query_error(exc, catalog, sql_to_run) from exc

    derived_info: dict[str, Any] | None = None
    if save_as is not None:
        from arena_evaluation.analytics.derived import save_derived_table

        full_df = frame if frame is not None else plan.collect()
        try:
            derived_info = save_derived_table(
                benchmark_path,
                name=save_as,
                frame=full_df,
                definition={"kind": "sql", "sql": sql_to_run},
                snapshot_id=catalog.snapshot_id,
                purpose=purpose.strip() or save_as,
                author="analyst",
            )
        except (ValueError, OSError) as exc:
            raise ToolError(
                f"cannot save derived table {save_as!r}: {exc}",
                hint="use a save_as of letters, digits, '_' or '-', not starting with '-'",
            ) from exc

    rec = (
        EvidenceStore(benchmark_path).record(
            frame,
            kind="sql",
            operation=operation,
            snapshot_id=catalog.snapshot_id,
            purpose=purpose.strip(),
            params=params or {},
            sql_text=sql_to_run,
        )
        if frame is not None
        else None
    )

    if derived_info is not None or rec is not None:
        produced = [
            *([derived_info["table_name"]] if derived_info is not None else []),
            *([rec.evidence_id] if rec is not None else []),
        ]
        append_audit_event(
            benchmark_path,
            role="analyst",
            tool="evaluation_query",
            args={
                "query": sql_to_run,
                "save_as": save_as,
                "pin_evidence": pin_evidence,
            },
            produced_ids=produced,
        )

    return {
        "run_id": benchmark_path.name,
        "snapshot_id": catalog.snapshot_id,
        "pin_status": "pinned" if rec is not None else "transient",
        "evidence_id": rec.evidence_id if rec is not None else None,
        "purpose": rec.purpose if rec is not None else "",
        "row_count": row_count,
        "returned_rows": returned_rows,
        "truncated": truncated,
        "limit": limit,
        "columns": list(preview.columns),
        "artifact_path": str(rec.artifact_path)
        if rec is not None and rec.artifact_path is not None
        else None,
        "data": preview.to_dicts(),
        **({"derived_table": derived_info["table_name"]} if derived_info else {}),
    }


@domain.tool(read_only=True)
def evaluation_queries(run_id: OptionalRunId = None) -> dict[str, Any]:
    """List the stored query declarations: the run's own first when a run is given, then the package store."""
    catalog = QueryCatalog(run_dir=run_path(run_id)) if run_id else QueryCatalog()
    return {"queries": [q.to_dict() for q in catalog.list_queries()]}


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_query_save(
    run_id: RunId,
    name: Annotated[
        str, pydantic.Field(description="Unique identifier for the query declaration")
    ],
    description: Annotated[
        str,
        pydantic.Field(
            description="Description of what this analytical query computes"
        ),
    ],
    sql: Annotated[
        str,
        pydantic.Field(
            description="Read-only SQL statement with optional :param placeholders"
        ),
    ],
    *,
    parameters: Annotated[
        dict[str, Any] | None,
        pydantic.Field(description="Parameter declarations with types and defaults"),
    ] = None,
    title: Annotated[
        str | None, pydantic.Field(description="Human-readable title")
    ] = None,
) -> dict[str, Any]:
    """Save a query declaration into the run's analysis queries; the package store is read only."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    decl = QueryDeclaration(
        name=safe_name,
        description=description.strip(),
        sql=sql.strip(),
        parameters=parameters or {},
        title=title or safe_name,
    )
    try:
        decl.render_sql(decl.default_parameters())
    except _QUERY_FAILURES as exc:
        raise ToolError(
            f"Query declaration validation failed: {exc}",
            hint="fix the SQL or the parameter declarations and save it again",
        ) from exc

    run_dir = run_path(run_id)
    try:
        saved = QueryCatalog(run_dir=run_dir).save_query(
            name=safe_name,
            description=decl.description,
            sql=decl.sql,
            parameters=decl.parameters,
            title=decl.title,
            directory=run_analysis_root(run_dir),
        )
    except OSError as exc:
        raise ToolError(
            f"cannot save query {safe_name!r} into run {run_dir.name!r}: {exc}",
            hint="the run's analysis folder must be a writable directory; evaluation_queries lists the stored queries",
        ) from exc
    return {
        "status": "saved",
        "name": saved.name,
        "title": saved.title,
        "description": saved.description,
    }


@domain.tool(read_only=True)
def evaluation_evidence(
    run_id: OptionalRunId = None,
    evidence_id: Annotated[
        str | None, pydantic.Field(description="Specific evidence_id to load data for")
    ] = None,
    purpose_filter: Annotated[
        str | None, pydantic.Field(description="Substring filter for evidence purpose")
    ] = None,
    limit: Annotated[
        int,
        pydantic.Field(description="most evidence rows to return", ge=1),
    ] = 200,
) -> dict[str, Any]:
    """Query evidence index or load specific pinned evidence DataFrame for a benchmark run."""
    benchmark_path = run_path(run_id)
    store = EvidenceStore(benchmark_path)

    if evidence_id is not None:
        clean_ev_id = component(evidence_id, "evidence_id")
        try:
            rec = store.get(clean_ev_id)
            frame = store.read_frame(clean_ev_id)
        except KeyError as exc:
            raise ToolError(
                f"Evidence {clean_ev_id!r} not found in run {benchmark_path.name!r}",
                hint="evaluation_evidence lists the run's pinned evidence ids",
            ) from exc
        preview = frame.head(limit)
        return {
            "run_id": benchmark_path.name,
            "snapshot_id": rec.snapshot_id,
            "evidence_id": rec.evidence_id,
            "kind": rec.kind,
            "operation": rec.operation,
            "purpose": rec.purpose,
            "row_count": frame.height,
            "returned_rows": preview.height,
            "truncated": frame.height > preview.height,
            "limit": limit,
            "columns": list(frame.columns),
            "created_at_ns": rec.created_at_ns,
            "artifact_path": str(rec.artifact_path)
            if rec.artifact_path is not None
            else None,
            "data": preview.to_dicts(),
        }

    index_df = store.load_index()
    if purpose_filter is not None and purpose_filter.strip():
        pat = purpose_filter.strip().lower()
        index_df = index_df.filter(
            pl.col("purpose").str.to_lowercase().str.contains(pat)
        )

    return {
        "run_id": benchmark_path.name,
        "snapshot_id": BenchmarkCatalog.open(benchmark_path).snapshot_id,
        "total_records": index_df.height,
        "records": index_df.to_dicts(),
    }
