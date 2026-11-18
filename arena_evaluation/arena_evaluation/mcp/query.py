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
from arena_evaluation.storage.star_schema import has_star_schema

from .base import OptionalRunId, append_audit_event, component, domain, run_path

__all__ = [
    "evaluation_evidence",
    "evaluation_queries",
    "evaluation_query",
    "evaluation_query_save",
]

PREVIEW_ROWS: Final[int] = 200
_TABLE_MISS = re.compile(r"relation '([^']+)' was not found")
_COLUMN_MISS = re.compile(r'unable to find column "([^"]+)"')
_VALID_COLUMNS = re.compile(r"valid columns: \[(.*?)\]", re.DOTALL)
_FROM_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _closest(missing: str, candidates: tuple[str, ...] | list[str]) -> list[str]:
    return difflib.get_close_matches(missing, candidates, n=5, cutoff=0.5)


def _named(names: list[str]) -> str:
    return f"; closest: {', '.join(names)}" if names else ""


_SHOWN_LIMIT: Final[int] = 20


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
            f"unknown table {missing!r}{_named(_closest(missing, catalog.table_names()))}",
            hint=f"tables: {_shown(catalog.table_names())}; evaluation_read_metrics shows the episode summary columns",
        )
    column = _COLUMN_MISS.search(message)
    if column is not None:
        missing = column.group(1)
        return ToolError(
            f"unknown column {missing!r}{_named(_closest(missing, _column_candidates(catalog, message, sql)))}",
            hint="evaluation_read_metrics lists the episode summary columns; evaluation_slice previews one table's rows",
        )
    return ToolError(
        f"Query execution failed: {exc}",
        hint="evaluation_read_metrics lists the run's summary columns",
    )


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

    if stored_query is not None:
        q_catalog = QueryCatalog(run_dir=benchmark_path)
        try:
            decl = q_catalog.get_query(stored_query)
        except KeyError as exc:
            raise ToolError(
                f"Stored query {stored_query!r} not found",
                hint="evaluation_queries lists available queries",
            ) from exc
        try:
            sql_to_run = decl.render_sql(params)
        except ValueError as exc:
            raise ToolError(
                f"Invalid parameters for {stored_query!r}: {exc}",
                hint="evaluation_queries shows the declaration's parameters and defaults",
            ) from exc
        operation = f"stored_query:{stored_query}"
    else:
        clean_query = query.strip() if query is not None else ""
        if params:
            decl = QueryDeclaration(
                name="ad_hoc", description="", sql=clean_query, parameters=params
            )
            try:
                sql_to_run = decl.render_sql(params)
            except ValueError as exc:
                raise ToolError(
                    f"Invalid query parameters: {exc}",
                    hint="pass :placeholder names as params, or drop params for SQL without placeholders",
                ) from exc
        else:
            try:
                sql_to_run = validate_readonly_sql(clean_query)
            except (
                ValueError,
                KeyError,
                OSError,
                RuntimeError,
                polars.exceptions.PolarsError,
            ) as exc:
                raise ToolError(
                    f"Invalid SQL query: {exc}",
                    hint="only one read-only SELECT or WITH ... SELECT statement is accepted",
                ) from exc
        operation = f"sql:{clean_query}"

    try:
        plan = engine.query_lazy(sql_to_run)
        row_count = plan.select(pl.len()).collect().item()
        preview = plan.head(limit).collect()
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        polars.exceptions.PolarsError,
    ) as exc:
        raise _query_error(exc, catalog, sql_to_run) from exc

    returned_rows = preview.height
    truncated = row_count > returned_rows
    should_pin = (pin_evidence or bool(purpose.strip())) and row_count > 0
    if should_pin:
        try:
            frame = plan.collect()
        except (
            ValueError,
            KeyError,
            OSError,
            RuntimeError,
            polars.exceptions.PolarsError,
        ) as exc:
            raise _query_error(exc, catalog, sql_to_run) from exc
    derived_info: dict[str, Any] | None = None
    if save_as is not None:
        from arena_evaluation.analytics.derived import save_derived_table

        full_df = frame if should_pin else plan.collect()
        derived_info = save_derived_table(
            benchmark_path,
            name=save_as,
            frame=full_df,
            definition={"kind": "sql", "sql": sql_to_run},
            snapshot_id=catalog.snapshot_id,
            purpose=purpose.strip() or save_as,
            author="analyst",
        )

    rec = None
    if should_pin:
        rec = EvidenceStore(benchmark_path).record(
            frame,
            kind="sql",
            operation=operation,
            snapshot_id=catalog.snapshot_id,
            purpose=purpose.strip(),
            params=params or {},
            sql_text=sql_to_run,
        )

    if save_as is not None or rec is not None:
        produced: list[str] = []
        if derived_info is not None:
            produced.append(derived_info["table_name"])
        if rec is not None:
            produced.append(rec.evidence_id)
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

    if rec is not None:
        return {
            "run_id": benchmark_path.name,
            "snapshot_id": catalog.snapshot_id,
            "pin_status": "pinned",
            "evidence_id": rec.evidence_id,
            "purpose": rec.purpose,
            "row_count": row_count,
            "returned_rows": returned_rows,
            "truncated": truncated,
            "limit": limit,
            "columns": list(preview.columns),
            "artifact_path": str(rec.artifact_path)
            if rec.artifact_path is not None
            else None,
            "data": preview.to_dicts(),
            **({"derived_table": derived_info["table_name"]} if derived_info else {}),
        }

    return {
        "run_id": benchmark_path.name,
        "snapshot_id": catalog.snapshot_id,
        "pin_status": "transient",
        "evidence_id": None,
        "purpose": "",
        "row_count": row_count,
        "returned_rows": returned_rows,
        "truncated": truncated,
        "limit": limit,
        "columns": list(preview.columns),
        "artifact_path": None,
        "data": preview.to_dicts(),
        **({"derived_table": derived_info["table_name"]} if derived_info else {}),
    }


@domain.tool(read_only=True)
def evaluation_queries() -> list[dict[str, Any]]:
    """List all stored query declarations in the SQL Query Store."""
    q_catalog = QueryCatalog()
    return [q.to_dict() for q in q_catalog.list_queries()]


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_query_save(
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
    """Save a new reusable query declaration to the SQL Query Store."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    decl = QueryDeclaration(
        name=safe_name,
        description=description.strip(),
        sql=sql.strip(),
        parameters=parameters or {},
        title=title or safe_name,
    )
    try:
        _ = decl.render_sql(decl.default_parameters())
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        polars.exceptions.PolarsError,
    ) as exc:
        raise ToolError(
            f"Query declaration validation failed: {exc}",
            hint="fix the SQL or the parameter declarations and save it again",
        ) from exc

    q_catalog = QueryCatalog()
    saved = q_catalog.save_query(
        name=safe_name,
        description=decl.description,
        sql=decl.sql,
        parameters=decl.parameters,
        title=decl.title,
    )
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
) -> dict[str, Any]:
    """Query evidence index or load specific pinned evidence DataFrame for a benchmark run."""
    import polars as pl

    benchmark_path = run_path(run_id)
    store = EvidenceStore(benchmark_path)

    if evidence_id is not None:
        clean_ev_id = component(evidence_id, "evidence_id")
        try:
            rec = store.get(clean_ev_id)
            df = store.read_frame(clean_ev_id)
        except KeyError as exc:
            raise ToolError(
                f"Evidence {clean_ev_id!r} not found in run {benchmark_path.name!r}",
                hint="evaluation_evidence lists the run's pinned evidence ids",
            ) from exc
        return {
            "run_id": benchmark_path.name,
            "evidence_id": rec.evidence_id,
            "kind": rec.kind,
            "operation": rec.operation,
            "purpose": rec.purpose,
            "row_count": df.height,
            "columns": list(df.columns),
            "created_at_ns": rec.created_at_ns,
            "artifact_path": str(rec.artifact_path)
            if rec.artifact_path is not None
            else None,
            "data": df.to_dicts(),
        }

    index_df = store.load_index()
    if purpose_filter is not None and purpose_filter.strip():
        pat = purpose_filter.strip().lower()
        index_df = index_df.filter(
            pl.col("purpose").str.to_lowercase().str.contains(pat)
        )

    return {
        "run_id": benchmark_path.name,
        "total_records": index_df.height,
        "records": index_df.to_dicts(),
    }
