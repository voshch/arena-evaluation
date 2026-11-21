from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

import pydantic

from arena_evaluation.analytics.computations import run_computation
from arena_evaluation.benchmark.sensitivity import rerun_sensitivity
from arena_mcp.core import ToolError
from arena_mcp.core.msgs import jsonable

from .base import (
    OptionalRunId,
    RunId,
    append_audit_event,
    data_root,
    domain,
    run_path,
)

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Mapping

    import polars as pl

    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.evidence import AnalysisResult

CONTESTANT = "planner"
_ROLLUP_RULE = "mean_over_episodes"
_DIRECTION_HINT = "ensure every metric has a declared direction ('higher' or 'lower') in dim_metrics"

_DEFAULT_METRICS: tuple[str, ...] = (
    "success",
    "time_to_goal",
    "path_efficiency",
    "energy_total_wh",
)
Metrics = Annotated[
    list[str] | None,
    pydantic.Field(description="metric columns, the metrics' declared primary outputs when omitted"),
]
References = Annotated[
    bool,
    pydantic.Field(description="keep reference episodes (is_reference), which are not contestant runs"),
]
Scalar = str | int | float | bool
Filter = Annotated[
    dict[str, Scalar | list[Scalar]] | None,
    pydantic.Field(description="equality filter, {column: value} or {column: [values]}"),
]


def _open_catalog(run_id: str) -> BenchmarkCatalog:
    """Open ``BenchmarkCatalog`` for ``run_id``, raising ToolError without Star Schema."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.storage.star_schema import has_star_schema

    path = run_path(run_id)
    if not has_star_schema(path):
        raise ToolError(
            f"run {run_id!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )
    return BenchmarkCatalog.open(path)


def _summary_frame(run_id: str) -> tuple[BenchmarkCatalog, pl.DataFrame]:
    """The catalog and its collected ``v_episode_summary`` for ``run_id``."""
    catalog = _open_catalog(run_id)
    return catalog, catalog.get_table("v_episode_summary").collect()


def _group_summary(
    catalog: BenchmarkCatalog,
    group_cols: tuple[str, ...],
    metrics: tuple[str, ...],
    *,
    filter: dict[str, Scalar | tuple[Scalar, ...]] | None = None,
    exclude_references: bool = False,
) -> AnalysisResult:
    """Layer 3 ``group_summary`` over ``metrics`` grouped by ``group_cols``."""
    params: dict[str, Any] = {
        "group_cols": group_cols,
        "metrics": metrics,
        "exclude_references": exclude_references,
    }
    if filter is not None:
        params["filter"] = filter
    return run_computation("group_summary", catalog, params)


def _computation_or_error(kind: str, catalog: BenchmarkCatalog, params: dict[str, Any], hint: str) -> AnalysisResult:
    """Run a Layer 3 computation, turning its ValueError into a hinted ToolError."""
    try:
        return run_computation(kind, catalog, params)
    except ValueError as exc:
        raise ToolError(str(exc), hint=hint) from exc


def _norm_filter(
    raw: dict[str, Scalar | list[Scalar]] | None,
) -> dict[str, Scalar | tuple[Scalar, ...]] | None:
    if not raw:
        return None
    return {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}


def load_metrics_with_snapshot(run_id: str) -> tuple[pl.DataFrame, str]:
    """Return ``(v_episode_summary, snapshot_id)``, raising ToolError without Star Schema."""
    catalog = _open_catalog(run_id)
    return catalog.get_table("v_episode_summary").collect(), catalog.snapshot_id


def episodes(frame: pl.DataFrame, include_references: bool) -> pl.DataFrame:
    import polars as pl

    if include_references:
        return frame
    return frame.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))


def columns(frame: pl.DataFrame, names: list[str], what: str) -> list[str]:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ToolError(
            f"no {what} column {', '.join(missing)}",
            hint=f"columns: {', '.join(frame.columns)}",
        )
    return names


def numeric_columns(frame: pl.DataFrame) -> list[str]:
    import polars as pl

    return [name for name, dtype in frame.schema.items() if dtype.is_numeric() or dtype == pl.Boolean or (isinstance(dtype, pl.List) and dtype.inner.is_numeric())]


def column_stats(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    import polars as pl

    stats: dict[str, dict[str, Any]] = {}
    for name, dtype in frame.schema.items():
        series = frame[name]
        entry: dict[str, Any] = {"dtype": str(dtype), "null_count": series.null_count()}
        if dtype.is_numeric():
            present = series.drop_nulls()
            if len(present):
                entry.update(
                    min=jsonable(float(cast("float", present.min()))),
                    max=jsonable(float(cast("float", present.max()))),
                    mean=jsonable(float(cast("float", present.mean()))),
                )
        elif isinstance(dtype, pl.List):
            entry["mean_length"] = jsonable(series.list.len().mean())
        elif dtype in (pl.String, pl.Boolean, pl.Categorical) or isinstance(dtype, pl.Enum):
            unique_s = series.drop_nulls().unique().sort()
            # agent-contracts: ignore JSON serialization boundary
            unique = unique_s.head(20).to_list()
            entry.update(n_unique=len(unique_s), unique_values=jsonable(unique[:20]))
        stats[name] = entry
    return stats


def _pivot_group_summary(summary_df: pl.DataFrame, group_cols: list[str], *, include_spread: bool = False) -> list[dict[str, Any]]:
    """Reshape long ``GroupSummaryAnalysis`` rows into one dict per group key."""
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary_df.to_dicts():
        key = tuple(row[col] for col in group_cols)
        entry = grouped.setdefault(key, {**{col: row[col] for col in group_cols}, "n_rows": int(row["n"])})
        metric = str(row["metric"])
        entry[metric] = jsonable(row["mean"])
        if include_spread:
            entry[f"{metric}_std"] = jsonable(row["std"])
        entry["n_rows"] = max(int(entry["n_rows"]), int(row["n"]))
    return list(grouped.values())


def _pivot_ranking(ranking_df: pl.DataFrame, *, include_norm: bool = True) -> list[dict[str, Any]]:
    """Reshape long ``CompositeRankingAnalysis`` rows into one ranked dict per planner."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in ranking_df.to_dicts():
        planner = str(row[CONTESTANT])
        entry = grouped.setdefault(
            planner,
            {
                CONTESTANT: planner,
                "rank": row["rank"],
                "composite_score": jsonable(row["composite_score"]),
            },
        )
        metric = str(row["metric"])
        entry[metric] = jsonable(row["raw_mean"])
        if include_norm:
            entry[f"{metric}_norm"] = jsonable(row["normalized_score"])
    return sorted(
        grouped.values(),
        key=lambda r: (
            r["rank"] is None,
            r["rank"] if r["rank"] is not None else 0,
            str(r[CONTESTANT]),
        ),
    )


def summary_of(catalog: BenchmarkCatalog, include_references: bool = False) -> dict[str, Any]:
    import polars as pl

    frame = catalog.get_table("v_episode_summary").collect()
    work = episodes(frame, include_references)
    exclude = not include_references
    planner_res = _group_summary(catalog, (CONTESTANT,), ("success",), exclude_references=exclude)
    contestants = [
        {
            "planner": row[CONTESTANT],
            "n_episodes": int(row["n"]),
            "success_rate": float(row["mean"]) if row["mean"] is not None else None,
        }
        for row in planner_res.frame.filter(pl.col(CONTESTANT).is_not_null()).sort(by=["mean", CONTESTANT], descending=[True, False], nulls_last=True).to_dicts()
    ]
    overall_res = _group_summary(catalog, (), ("success",), exclude_references=exclude)
    overall_val = overall_res.frame["mean"][0] if overall_res.frame.height else None
    distinct = {
        # agent-contracts: ignore AC004 - JSON MCP response boundary
        name: sorted(work[name].drop_nulls().unique().to_list())
        for name in (CONTESTANT, "stage", "world")
    }
    return {
        "n_rows": len(frame),
        "n_episodes": len(work),
        "n_references": len(frame) - len(work),
        "planners": distinct[CONTESTANT],
        "stages": distinct["stage"],
        "maps": distinct["world"],
        "overall_success_rate": float(overall_val) if overall_val is not None else None,
        "per_planner": contestants,
        "_snapshot_id": planner_res.snapshot_id,
    }


def _dim_metrics_rows(run_id: str | None) -> list[dict[str, Any]]:
    """Read ``dim_metrics`` rows from ``run_id`` or latest run when available."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.storage.data_root import latest_benchmark
    from arena_evaluation.storage.star_schema import has_star_schema

    target = run_path(run_id) if run_id is not None else latest_benchmark(data_root())
    if target is not None and has_star_schema(target):
        dm = BenchmarkCatalog.open(target).get_table("dim_metrics").collect()
        return dm.sort("metric").to_dicts()
    return []


@domain.tool(read_only=True)
def evaluation_metrics(run_id: OptionalRunId = None) -> dict[str, Any]:
    """Every declared metric from ``dim_metrics`` (when a run is available) or registered metric definitions."""
    from arena_evaluation.storage.registry import defined_metrics

    dim_rows = _dim_metrics_rows(run_id)
    if dim_rows:
        return {"run_id": run_id, "metrics": dim_rows}
    return {
        "metrics": [
            {
                "metric": m.name,
                "unit": m.unit,
                "direction": None,
                "description": m.description,
                "scopes": [s.value for s in m.scopes],
            }
            for m in sorted(defined_metrics(), key=lambda entry: entry.name)
        ]
    }


@domain.tool(read_only=True)
def evaluation_describe_metric(
    metric_name: str,
    run_id: OptionalRunId = None,
) -> dict[str, Any]:
    """One metric's unit, scopes, direction from ``dim_metrics``, and rollup rule."""
    from arena_evaluation.storage.registry import defined_metrics

    dim_rows = _dim_metrics_rows(run_id)
    matched = [r for r in dim_rows if r.get("metric") == metric_name]
    if matched:
        first = matched[0]
        direction = first.get("direction")
        return {
            "name": metric_name,
            "unit": first.get("unit"),
            "direction": direction,
            "lower_is_better": None if direction not in ("higher", "lower") else direction == "lower",
            "description": first.get("description"),
            "scopes": sorted({str(m["scope"]) for m in matched if m.get("scope")}),
            "fact_table": first.get("fact_table"),
            "producer": first.get("producer"),
            "rollup_rule": _ROLLUP_RULE,
        }

    metric_def = next((m for m in defined_metrics() if m.name == metric_name), None)
    if metric_def is not None:
        return {
            "name": metric_name,
            "unit": metric_def.unit,
            "direction": None,
            "lower_is_better": None,
            "description": metric_def.description,
            "scopes": [s.value for s in metric_def.scopes],
            "rollup_rule": _ROLLUP_RULE,
        }

    known = sorted({str(r["metric"]) for r in dim_rows} | {m.name for m in defined_metrics()})
    raise ToolError(
        f"no metric {metric_name!r}",
        hint=f"metrics: {', '.join(known)}" if known else "pass run_id with Star Schema tables",
    )


@domain.tool(read_only=True)
def evaluation_aggregate(
    run_id: RunId,
    dimension: Annotated[
        list[str] | str,
        pydantic.Field(description="column or columns to group by, e.g. planner or ['stage', 'planner']"),
    ] = "planner",
    metrics: Metrics = None,
    filter: Filter = None,
    sort_by: Annotated[str | None, pydantic.Field(description="metric column to sort by")] = None,
    descending: bool = True,
    include_references: References = False,
) -> dict[str, Any]:
    """Mean of metrics per combination of dimensions via Layer 3 ``group_summary``; one metric table, optional filter and sort."""
    import polars as pl

    catalog, frame = _summary_frame(run_id)
    dimensions = [dimension] if isinstance(dimension, str) else list(dimension)
    if not dimensions:
        raise ToolError(
            "no dimension to group by",
            hint="pass at least one column, e.g. planner or ['stage', 'planner']",
        )
    columns(frame, dimensions, "dimension")
    if filter:
        columns(frame, list(filter), "filter")
    chosen = columns(frame, metrics, "metric") if metrics else [name for name in numeric_columns(frame) if name not in dimensions and not isinstance(frame.schema[name], pl.List)]
    if sort_by is not None and sort_by not in chosen:
        raise ToolError(
            f"sort_by {sort_by!r} is not one of the metrics {chosen}",
            hint=f"sort_by must be one of {chosen}",
        )
    res = _group_summary(
        catalog,
        tuple(dimensions),
        tuple(chosen),
        filter=_norm_filter(filter),
        exclude_references=not include_references,
    )
    rows = _pivot_group_summary(res.frame, dimensions)
    if sort_by is not None:
        rows.sort(
            key=lambda row: (
                row[sort_by] is None,
                -(row[sort_by] or 0.0) if descending else (row[sort_by] or 0.0),
            )
        )
    return {
        "run_id": run_id,
        "snapshot_id": res.snapshot_id,
        "dimension": dimensions,
        "metrics": chosen,
        "n_rows": len(rows),
        "data": rows,
        "excluded_rows": dict(res.excluded_rows),
    }


@domain.tool(read_only=True)
def evaluation_compare_planners(
    run_id: RunId,
    planners: Annotated[
        list[str] | None,
        pydantic.Field(description="contestant names, all when omitted"),
    ] = None,
    metrics: Metrics = None,
    group_by_stage: bool = False,
    normalize: Annotated[
        bool,
        pydantic.Field(description="include <metric>_norm in [0, 1] and composite rank"),
    ] = True,
    include_references: References = False,
    n: Annotated[
        int | None,
        pydantic.Field(description="return only the n best contestants by weighted normalized score (the former evaluation_top_n)"),
    ] = None,
    weights: Annotated[
        list[float] | None,
        pydantic.Field(description="one weight per metric when n is set, equal weights when omitted"),
    ] = None,
) -> dict[str, Any]:
    """Contestants side by side on metric means and composite ranking via Layer 3 ``composite_ranking``; ``n`` returns the top contestants."""
    catalog, frame = _summary_frame(run_id)
    chosen = columns(frame, metrics or list(_DEFAULT_METRICS), "metric")
    row_filter = {"planner": tuple(planners)} if planners else None
    exclude = not include_references
    if n is not None:
        if weights is not None and len(weights) != len(chosen):
            raise ToolError(
                f"{len(weights)} weights for {len(chosen)} metrics",
                hint="give one weight per metric, in the same order",
            )
        weight_map = dict(zip(chosen, weights, strict=True)) if weights is not None else {metric: 1.0 for metric in chosen}
        top_res = _computation_or_error(
            "top_n_ranking",
            catalog,
            {
                "group_col": CONTESTANT,
                "metrics": tuple(chosen),
                "weights": weight_map,
                "top_n": n,
                "exclude_references": exclude,
            },
            f"{_DIRECTION_HINT}; evaluation_describe_metric checks one",
        )
        return {
            "run_id": run_id,
            "snapshot_id": top_res.snapshot_id,
            "group_by": [CONTESTANT],
            "metrics": chosen,
            "ranked_on": chosen,
            "weights": weight_map,
            "n": n,
            "rankings": _pivot_ranking(top_res.frame, include_norm=True),
        }

    res = _computation_or_error(
        "composite_ranking",
        catalog,
        {
            "group_col": CONTESTANT,
            "metrics": tuple(chosen),
            "filter": row_filter,
            "exclude_references": exclude,
        },
        f"{_DIRECTION_HINT}; evaluation_metrics lists them",
    )
    out: dict[str, Any] = {
        "run_id": run_id,
        "snapshot_id": res.snapshot_id,
        "group_by": ["stage", CONTESTANT] if group_by_stage else [CONTESTANT],
        "metrics": chosen,
        "ranked_on": chosen,
        "rankings": _pivot_ranking(res.frame, include_norm=normalize),
    }
    if group_by_stage:
        stage_res = _group_summary(
            catalog,
            ("stage", CONTESTANT),
            tuple(chosen),
            filter=row_filter,
            exclude_references=exclude,
        )
        out["by_stage"] = _pivot_group_summary(stage_res.frame, ["stage", CONTESTANT], include_spread=True)
    return out


@domain.tool(read_only=True)
def evaluation_correlation(run_id: RunId, metrics: list[str], include_references: References = False) -> dict[str, Any]:
    """Pairwise Pearson correlation matrix of scalar metrics via Layer 3 ``correlation_matrix``."""
    import polars as pl

    catalog, frame = _summary_frame(run_id)
    columns(frame, metrics, "metric")
    if len(metrics) < 2:
        raise ToolError(
            "correlation needs at least two metrics",
            hint="pass at least two metric names, e.g. ['time_to_goal', 'path_length_m']",
        )
    lists = [name for name in metrics if isinstance(frame.schema[name], pl.List)]
    if lists:
        raise ToolError(
            f"{', '.join(lists)} hold one list per episode, correlate scalar metrics",
            hint="evaluation_summarize shows each column's dtype",
        )

    res = run_computation(
        "correlation_matrix",
        catalog,
        {"metrics": tuple(metrics), "exclude_references": not include_references},
    )
    matrix: dict[str, dict[str, float | None]] = {m: {} for m in metrics}
    for row in res.frame.to_dicts():
        a, b = str(row["metric_x"]), str(row["metric_y"])
        r_val = jsonable(row["pearson_r"])
        matrix.setdefault(a, {})[b] = r_val
        matrix.setdefault(b, {})[a] = r_val
    return {
        "run_id": run_id,
        "snapshot_id": res.snapshot_id,
        "metrics": metrics,
        "correlation_matrix": matrix,
    }


@domain.tool(read_only=True)
def evaluation_summarize(run_id: RunId, include_references: References = False) -> dict[str, Any]:
    """A run's progress and results: step counts, contestants, stages and maps via Layer 3 ``group_summary``."""
    from .runs import run_status

    path = run_path(run_id)
    status = run_status(path)
    catalog, frame = _summary_frame(run_id)
    summary = summary_of(catalog, include_references)
    snapshot_id = str(summary.pop("_snapshot_id", catalog.snapshot_id))
    per_planner = summary["per_planner"]
    lines = [
        f"Benchmark {run_id} ({status['suite']} x {status['contest']}): {status['state']}, {status['steps']['ok']}/{status['steps_total']} steps ok",
        f"Contestants: {', '.join(map(str, summary['planners']))}",
        f"Stages: {', '.join(map(str, summary['stages']))}",
        f"Maps: {', '.join(map(str, summary['maps']))}",
        f"Episodes: {summary['n_episodes']} ({summary['n_references']} reference rows left out)" if not include_references else f"Episodes: {summary['n_episodes']} including references",
    ]
    lines += [f"  {entry['planner']}: {entry['n_episodes']} episodes, success " + ("n/a" if entry["success_rate"] is None else f"{entry['success_rate'] * 100:.1f}%") for entry in per_planner]
    return {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "summary": "\n".join(lines),
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "columns": frame.columns,
        "column_stats": column_stats(frame),
        **summary,
        "status": status,
    }


@domain.resource(
    "arena://evaluation/runs/{run_id}/metrics/schema",
    mime_type="application/json",
    description="Columns and dtypes of a run's canonical episode summary view",
)
def metrics_schema_resource(run_id: str) -> dict[str, Any]:
    frame, snapshot_id = load_metrics_with_snapshot(run_id)
    return {
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "n_rows": len(frame),
        "dtypes": {name: str(dtype) for name, dtype in frame.schema.items()},
    }


@domain.resource(
    "arena://evaluation/runs/{run_id}/metrics/summary",
    mime_type="application/json",
    description="Contestants, stages, maps and success rates of a run",
)
def metrics_summary_resource(run_id: str) -> dict[str, Any]:
    summary = summary_of(_open_catalog(run_id))
    summary.pop("_snapshot_id", None)
    return {"run_id": run_id, **summary}


@domain.tool(read_only=False, destructive=False, idempotent=True)
def evaluation_analysis(
    procedure: Annotated[
        str | None,
        pydantic.Field(description="Named analysis procedure to run; omit to list all procedures and parameter schemas"),
    ] = None,
    params: Annotated[
        dict[str, Any] | None,
        pydantic.Field(description="Procedure parameter dictionary"),
    ] = None,
    run_id: OptionalRunId = None,
    save_as: Annotated[
        str | None,
        pydantic.Field(description="Optional purpose/label for pinned evidence"),
    ] = None,
) -> dict[str, Any]:
    """List or run a named statistical analysis procedure over a run and pin its evidence."""
    from arena_evaluation.analytics.analyses import list_procedures, run_procedure
    from arena_evaluation.analytics.evidence import EvidenceStore

    if procedure is None:
        return {"procedures": list_procedures()}

    path = run_path(run_id)
    catalog = _open_catalog(run_id)
    store = EvidenceStore(path)
    try:
        result = run_procedure(
            procedure,
            catalog,
            params,
            evidence_store=store,
            save_as=save_as,
        )
    except (KeyError, ValueError, pydantic.ValidationError) as exc:
        raise ToolError(
            str(exc),
            hint="call evaluation_analysis() without procedure to see valid procedures and schemas",
        ) from exc

    summary_rows = result.summary.to_dicts()
    derived_table: str | None = None
    derived_summary_table: str | None = None
    if save_as is not None:
        from arena_evaluation.analytics.derived import save_derived_table

        definition = {
            "kind": "procedure",
            "procedure": procedure,
            "params": params or {},
        }
        specs = (
            (save_as, result.frame, definition),
            (
                f"{save_as}_summary",
                result.summary,
                {**definition, "kind": "procedure_summary"},
            ),
        )
        try:
            saved = [
                save_derived_table(
                    path,
                    name=name,
                    frame=frame,
                    definition=spec_definition,
                    snapshot_id=result.snapshot_id,
                    purpose=save_as,
                    author="analyst",
                )
                for name, frame, spec_definition in specs
            ]
        except (ValueError, OSError) as exc:
            raise ToolError(
                f"cannot save derived table {save_as!r}: {exc}",
                hint="use a save_as of letters, digits, '_' or '-', not starting with '-'",
            ) from exc
        derived_table, derived_summary_table = (info["table_name"] for info in saved)
        append_audit_event(
            path,
            role="analyst",
            tool="evaluation_analysis",
            args={"procedure": procedure, "save_as": save_as},
            produced_ids=[
                result.evidence_id,
                derived_table,
                derived_summary_table,
            ],
        )
    return {
        "run_id": path.name,
        "procedure": procedure,
        "snapshot_id": result.snapshot_id,
        "evidence_id": result.evidence_id,
        "n_rows": result.frame.height,
        "summary": summary_rows[0] if len(summary_rows) == 1 else summary_rows,
        "preview": result.frame.head(20).to_dicts(),
        "excluded_rows": dict(result.excluded_rows),
        "seeds_used": list(result.seeds_used),
        **({"derived_table": derived_table} if derived_table else {}),
        **({"derived_summary_table": derived_summary_table} if derived_summary_table else {}),
    }


def rerun_sensitivity_params(run_dir: pathlib.Path, params: Mapping[str, Any]) -> pl.DataFrame:
    """Rerun sensitivity telemetry from persisted evidence parameters."""
    raw_episodes = params.get("episodes")
    episodes = [int(e) for e in raw_episodes] if isinstance(raw_episodes, list) else None
    raw_dp = params.get("detector_params")
    dp = raw_dp if isinstance(raw_dp, dict) else None
    return rerun_sensitivity(run_dir, episode_ids=episodes, detector_params=dp)


@domain.tool(read_only=False, destructive=False, idempotent=True)
def evaluation_sensitivity(
    run_id: RunId,
    episodes: Annotated[
        list[int] | None,
        pydantic.Field(description="Episode IDs to rerun in memory; omit for all"),
    ] = None,
    detector_params: Annotated[
        dict[str, Any] | None,
        pydantic.Field(description="Detector parameters for sensitivity rerun"),
    ] = None,
    save_as: Annotated[str | None, pydantic.Field(description="Optional label for pinned evidence")] = None,
) -> dict[str, Any]:
    """Run an in-memory detector-level sensitivity rerun over telemetry; it writes no variant and no published file."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.evidence import EvidenceStore

    path = run_path(run_id)
    catalog = BenchmarkCatalog.open(path, variant="sensitivity")
    frame = rerun_sensitivity(path, episode_ids=episodes, detector_params=detector_params)
    store = EvidenceStore(path)
    record = store.record(
        frame,
        kind="computation",
        operation="evaluation_sensitivity",
        snapshot_id=catalog.snapshot_id,
        purpose=save_as or "sensitivity",
        params={"episodes": episodes, "detector_params": detector_params or {}},
    )
    append_audit_event(
        path,
        role="analyst",
        tool="evaluation_sensitivity",
        args={
            "run_id": run_id,
            "episodes": episodes,
            "detector_params": detector_params or {},
            "save_as": save_as,
        },
        produced_ids=[record.evidence_id],
    )
    return {
        "run_id": path.name,
        "snapshot_id": catalog.snapshot_id,
        "evidence_id": record.evidence_id,
        "is_sensitivity": True,
        "n_rows": frame.height,
        "preview": frame.head(20).to_dicts(),
    }


@domain.tool(read_only=True)
def evaluation_tables(run_id: RunId) -> dict[str, Any]:
    """List derived tables saved under ``<run>/analysis/derived`` with freshness status."""
    from arena_evaluation.analytics.derived import list_derived_tables

    path = run_path(run_id)
    catalog = _open_catalog(run_id)
    tables = list_derived_tables(path, current_snapshot_id=catalog.snapshot_id)
    return {"run_id": path.name, "snapshot_id": catalog.snapshot_id, "tables": tables}


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_refresh_table(run_id: RunId, name: str) -> dict[str, Any]:
    """Recompute a stale derived table from its stored definition against the current snapshot."""
    from arena_evaluation.analytics.derived import refresh_derived_table

    path = run_path(run_id)
    try:
        info = refresh_derived_table(path, name)
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(
            str(exc),
            hint="evaluation_tables lists saved derived tables",
        ) from exc
    append_audit_event(
        path,
        role="analyst",
        tool="evaluation_refresh_table",
        args={"run_id": run_id, "name": name},
        produced_ids=[info["table_name"]],
    )
    return {"run_id": path.name, **info}
