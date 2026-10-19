from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, cast

import pydantic

from arena_mcp.core import ToolError
from arena_mcp.core.msgs import jsonable

from .base import RunId, domain, run_path

if TYPE_CHECKING:
    import polars as pl

CONTESTANT = "planner"
Metrics = Annotated[list[str] | None, pydantic.Field(description="metric columns, the metrics' declared primary outputs when omitted")]
References = Annotated[bool, pydantic.Field(description="keep reference episodes (is_reference), which are not contestant runs")]
Scalar = str | int | float | bool
Filter = Annotated[dict[str, Scalar | list[Scalar]] | None, pydantic.Field(description="equality filter, {column: value} or {column: [values]}")]


def metric_rows() -> list[dict[str, Any]]:
    """Every registered metric calculator with its outputs, units, primary outputs and declared directions."""
    from arena_evaluation.processing.metrics.registry import MetricRegistry
    from arena_evaluation.storage.schemas import RobotParams

    rows = MetricRegistry(RobotParams()).list_metrics()
    units = MetricRegistry.get_all_units()
    for row in rows:
        row["units"] = {key: units.get(key) for key in row["outputs"]}
    return rows


def directions(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Output key to 'higher' or 'lower', for the outputs whose calculator declares one."""
    out: dict[str, str] = {}
    for row in rows:
        out.update(row["output_directions"])
    return out


def primary_outputs(rows: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(key for row in rows for key in row["primary_outputs"]))


def load_metrics(run_id: str) -> pl.DataFrame:
    """combined_metrics.parquet of a run, a ToolError when it was not processed yet."""
    from arena_evaluation.processing.parquet_store import ParquetStore

    path = run_path(run_id) / "combined_metrics.parquet"
    if not path.is_file():
        raise ToolError(f"run {run_id!r} has no combined_metrics.parquet", hint="evaluation_process computes it")
    frame, _ = ParquetStore.read(path)
    return frame


def episodes(frame: pl.DataFrame, include_references: bool) -> pl.DataFrame:
    import polars as pl

    if include_references or "is_reference" not in frame.columns:
        return frame
    return frame.filter(~pl.col("is_reference").fill_null(False).cast(pl.Boolean))


def columns(frame: pl.DataFrame, names: list[str], what: str) -> list[str]:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ToolError(f"no {what} column {', '.join(missing)}", hint=f"columns: {', '.join(frame.columns)}")
    return names


def scalar_frame(frame: pl.DataFrame, keep: list[str]) -> pl.DataFrame:
    """The kept columns, list columns exploded so means run over their elements."""
    import polars as pl

    selected = frame.select(list(dict.fromkeys(keep)))
    lists = [name for name, dtype in selected.schema.items() if isinstance(dtype, pl.List)]
    return selected.explode(lists) if lists else selected


def numeric_columns(frame: pl.DataFrame) -> list[str]:
    import polars as pl

    return [name for name, dtype in frame.schema.items() if dtype.is_numeric() or dtype == pl.Boolean or (isinstance(dtype, pl.List) and dtype.inner.is_numeric())]


def means(frame: pl.DataFrame, group: list[str], metrics: list[str], *, spread: bool = False) -> list[dict[str, Any]]:
    import polars as pl

    work = scalar_frame(frame, [*group, *metrics]).with_columns([pl.col(name).cast(pl.Float64) for name in metrics])
    aggregations = [pl.col(name).mean().alias(name) for name in metrics]
    if spread:
        aggregations += [pl.col(name).std().alias(f"{name}_std") for name in metrics]
    aggregations.append(pl.len().alias("n_rows"))
    return [cast("dict[str, Any]", jsonable(row)) for row in work.group_by(group).agg(aggregations).sort(group).to_dicts()]


def normalized(rows: list[dict[str, Any]], metric: str, direction: str) -> None:
    """Adds <metric>_norm in [0, 1] to every row, 1 for the best row, 0.5 when all rows tie."""
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if not values:
        return
    low, high = min(values), max(values)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        norm = 0.5 if high == low else (value - low) / (high - low)
        row[f"{metric}_norm"] = round(norm if direction == "higher" else 1.0 - norm, 3)


def ranked(rows: list[dict[str, Any]], metrics: list[str], weights: dict[str, float]) -> list[dict[str, Any]]:
    """Rows with a weighted composite_score over the normalized metrics, best first, rows without a score last."""
    for row in rows:
        scored = [(weights[metric], row[f"{metric}_norm"]) for metric in metrics if row.get(f"{metric}_norm") is not None]
        total = sum(weight for weight, _ in scored)
        row["composite_score"] = round(sum(weight * value for weight, value in scored) / total, 3) if scored and total > 0 else None
    rows.sort(key=lambda row: (row["composite_score"] is None, -(row["composite_score"] or 0.0)))
    for index, row in enumerate(rows):
        row["rank"] = index + 1 if row["composite_score"] is not None else None
    return rows


def split_by_direction(metrics: list[str], declared: dict[str, str]) -> tuple[list[str], list[str]]:
    return [metric for metric in metrics if metric in declared], [metric for metric in metrics if metric not in declared]


def compare_frame(frame: pl.DataFrame, metrics: list[str], declared: dict[str, str], *, planners: list[str] | None = None, group_by_stage: bool = False, normalize: bool = True, include_references: bool = False) -> dict[str, Any]:
    """Contestants compared on metric means, ranked over the metrics with a declared direction."""
    import polars as pl

    frame = episodes(frame, include_references)
    columns(frame, [CONTESTANT], "contestant")
    if planners:
        frame = frame.filter(pl.col(CONTESTANT).is_in(planners))
    columns(frame, metrics, "metric")
    group = [CONTESTANT, *(["stage"] if group_by_stage and "stage" in frame.columns else [])]
    rows = means(frame, group, metrics, spread=True)
    scored, unranked = split_by_direction(metrics, declared)
    if normalize:
        for metric in scored:
            normalized(rows, metric, declared[metric])
        ranked(rows, scored, dict.fromkeys(scored, 1.0))
    return {"contestant_column": CONTESTANT, "n_rows": len(rows), "metrics": metrics, "ranked_on": scored, "unranked_metrics": unranked, "rankings": rows}


def top_n_frame(frame: pl.DataFrame, metrics: list[str], declared: dict[str, str], *, n: int = 3, weights: list[float] | None = None, include_references: bool = False) -> dict[str, Any]:
    """The n best contestants by a weighted composite of their normalized metric means."""
    if weights is not None and len(weights) != len(metrics):
        raise ToolError(f"{len(weights)} weights for {len(metrics)} metrics", hint="give one weight per metric, in the same order")
    frame = episodes(frame, include_references)
    columns(frame, [CONTESTANT], "contestant")
    columns(frame, metrics, "metric")
    weight_of = dict(zip(metrics, weights or [1.0] * len(metrics), strict=True))
    scored, unranked = split_by_direction(metrics, declared)
    if not scored:
        raise ToolError(f"none of {metrics} declares whether higher or lower is better, so nothing can be ranked", hint="evaluation_describe_metric shows each output's direction")
    rows = means(frame, [CONTESTANT], scored)
    for metric in scored:
        normalized(rows, metric, declared[metric])
    ranked(rows, scored, weight_of)
    return {"ranked_on": scored, "unranked_metrics": unranked, "weights": {metric: weight_of[metric] for metric in scored}, "top_n": rows[:n]}


def correlation_frame(frame: pl.DataFrame, metrics: list[str]) -> dict[str, Any]:
    """Pairwise Pearson correlation of per-episode metric values, None where undefined."""
    import polars as pl

    columns(frame, metrics, "metric")
    if len(metrics) < 2:
        raise ToolError("correlation needs at least two metrics")
    lists = [name for name in metrics if isinstance(frame.schema[name], pl.List)]
    if lists:
        raise ToolError(f"{', '.join(lists)} hold one list per episode, correlate scalar metrics", hint="evaluation_read_combined_metrics shows each column's dtype")
    work = frame.select([pl.col(name).cast(pl.Float64) for name in metrics]).to_pandas().corr()
    matrix = {}
    for row in metrics:
        matrix[row] = {}
        for col in metrics:
            value = float(work.loc[row, col])
            matrix[row][col] = round(value, 3) if math.isfinite(value) else None
    return {"metrics": metrics, "correlation_matrix": matrix}


def column_stats(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    import polars as pl

    stats: dict[str, dict[str, Any]] = {}
    for name, dtype in frame.schema.items():
        series = frame[name]
        entry: dict[str, Any] = {"dtype": str(dtype), "null_count": series.null_count()}
        if dtype.is_numeric():
            present = series.drop_nulls()
            if len(present):
                entry.update(min=jsonable(float(cast("float", present.min()))), max=jsonable(float(cast("float", present.max()))), mean=jsonable(float(cast("float", present.mean()))))
        elif isinstance(dtype, pl.List):
            entry["mean_length"] = jsonable(series.list.len().mean())
        elif dtype in (pl.String, pl.Boolean, pl.Categorical) or isinstance(dtype, pl.Enum):
            unique = series.drop_nulls().unique().sort().to_list()
            entry.update(n_unique=len(unique), unique_values=jsonable(unique[:20]))
        stats[name] = entry
    return stats


def summary_of(frame: pl.DataFrame, include_references: bool = False) -> dict[str, Any]:
    import polars as pl

    work = episodes(frame, include_references)

    def distinct(name: str) -> list[Any]:
        return sorted(work[name].drop_nulls().unique().to_list()) if name in work.columns else []

    contestants = []
    if CONTESTANT in work.columns:
        for name in distinct(CONTESTANT):
            rows = work.filter(pl.col(CONTESTANT) == name)
            success = rows["success"].cast(pl.Float64).drop_nulls().mean() if "success" in rows.columns else None
            contestants.append({"planner": name, "n_episodes": len(rows), "success_rate": round(float(cast("float", success)), 3) if success is not None else None})
    overall = work["success"].cast(pl.Float64).drop_nulls().mean() if "success" in work.columns else None
    return {
        "n_rows": len(frame),
        "n_episodes": len(work),
        "n_references": len(frame) - len(work),
        "planners": distinct(CONTESTANT),
        "stages": distinct("stage"),
        "maps": distinct("map"),
        "overall_success_rate": round(float(cast("float", overall)), 3) if overall is not None else None,
        "per_planner": sorted(contestants, key=lambda entry: (entry["success_rate"] is None, -(entry["success_rate"] or 0.0))),
    }


@domain.tool(read_only=True)
def evaluation_metrics() -> dict[str, Any]:
    """Every registered metric: category, outputs with units, primary outputs, declared directions, required topics, dependencies."""
    return {"metrics": metric_rows()}


@domain.tool(read_only=True)
def evaluation_describe_metric(metric_name: str) -> dict[str, Any]:
    """One metric's outputs with unit and direction (higher, lower or undeclared), dependencies, required topics and pedsim need."""
    rows = metric_rows()
    row = next((entry for entry in rows if entry["name"] == metric_name), None)
    if row is None:
        raise ToolError(f"no metric {metric_name!r}", hint=f"metrics: {', '.join(sorted(entry['name'] for entry in rows))}")
    declared = row["output_directions"]
    return {
        **{key: row[key] for key in ("name", "category", "requires_pedsim", "depends_on", "required_topics", "primary_outputs")},
        "outputs": {key: {"unit": row["units"].get(key), "direction": declared.get(key), "lower_is_better": None if key not in declared else declared[key] == "lower"} for key in row["outputs"]},
    }


@domain.tool(read_only=True)
def evaluation_read_combined_metrics(run_id: RunId) -> dict[str, Any]:
    """Shape, columns and per-column statistics of a run's combined_metrics.parquet."""
    frame = load_metrics(run_id)
    return {"run_id": run_id, "n_rows": len(frame), "n_columns": len(frame.columns), "columns": frame.columns, "column_stats": column_stats(frame)}


@domain.tool(read_only=True)
def evaluation_query_metrics(
    run_id: RunId,
    group_by: list[str],
    metrics: list[str],
    filter: Filter = None,
    sort_by: Annotated[str | None, pydantic.Field(description="metric column to sort by")] = None,
    descending: bool = True,
    include_references: References = False,
) -> dict[str, Any]:
    """Mean of metric columns grouped by columns, after an equality filter, over combined_metrics.parquet."""
    import polars as pl

    frame = episodes(load_metrics(run_id), include_references)
    for name, value in (filter or {}).items():
        columns(frame, [name], "filter")
        frame = frame.filter(pl.col(name).is_in(value) if isinstance(value, list) else pl.col(name) == value)
    columns(frame, group_by, "group_by")
    columns(frame, metrics, "metric")
    rows = means(frame, group_by, metrics)
    if sort_by is not None:
        if sort_by not in metrics:
            raise ToolError(f"sort_by {sort_by!r} is not one of the metrics {metrics}")
        rows.sort(key=lambda row: (row[sort_by] is None, -(row[sort_by] or 0.0) if descending else (row[sort_by] or 0.0)))
    return {"run_id": run_id, "n_rows": len(rows), "data": rows}


@domain.tool(read_only=True)
def evaluation_aggregate(
    run_id: RunId,
    dimension: Annotated[str, pydantic.Field(description="column to group by, e.g. planner, stage, map, robot")],
    metrics: Annotated[list[str] | None, pydantic.Field(description="metric columns, every numeric column when omitted")] = None,
    include_references: References = False,
) -> dict[str, Any]:
    """Mean of metrics per value of one dimension."""
    frame = episodes(load_metrics(run_id), include_references)
    columns(frame, [dimension], "dimension")
    chosen = columns(frame, metrics, "metric") if metrics else [name for name in numeric_columns(frame) if name != dimension]
    return {"run_id": run_id, "dimension": dimension, "metrics": chosen, "data": means(frame, [dimension], chosen)}


@domain.tool(read_only=True)
def evaluation_compare_planners(
    run_id: RunId,
    planners: Annotated[list[str] | None, pydantic.Field(description="contestant names, all when omitted")] = None,
    metrics: Metrics = None,
    group_by_stage: bool = False,
    normalize: Annotated[bool, pydantic.Field(description="add <metric>_norm in [0, 1], best = 1, and a composite rank")] = True,
    include_references: References = False,
) -> dict[str, Any]:
    """Contestants (the planner column, one per contest entry) side by side on metric means and spreads, ranked over metrics with a declared direction."""
    rows = metric_rows()
    frame = load_metrics(run_id)
    chosen = metrics or [name for name in primary_outputs(rows) if name in frame.columns]
    return {"run_id": run_id, **compare_frame(frame, chosen, directions(rows), planners=planners, group_by_stage=group_by_stage, normalize=normalize, include_references=include_references)}


@domain.tool(read_only=True)
def evaluation_top_n(
    run_id: RunId,
    n: int = 3,
    metrics: Metrics = None,
    weights: Annotated[list[float] | None, pydantic.Field(description="one weight per metric, equal weights when omitted")] = None,
    include_references: References = False,
) -> dict[str, Any]:
    """The n best contestants by a weighted composite of normalized metric means, metrics without a declared direction left out."""
    rows = metric_rows()
    frame = load_metrics(run_id)
    chosen = metrics or [name for name in primary_outputs(rows) if name in frame.columns]
    return {"run_id": run_id, **top_n_frame(frame, chosen, directions(rows), n=n, weights=weights, include_references=include_references)}


@domain.tool(read_only=True)
def evaluation_correlation(run_id: RunId, metrics: list[str], include_references: References = False) -> dict[str, Any]:
    """Pairwise Pearson correlation matrix of scalar metrics across episodes."""
    return {"run_id": run_id, **correlation_frame(episodes(load_metrics(run_id), include_references), metrics)}


@domain.tool(read_only=True)
def evaluation_summarize(run_id: RunId, include_references: References = False) -> dict[str, Any]:
    """A run's progress and results: step counts, contestants with episode counts and success rates, stages and maps, as data and text."""
    from .runs import run_status

    status = run_status(run_path(run_id))
    summary = summary_of(load_metrics(run_id), include_references)
    lines = [
        f"Benchmark {run_id} ({status['suite']} x {status['contest']}): {status['state']}, {status['steps']['ok']}/{status['steps_total']} steps ok",
        f"Contestants: {', '.join(map(str, summary['planners']))}",
        f"Stages: {', '.join(map(str, summary['stages']))}",
        f"Maps: {', '.join(map(str, summary['maps']))}",
        f"Episodes: {summary['n_episodes']} ({summary['n_references']} reference rows left out)" if not include_references else f"Episodes: {summary['n_episodes']} including references",
    ]
    for entry in summary["per_planner"]:
        rate = "n/a" if entry["success_rate"] is None else f"{entry['success_rate'] * 100:.1f}%"
        lines.append(f"  {entry['planner']}: {entry['n_episodes']} episodes, success {rate}")
    return {"run_id": run_id, "summary": "\n".join(lines), **summary, "status": status}


@domain.resource("arena://evaluation/runs/{run_id}/metrics/schema", mime_type="application/json", description="Columns and dtypes of a run's combined_metrics.parquet")
def metrics_schema_resource(run_id: str) -> dict[str, Any]:
    frame = load_metrics(run_id)
    return {"run_id": run_id, "n_rows": len(frame), "dtypes": {name: str(dtype) for name, dtype in frame.schema.items()}}


@domain.resource("arena://evaluation/runs/{run_id}/metrics/summary", mime_type="application/json", description="Contestants, stages, maps and success rates of a run")
def metrics_summary_resource(run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, **summary_of(load_metrics(run_id))}
