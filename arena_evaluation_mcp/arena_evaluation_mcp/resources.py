from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp.types import (
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextResourceContents,
)

if TYPE_CHECKING:
    from .eval_bridge import EvalBridge


def build_resources_list() -> list[Resource]:
    """Return the complete list of available resources."""
    return [
        Resource(
            uri="arena_eval://benchmarks",
            name="All benchmark runs",
            description="JSON array of all benchmark runs with id, suite, contest, step counts, status.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://benchmarks/{benchmark_id}/state",
            name="Benchmark run state",
            description="manifest.yaml + .benchmark_state.json contents combined as JSON.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://benchmarks/{benchmark_id}/metrics/schema",
            name="Metrics schema",
            description="Column names and dtypes of combined_metrics.parquet.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://benchmarks/{benchmark_id}/metrics/summary",
            name="Metrics quick summary",
            description="Row count, planner list, stage list, success rates.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://benchmarks/{benchmark_id}/notes",
            name="Agent-written notes",
            description="Contents of notes.yaml as JSON array of {label, value} objects.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://benchmarks/{benchmark_id}/report",
            name="Report path",
            description="Absolute path to report.html if it exists, or null.",
            mime_type="text/plain",
        ),
        Resource(
            uri="arena_eval://manifests",
            name="Available report manifests",
            description="List of bundled manifest YAML stems.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://suites",
            name="Available benchmark suites",
            description="List of bundled suite YAML stems.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://contests",
            name="Available benchmark contests",
            description="List of bundled contest YAML stems.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://maps",
            name="Available world/map names",
            description="List of known world/map names usable in suite stages.",
            mime_type="application/json",
        ),
        Resource(
            uri="arena_eval://robots",
            name="Available robot models",
            description="List of known robot models usable in suite stages.",
            mime_type="application/json",
        ),
    ]


def read_resource_content(params: ReadResourceRequestParams, bridge: EvalBridge) -> ReadResourceResult:
    """Handle resource read requests."""
    try:
        payload = _read(str(params.uri), bridge)
    except ValueError as exc:
        payload = {"error": str(exc)}
    if isinstance(payload, str):
        return _text_result(str(params.uri), payload)
    return _json_result(str(params.uri), payload)


def _json_result(uri: str, data: object) -> ReadResourceResult:
    """Create a ReadResourceResult with JSON text content."""
    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=uri,
                mime_type="application/json",
                text=json.dumps(data, default=str),
            )
        ]
    )


def _text_result(uri: str, text: str) -> ReadResourceResult:
    """Create a ReadResourceResult with plain text content."""
    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=uri,
                mime_type="text/plain",
                text=text,
            )
        ]
    )


def _read(uri: str, bridge: EvalBridge) -> object:
    """Resolve a resource URI to its payload (dict for JSON, str for plain text)."""
    if uri == "arena_eval://benchmarks":
        return bridge.list_benchmarks()

    if uri == "arena_eval://manifests":
        return bridge.discover_available_manifests()

    if uri == "arena_eval://suites":
        return bridge.list_suite_stems()

    if uri == "arena_eval://contests":
        return bridge.list_contest_stems()

    if uri == "arena_eval://maps":
        return bridge.discover_available_maps()

    if uri == "arena_eval://robots":
        return bridge.discover_available_robots()

    prefix = "arena_eval://benchmarks/"
    if uri.startswith(prefix):
        rest = uri[len(prefix) :]
        parts = rest.split("/", 1)
        benchmark_id = parts[0]
        sub = parts[1] if len(parts) > 1 else ""

        if sub == "state":
            manifest = bridge.read_benchmark_manifest(benchmark_id)
            state = bridge.read_benchmark_state(benchmark_id)
            return {
                "benchmark_id": benchmark_id,
                "manifest": manifest,
                "state": state,
            }

        if sub == "metrics/schema":
            df = bridge.load_combined_metrics(benchmark_id)
            if df is None:
                return {"error": "No combined_metrics.parquet found"}
            schema = {c: str(df[c].dtype) for c in df.columns}
            return {
                "benchmark_id": benchmark_id,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": df.columns,
                "dtypes": schema,
            }

        if sub == "metrics/summary":
            df = bridge.load_combined_metrics(benchmark_id)
            if df is None:
                return {"error": "No combined_metrics.parquet found"}
            planner_col = "local_planner" if "local_planner" in df.columns else "planner"
            planners = df[planner_col].drop_nulls().unique().to_list() if planner_col in df.columns else []
            stages = df["stage"].drop_nulls().unique().to_list() if "stage" in df.columns else []
            maps_list = df["map"].drop_nulls().unique().to_list() if "map" in df.columns else []
            succ_mean = df["success"].drop_nulls().mean() if "success" in df.columns else None
            succ = float(succ_mean) if isinstance(succ_mean, (int, float)) else None
            return {
                "benchmark_id": benchmark_id,
                "n_rows": len(df),
                "planners": planners,
                "stages": stages,
                "maps": maps_list,
                "overall_success_rate": round(succ, 3) if succ is not None else None,
            }

        if sub == "notes":
            notes = bridge.read_notes(benchmark_id)
            return {"benchmark_id": benchmark_id, "notes": notes}

        if sub == "report":
            report_path = bridge.benchmark_dir(benchmark_id) / "report.html"
            return str(report_path) if report_path.exists() else "null"

    return {"error": f"unknown resource: {uri}"}
