"""Standalone hypothesis-to-report evaluation script via an in-process MCPServer.

The proof is allowed to fail. A hypothesis's verdict is direction aware (a lower-than
claim needs a strictly negative interval, a two-sided claim only excludes zero), the
interval must have a non-zero width, and every quoted number is recomputed from the pinned
compare evidence before it is published. The run also writes its result table and tool
transcript next to the run's analysis artifacts (H-001, L5-037).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
from typing import Any

import polars as pl
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

RESULTS_FILENAME = "agent_eval_results.parquet"
TRANSCRIPT_FILENAME = "agent_eval_transcript.json"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER_TOLERANCE = 1e-4


async def _call(
    server: MCPServer,
    name: str,
    arguments: dict[str, Any],
    *,
    called: list[str] | None = None,
) -> dict[str, Any]:
    """Call an MCP tool on ``server`` and return its structured content."""
    if called is not None:
        called.append(name)
    res = await server.call_tool(name, arguments)
    assert res.structured_content is not None, f"{name} returned no structured_content"
    return dict(res.structured_content)


def _require_identifier(value: str, what: str) -> str:
    """Refuse a metric or planner name that is not a plain SQL identifier."""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{what} {value!r} is not a plain identifier")
    return value


def verdict_for(direction: str, ci_low: float, ci_high: float) -> str:
    """Map a claim direction and its interval to ``verified``/``refuted``.

    ``lower`` needs the whole interval below zero, ``higher`` above zero, and
    ``different`` needs the interval to exclude zero on either side.
    """
    match direction:
        case "lower":
            verified = ci_high < 0.0
        case "higher":
            verified = ci_low > 0.0
        case "different":
            verified = ci_low > 0.0 or ci_high < 0.0
        case _:
            raise ValueError(
                f"unknown claim direction {direction!r}; use lower, higher or different"
            )
    return "verified" if verified else "refuted"


async def _quoted_planner_mean(
    server: MCPServer,
    run_id: str,
    *,
    derived_table: str,
    planner: str,
    metric: str,
    called: list[str],
) -> tuple[float, str]:
    """Quote the claim planner's mean from the comparison table and pin it as evidence."""
    planner = _require_identifier(planner, "planner")
    metric = _require_identifier(metric, "metric")
    query = (
        f"SELECT AVG({metric}) AS val FROM {derived_table} WHERE planner = '{planner}'"
    )
    result = await _call(
        server,
        "evaluation_query",
        {"run_id": run_id, "query": query, "pin_evidence": True},
        called=called,
    )
    value = float(result["data"][0]["val"])
    return value, str(result["evidence_id"])


def _recompute_from_evidence(
    evidence_rows: list[dict[str, Any]], planner: str, metric: str
) -> tuple[float, float]:
    """Recompute the planner mean and the pooled mean from the pinned compare frame."""
    frame = pl.DataFrame(evidence_rows)
    planner_rows = frame.filter(pl.col("planner") == planner)
    if planner_rows.height == 0:
        raise AssertionError(f"no evidence rows for planner {planner!r}")
    return (
        float(planner_rows[metric].mean()),
        float(frame[metric].mean()),
    )


def _run_analysis_dir(run_id: str) -> pathlib.Path:
    """Return ``<run>/analysis`` for ``run_id`` and make sure it exists."""
    from arena_evaluation.storage.data_root import benchmarks_root

    analysis = benchmarks_root() / run_id / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    return analysis


def _write_artifacts(
    run_id: str, results: list[dict[str, Any]], called: list[str]
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the result table and the tool transcript into the run's analysis directory."""
    analysis = _run_analysis_dir(run_id)
    results_path = analysis / RESULTS_FILENAME
    transcript_path = analysis / TRANSCRIPT_FILENAME
    pl.DataFrame(results).write_parquet(results_path)
    transcript_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "called_tools": called,
                "results_path": str(results_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results_path, transcript_path


async def evaluate_hypotheses_async(
    run_id: str, truth_path: pathlib.Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the scripted hypothesis-to-report loop through an in-process MCPServer session."""
    from arena_evaluation.mcp import domain

    server = MCPServer("agent-eval")
    domain.install(server)
    called: list[str] = []

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    sch = await _call(server, "evaluation_schema", {"run_id": run_id}, called=called)
    draft = await _call(
        server,
        "evaluation_report_draft",
        {"run_id": run_id, "title": f"Autonomous Proof Report ({run_id})"},
        called=called,
    )
    draft_id = str(draft["draft_id"])
    created_ids: list[str] = []
    measurements: dict[str, dict[str, Any]] = {}

    for hyp in truth.get("hypotheses", []):
        save_as = f"proof_{hyp['id']}"
        cmp_res = await _call(
            server,
            "evaluation_analysis",
            {
                "procedure": "compare",
                "run_id": run_id,
                "params": {
                    "planner_a": hyp["planner_a"],
                    "planner_b": hyp["planner_b"],
                    "metric": hyp["metric"],
                },
                "save_as": save_as,
            },
            called=called,
        )
        ev_id = str(cmp_res["evidence_id"])
        summary = cmp_res["summary"]
        measured: dict[str, Any] = {
            "direction": hyp.get("direction", "different"),
            "ci_low": None,
            "ci_high": None,
            "ci_width": None,
            "n_pairs": None,
            "mean_diff": None,
            "quoted_number": None,
            "evidence_recomputed_number": None,
            "pooled_number": None,
            "number_resolved": False,
            "derived_table": cmp_res.get("derived_table"),
        }

        if cmp_res["n_rows"] == 0 or summary.get("reason") == "all_null":
            f_obj = await _call(
                server,
                "evaluation_finding_create",
                {
                    "run_id": run_id,
                    "claim": hyp["claim"],
                    "evidence_ids": [ev_id],
                    "author": "analyst",
                },
                called=called,
            )
            target_status = "needs_work"
        else:
            ci_low = float(summary["ci_low"])
            ci_high = float(summary["ci_high"])
            mean_diff = float(summary["mean_diff"])
            assert ci_high > ci_low, (
                f"{hyp['id']}: interval [{ci_low}, {ci_high}] has zero width; "
                "the proof cannot falsify a deterministic fixture"
            )
            derived_table = cmp_res.get("derived_table")
            assert derived_table, f"{hyp['id']}: compare wrote no derived table"
            quoted, q_ev_id = await _quoted_planner_mean(
                server,
                run_id,
                derived_table=str(derived_table),
                planner=str(hyp["planner_a"]),
                metric=str(hyp["metric"]),
                called=called,
            )
            evidence = await _call(
                server,
                "evaluation_evidence",
                {"run_id": run_id, "evidence_id": ev_id},
                called=called,
            )
            recomputed, pooled = _recompute_from_evidence(
                evidence["data"], str(hyp["planner_a"]), str(hyp["metric"])
            )
            assert abs(recomputed - quoted) <= NUMBER_TOLERANCE, (
                f"{hyp['id']}: quoted {quoted} != evidence-recomputed {recomputed}"
            )
            if abs(mean_diff) > 1.0:
                assert abs(quoted - pooled) > 0.5, (
                    f"{hyp['id']}: quoted number {quoted} is the pooled mean {pooled}, "
                    "not the compared planner's mean"
                )
            measured.update(
                {
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_width": ci_high - ci_low,
                    "n_pairs": summary.get("n_pairs"),
                    "mean_diff": mean_diff,
                    "quoted_number": quoted,
                    "evidence_recomputed_number": recomputed,
                    "pooled_number": pooled,
                    "number_resolved": True,
                }
            )
            f_obj = await _call(
                server,
                "evaluation_finding_create",
                {
                    "run_id": run_id,
                    "claim": hyp["claim"],
                    "evidence_ids": [ev_id, q_ev_id],
                    "structured_numbers": [
                        {
                            "evidence_id": q_ev_id,
                            "column": "val",
                            "value": quoted,
                            "tolerance": NUMBER_TOLERANCE,
                        }
                    ],
                    "author": "analyst",
                },
                called=called,
            )
            target_status = verdict_for(str(measured["direction"]), ci_low, ci_high)

        finding_id = str(f_obj["finding_id"])
        verify_res = await _call(
            server,
            "evaluation_finding_verify",
            {"run_id": run_id, "finding_id": finding_id},
            called=called,
        )
        if not verify_res["mechanical_passed"]:
            target_status = "needs_work"
        await _call(
            server,
            "evaluation_finding_status",
            {
                "run_id": run_id,
                "finding_id": finding_id,
                "status": target_status,
                "by": "verifier",
            },
            called=called,
        )
        created_ids.append(finding_id)
        measurements[finding_id] = measured

    stored = await _call(
        server, "evaluation_findings", {"run_id": run_id}, called=called
    )
    by_id = {f["finding_id"]: f for f in stored["findings"]}

    await _call(
        server,
        "evaluation_report_plot",
        {
            "run_id": run_id,
            "draft_id": draft_id,
            "plot": {
                "id": "proof_summary",
                "title": "Planted Difference",
                "type": "interval",
                "data": {"source": "derived_proof_H_diff_summary"},
                "channels": {"x": "metric", "y": "mean_diff"},
                "options": {"low": "ci_low", "high": "ci_high"},
            },
        },
        called=called,
    )
    called.append("evaluation_report_view")
    view_res = await server.call_tool(
        "evaluation_report_view",
        {"run_id": run_id, "draft_id": draft_id, "plot_id": "proof_summary"},
    )
    assert any(isinstance(c, ImageContent) for c in view_res.content)
    await _call(
        server,
        "evaluation_report_review",
        {
            "run_id": run_id,
            "draft_id": draft_id,
            "plot_id": "proof_summary",
            "verdict": "approved",
            "reason": "Verified against synthetic truth",
            "by": "reporter",
        },
        called=called,
    )
    report = await _call(
        server,
        "evaluation_report_assemble",
        {
            "run_id": run_id,
            "draft_id": draft_id,
            "name": "proof_report",
        },
        called=called,
    )

    results: list[dict[str, Any]] = []
    for hyp, finding_id in zip(truth.get("hypotheses", []), created_ids, strict=True):
        f_stored = by_id[finding_id]
        measured = measurements[finding_id]
        results.append(
            {
                "hypothesis_id": hyp["id"],
                "direction": measured["direction"],
                "expected_verdict": hyp["expected_verdict"],
                "actual_verdict": f_stored["status"],
                "mechanical_passed": f_stored["mechanical_passed"],
                "finding_id": finding_id,
                "evidence_id": f_stored["evidence_id"],
                "snapshot_id": sch["snapshot_id"],
                "ci_low": measured["ci_low"],
                "ci_high": measured["ci_high"],
                "ci_width": measured["ci_width"],
                "n_pairs": measured["n_pairs"],
                "mean_diff": measured["mean_diff"],
                "quoted_number": measured["quoted_number"],
                "evidence_recomputed_number": measured["evidence_recomputed_number"],
                "pooled_number": measured["pooled_number"],
                "number_resolved": measured["number_resolved"],
                "planted_delta": truth.get("planted_delta"),
                "report_path": report["report_path"],
            }
        )

    results_path, transcript_path = _write_artifacts(run_id, results, called)
    for row in results:
        row["results_path"] = str(results_path)
        row["transcript_path"] = str(transcript_path)
    return results, called


def evaluate_hypotheses(
    run_id: str, truth_path: pathlib.Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the scripted hypothesis-to-report loop synchronously over an MCPServer session."""
    return asyncio.run(evaluate_hypotheses_async(run_id, truth_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run autonomous MCP hypothesis proof.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    out, _ = evaluate_hypotheses(args.run_id, args.truth)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
