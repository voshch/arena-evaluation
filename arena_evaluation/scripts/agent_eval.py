"""Standalone hypothesis-to-report evaluation script via an in-process MCPServer."""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent


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

    for hyp in truth.get("hypotheses", []):
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
                "save_as": f"proof_{hyp['id']}",
            },
            called=called,
        )
        ev_id = str(cmp_res["evidence_id"])
        summary = cmp_res["summary"]

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
            mean_diff = float(summary["mean_diff"])
            ci_low = float(summary["ci_low"])
            ci_high = float(summary["ci_high"])
            q_res = await _call(
                server,
                "evaluation_query",
                {
                    "run_id": run_id,
                    "query": f"SELECT AVG({hyp['metric']}) AS val FROM derived_proof_{hyp['id']}",
                    "pin_evidence": True,
                },
                called=called,
            )
            q_ev_id = str(q_res["evidence_id"])
            q_val = float(q_res["data"][0]["val"])
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
                            "value": q_val,
                            "tolerance": 1e-4,
                        }
                    ],
                    "numbers": {"mean_diff": mean_diff, "mean_val": q_val},
                    "author": "analyst",
                },
                called=called,
            )
            target_status = "verified" if ci_high < 0.0 or ci_low > 0.0 else "refuted"

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
        results.append(
            {
                "hypothesis_id": hyp["id"],
                "expected_verdict": hyp["expected_verdict"],
                "actual_verdict": f_stored["status"],
                "mechanical_passed": f_stored["mechanical_passed"],
                "finding_id": finding_id,
                "evidence_id": f_stored["evidence_id"],
                "snapshot_id": sch["snapshot_id"],
                "report_path": report["report_path"],
            }
        )
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
