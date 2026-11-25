"""Contract proof tests: scripted hypothesis-to-report loop over MCPServer."""

from __future__ import annotations

import asyncio
import json
import pathlib

import polars as pl
import pytest

from agent_eval import RESULTS_FILENAME, TRANSCRIPT_FILENAME, evaluate_hypotheses
from h6_fixture import write_noisy_benchmark
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp_fixtures import arena_data

from arena_evaluation.mcp import domain, roles


def test_h6_scripted_truth_loop_verifies_refutes_and_flags_missing(
    tmp_path: pathlib.Path,
) -> None:
    """Every hypothesis reaches its expected verdict; the interval is noisy, direction-aware, and evidence-checked."""
    run_dir = tmp_path / "data" / "benchmarks" / "h6_noisy"
    truth = write_noisy_benchmark(run_dir)

    with arena_data(tmp_path):
        results, called_tools = evaluate_hypotheses("h6_noisy", run_dir / "truth.json")

    assert len(results) == 3
    for row in results:
        assert row["actual_verdict"] == row["expected_verdict"], row
        assert row["mechanical_passed"] is True
        assert row["evidence_id"]
        assert pathlib.Path(row["report_path"]).is_file()

    by_id = {row["hypothesis_id"]: row for row in results}
    difference = by_id["H_diff"]
    assert difference["ci_width"] is not None and difference["ci_width"] > 0.0
    assert difference["ci_low"] <= truth["planted_delta"] <= difference["ci_high"]
    assert difference["quoted_number"] is not None
    assert abs(difference["quoted_number"] - difference["evidence_recomputed_number"]) <= 1e-4
    assert abs(difference["quoted_number"] - difference["pooled_number"]) > 0.5
    assert difference["number_resolved"] is True

    null = by_id["H_null"]
    assert null["ci_width"] is not None and null["ci_width"] > 0.0
    assert null["ci_low"] < 0.0 < null["ci_high"]
    assert null["actual_verdict"] == "refuted"

    missing = by_id["H_missing"]
    assert missing["ci_low"] is None
    assert missing["number_resolved"] is False

    results_path = pathlib.Path(difference["results_path"])
    transcript_path = pathlib.Path(difference["transcript_path"])
    assert results_path.name == RESULTS_FILENAME
    assert transcript_path.name == TRANSCRIPT_FILENAME
    table = pl.read_parquet(results_path)
    assert table.height == 3
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert "evaluation_finding_create" in transcript["called_tools"]

    audit_path = run_dir / "analysis" / "audit.jsonl"
    assert audit_path.is_file()
    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert entries
    by_name = {t.name: t for t in domain.tools}
    for tool_name in set(called_tools):
        tool_obj = by_name[tool_name]
        if not tool_obj.annotations.read_only_hint:
            expected_calls = called_tools.count(tool_name)
            actual_logged = sum(1 for e in entries if e["tool"] == tool_name)
            assert actual_logged == expected_calls, f"audit mismatch for {tool_name}: logged {actual_logged}, called {expected_calls}"
    for entry in entries:
        role_name = entry["role"]
        tool_name = entry["tool"]
        assert role_name != "operator", f"operator-only tool called: {entry}"
        assert tool_name in roles.ROLES[role_name]["tools"], f"{tool_name} not in {role_name} allowlist"
        if by_name[tool_name].annotations.destructive_hint:
            assert tool_name in roles.AGENT_AREA_TOOLS, f"destructive tool {tool_name} outside AGENT_AREA_TOOLS"


def test_h6_planted_refusals_are_enforced_by_mcp_server(
    tmp_path: pathlib.Path,
) -> None:
    """Missing structured numbers, self-verification, and unverified attach are refused; role allowlists exclude operator tools."""
    run_dir = tmp_path / "data" / "benchmarks" / "h6_refusals"
    write_noisy_benchmark(run_dir)
    (run_dir / "truth.json").unlink()

    server = MCPServer("test-refusals")
    domain.install(server)

    async def exercise() -> None:
        cmp_res = await server.call_tool(
            "evaluation_analysis",
            {
                "procedure": "compare",
                "run_id": "h6_refusals",
                "params": {
                    "planner_a": "planner_a",
                    "planner_b": "planner_b",
                    "metric": "time_to_goal",
                },
                "save_as": "refusal_cmp",
            },
        )
        ev_id = str(cmp_res.structured_content["evidence_id"])

        with pytest.raises(MCPToolError, match="not found in evidence"):
            await server.call_tool(
                "evaluation_finding_create",
                {
                    "run_id": "h6_refusals",
                    "claim": "bogus claim",
                    "evidence_ids": [ev_id],
                    "structured_numbers": [{"evidence_id": ev_id, "column": "mean_diff", "value": 99999.0}],
                    "author": "analyst",
                },
            )

        created = await server.call_tool(
            "evaluation_finding_create",
            {
                "run_id": "h6_refusals",
                "claim": "valid claim",
                "evidence_ids": [ev_id],
                "author": "analyst",
            },
        )
        fid = str(created.structured_content["finding_id"])

        (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
        with pytest.raises(MCPToolError, match="only a verified finding"):
            await server.call_tool(
                "evaluation_finding_attach",
                {"run_id": "h6_refusals", "finding_id": fid},
            )

        await server.call_tool(
            "evaluation_finding_verify",
            {"run_id": "h6_refusals", "finding_id": fid},
        )
        with pytest.raises(MCPToolError, match="cannot verify its own finding"):
            await server.call_tool(
                "evaluation_finding_status",
                {
                    "run_id": "h6_refusals",
                    "finding_id": fid,
                    "status": "verified",
                    "by": "analyst",
                },
            )

    with arena_data(tmp_path):
        asyncio.run(exercise())

    assert "evaluation_benchmark" not in roles.ROLES["verifier"]["tools"]
    assert "evaluation_finding_status" not in roles.ROLES["analyst"]["tools"]
