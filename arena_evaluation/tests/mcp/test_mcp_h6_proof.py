"""Contract proof tests for PR5/H6: scripted hypothesis-to-report loop over MCPServer."""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest


from agent_eval import evaluate_hypotheses
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp_fixtures import arena_data
from synthetic_benchmark import write_synthetic_benchmark

from arena_evaluation.mcp import domain, roles


def test_h6_scripted_truth_loop_verifies_refutes_and_flags_missing(
    tmp_path: pathlib.Path,
) -> None:
    """Every hypothesis in truth.json reaches its expected stored verdict over MCPServer."""
    run_dir = tmp_path / "data" / "benchmarks" / "h6_proof"
    write_synthetic_benchmark(run_dir, include_swap=False)

    with arena_data(tmp_path):
        results, called_tools = evaluate_hypotheses("h6_proof", run_dir / "truth.json")

    assert len(results) == 3
    for row in results:
        assert row["actual_verdict"] == row["expected_verdict"], row
        assert row["mechanical_passed"] is True
        assert row["evidence_id"]
        assert pathlib.Path(row["report_path"]).is_file()

    audit_path = run_dir / "analysis" / "audit.jsonl"
    assert audit_path.is_file()
    entries = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert entries
    by_name = {t.name: t for t in domain.tools}
    for tool_name in set(called_tools):
        tool_obj = by_name[tool_name]
        if not tool_obj.annotations.read_only_hint:
            expected_calls = called_tools.count(tool_name)
            actual_logged = sum(1 for e in entries if e["tool"] == tool_name)
            assert actual_logged == expected_calls, (
                f"audit mismatch for {tool_name}: logged {actual_logged}, called {expected_calls}"
            )
    for entry in entries:
        role_name = entry["role"]
        tool_name = entry["tool"]
        assert role_name != "operator", f"operator-only tool called: {entry}"
        assert tool_name in roles.ROLES[role_name]["tools"], (
            f"{tool_name} not in {role_name} allowlist"
        )
        if by_name[tool_name].annotations.destructive_hint:
            assert tool_name in roles.AGENT_AREA_TOOLS, (
                f"destructive tool {tool_name} outside AGENT_AREA_TOOLS"
            )


def test_h6_planted_refusals_are_enforced_by_mcp_server(
    tmp_path: pathlib.Path,
) -> None:
    """Missing structured numbers, self-verification, and unverified attach are refused; role allowlists exclude operator tools."""
    run_dir = tmp_path / "data" / "benchmarks" / "h6_refusals"
    write_synthetic_benchmark(run_dir, include_swap=False)

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

        # 1. Unfounded structured number must be refused
        with pytest.raises(MCPToolError, match="not found in evidence"):
            await server.call_tool(
                "evaluation_finding_create",
                {
                    "run_id": "h6_refusals",
                    "claim": "bogus claim",
                    "evidence_ids": [ev_id],
                    "structured_numbers": [
                        {"evidence_id": ev_id, "column": "mean_diff", "value": 99999.0}
                    ],
                    "author": "analyst",
                },
            )

        # Create valid proposed finding
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

        # 2. Attaching unverified finding must be refused
        (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
        with pytest.raises(MCPToolError, match="only a verified finding"):
            await server.call_tool(
                "evaluation_finding_attach",
                {"run_id": "h6_refusals", "finding_id": fid},
            )

        # 3. Self-verification by author role must be refused
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

    # 4. Allowlist check: non-operator roles exclude operator and cross-role tools
    assert "evaluation_benchmark" not in roles.ROLES["verifier"]["tools"]
    assert "evaluation_finding_status" not in roles.ROLES["analyst"]["tools"]
