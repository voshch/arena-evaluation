"""P2 regressions: run-list keys, evidence truncation, and hinted save_as errors."""

from __future__ import annotations

import pathlib

import pytest

from arena_mcp.core import ToolError

from mcp_fixtures import arena_data, build_runs

from arena_evaluation.mcp import analysis, query, runs


def test_evaluation_runs_does_not_advertise_ros_domain_id(
    tmp_path: pathlib.Path,
) -> None:
    """The runner never publishes ros_domain_id, so the run list must not advertise it."""
    data_dir = tmp_path / "data"
    build_runs(data_dir / "benchmarks")
    with arena_data(tmp_path):
        listing = runs.evaluation_runs()
    assert listing["runs"]
    for row in listing["runs"]:
        assert "ros_domain_id" not in row


def test_evaluation_evidence_truncates_and_reports_snapshot(
    tmp_path: pathlib.Path,
) -> None:
    """Pinned evidence returns a bounded preview with row counts, truncation and snapshot id."""
    data_dir = tmp_path / "data"
    run = build_runs(data_dir / "benchmarks")
    with arena_data(tmp_path):
        pinned = query.evaluation_query(
            run_id=run.name,
            query="SELECT * FROM v_episode_summary LIMIT 5",
            pin_evidence=True,
        )
        evidence = query.evaluation_evidence(
            run_id=run.name,
            evidence_id=str(pinned["evidence_id"]),
            limit=2,
        )
    assert evidence["row_count"] == 5
    assert evidence["returned_rows"] == 2
    assert evidence["truncated"] is True
    assert evidence["limit"] == 2
    assert evidence["snapshot_id"]
    assert len(evidence["data"]) == 2

    with arena_data(tmp_path):
        listing = query.evaluation_evidence(run_id=run.name)
    assert listing["snapshot_id"]


def test_invalid_save_as_names_are_hinted_tool_errors(tmp_path: pathlib.Path) -> None:
    """A save_as the derived store rejects surfaces as a ToolError with an actionable hint."""
    data_dir = tmp_path / "data"
    run = build_runs(data_dir / "benchmarks")
    with arena_data(tmp_path):
        with pytest.raises(ToolError) as query_error:
            query.evaluation_query(
                run_id=run.name,
                query="SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner",
                save_as="-bad-name",
            )
        assert "save_as" in str(query_error.value)
        assert "not starting with" in str(query_error.value.hint)

        with pytest.raises(ToolError) as analysis_error:
            analysis.evaluation_analysis(
                procedure="triage",
                params={},
                run_id=run.name,
                save_as="-bad-name",
            )
        assert "save_as" in str(analysis_error.value)
        assert "not starting with" in str(analysis_error.value.hint)
