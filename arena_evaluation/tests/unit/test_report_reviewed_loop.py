"""Reviewed report loop after Phase 2: one builder page, real exports, honest appendices."""

from __future__ import annotations

import pathlib

import pytest
from synthetic_benchmark import write_synthetic_benchmark

from arena_evaluation.mcp import findings as findings_tool
from arena_evaluation.mcp import query as query_tool
from arena_evaluation.mcp import reporting


def _run(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "reviewed_run"
    write_synthetic_benchmark(run_dir, include_swap=True)
    monkeypatch.setenv("ARENA_DATA_DIR", str(data_dir))
    return run_dir


def _pinned_query(run_id: str) -> str:
    result = query_tool.evaluation_query(
        run_id=run_id,
        query=(
            "SELECT planner, AVG(time_to_goal) AS mean_ttg FROM v_episode_summary "
            "GROUP BY planner ORDER BY planner"
        ),
        pin_evidence=True,
        save_as="planner_ttg",
    )
    return str(result["evidence_id"])


def _approved_bar(draft_id: str) -> str:
    plot_id = "bar_ttg"
    shown = reporting.evaluation_report_plot(
        run_id="reviewed_run",
        draft_id=draft_id,
        plot={
            "id": plot_id,
            "title": "Mean time to goal",
            "type": "bar",
            "data": {"source": "derived_planner_ttg"},
            "channels": {"x": "planner", "y": "mean_ttg"},
        },
    )
    assert shown["status"] == "rendered", shown
    reporting.evaluation_report_review(
        run_id="reviewed_run",
        draft_id=draft_id,
        plot_id=plot_id,
        verdict="approved",
        reason="clear",
        by="reporter",
    )
    return plot_id


def _failing_plot(draft_id: str) -> str:
    plot_id = "broken_col"
    failed = reporting.evaluation_report_plot(
        run_id="reviewed_run",
        draft_id=draft_id,
        plot={
            "id": plot_id,
            "title": "Broken column",
            "type": "bar",
            "data": {"source": "derived_planner_ttg"},
            "channels": {"x": "planner", "y": "nonexistent_col_xyz"},
        },
    )
    assert failed["status"] == "failed", failed
    return plot_id


def test_failed_and_unverified_items_are_listed_not_dropped(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path, monkeypatch)
    evidence_id = _pinned_query("reviewed_run")
    findings_tool.evaluation_finding_create(
        run_id="reviewed_run",
        claim="A claim nobody verified",
        evidence_ids=[evidence_id],
        author="analyst",
    )
    draft_id = reporting.evaluation_report_draft(
        run_id="reviewed_run", title="Reviewed"
    )["draft_id"]
    _approved_bar(draft_id)
    broken = _failing_plot(draft_id)

    assembled = reporting.evaluation_report_assemble(
        run_id="reviewed_run", draft_id=draft_id, name="honest"
    )

    assert assembled["complete"] is True
    assert assembled["plots_failed"] == [broken]
    assert assembled["plots_failed_count"] == 1
    assert assembled["findings_not_used"], assembled
    statuses = {row["status"] for row in assembled["findings_not_used"]}
    assert "proposed" in statuses
    page = pathlib.Path(assembled["report_path"]).read_text(encoding="utf-8")
    assert broken in page
    assert "Appendix — Excluded Plots and Findings" in page
    assert "A claim nobody verified" not in page


def test_assemble_without_an_approved_plot_is_incomplete(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path, monkeypatch)
    _pinned_query("reviewed_run")
    draft_id = reporting.evaluation_report_draft(
        run_id="reviewed_run", title="Reviewed"
    )["draft_id"]
    _failing_plot(draft_id)

    assembled = reporting.evaluation_report_assemble(
        run_id="reviewed_run", draft_id=draft_id, name="empty", exports=["svg"]
    )

    assert assembled["complete"] is False
    assert assembled["plots_used"] == []
    assert assembled["plots_failed_count"] == 1
    assert any("svg" in warning for warning in assembled["export_warnings"])
    assert pathlib.Path(assembled["report_path"]).is_file()


def test_requested_exports_are_written_and_validated(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(tmp_path, monkeypatch)
    _pinned_query("reviewed_run")
    draft_id = reporting.evaluation_report_draft(
        run_id="reviewed_run", title="Reviewed"
    )["draft_id"]
    _approved_bar(draft_id)

    assembled = reporting.evaluation_report_assemble(
        run_id="reviewed_run",
        draft_id=draft_id,
        name="exported",
        exports=["svg", "pdf", "latex"],
    )

    for fmt in ("svg", "pdf", "latex"):
        artifacts = assembled["export_paths"][fmt]
        assert artifacts, f"{fmt} produced no artifact: {assembled['export_warnings']}"
        target = pathlib.Path(artifacts[0])
        assert target.is_file() and target.stat().st_size > 0
        assert pathlib.Path(f"{target}.meta.yaml").is_file()
    assert assembled["export_warnings"] == []


