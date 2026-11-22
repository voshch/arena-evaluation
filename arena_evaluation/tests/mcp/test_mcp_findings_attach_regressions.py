"""Regression tests for attaching findings to the report the assembler writes."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_mcp.core import ToolError

from arena_evaluation.mcp import query, reporting
from arena_evaluation.mcp.findings import (
    evaluation_finding_attach,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_finding_verify,
)

from mcp_fixtures import RUN_ID, arena_data, build_runs

EPISODE_IDS_SQL = "SELECT episode_id FROM dim_episodes"


@pytest.fixture
def run_dir(tmp_path):
    path = build_runs(tmp_path / "data" / "benchmarks")
    assert path.name == RUN_ID, path.name
    with arena_data(tmp_path):
        yield path


def _verified_finding(claim: str, evidence_id: str) -> str:
    finding = evaluation_finding_create(RUN_ID, claim, evidence_ids=[evidence_id])
    evaluation_finding_verify(RUN_ID, finding.finding_id)
    evaluation_finding_status(RUN_ID, finding.finding_id, "verified", by="verifier")
    return finding.finding_id


def test_attach_without_a_report_name_uses_the_report_the_assembler_writes(run_dir):
    """The default attach target is the report evaluation_report_assemble writes."""
    pinned = query.evaluation_query(RUN_ID, EPISODE_IDS_SQL, pin_evidence=True)
    finding_id = _verified_finding("episode ids are recorded", pinned["evidence_id"])
    draft = reporting.evaluation_report_draft(run_id=RUN_ID, title="Attach")
    reporting.evaluation_report_assemble(run_id=RUN_ID, draft_id=draft["draft_id"])

    attached = evaluation_finding_attach(RUN_ID, finding_id)

    assert attached.report == "reviewed_report"


def test_attach_refuses_a_report_that_was_never_assembled(run_dir):
    """Attach does not invent a report file: before assembly it refuses with a hint naming the assembler."""
    pinned = query.evaluation_query(RUN_ID, EPISODE_IDS_SQL, pin_evidence=True)
    finding_id = _verified_finding("episode ids are recorded", pinned["evidence_id"])

    with pytest.raises(ToolError) as refused:
        evaluation_finding_attach(RUN_ID, finding_id)

    assert "evaluation_report_assemble" in (refused.value.hint or "")
