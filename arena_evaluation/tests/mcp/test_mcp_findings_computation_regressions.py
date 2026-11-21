"""Regression tests for verifying findings on computation evidence (L3-021)."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import run_computation
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.mcp.findings import (
    evaluation_finding_create,
    evaluation_finding_verify,
)

from mcp_fixtures import RUN_ID, arena_data, build_runs


@pytest.fixture
def run_dir(tmp_path):
    path = build_runs(tmp_path / "data" / "benchmarks")
    assert path.name == RUN_ID, path.name
    with arena_data(tmp_path):
        yield path


def test_verify_reruns_evidence_from_a_computation(run_dir):
    """A finding on computation evidence verifies by rerunning that computation."""
    result = run_computation(
        "group_summary",
        BenchmarkCatalog.open(run_dir),
        {"group_cols": ["planner"], "metrics": ["time_to_goal"]},
        evidence_store=EvidenceStore(run_dir),
        purpose="planner means",
    )
    finding = evaluation_finding_create(RUN_ID, "planner means are recorded", evidence_id=result.evidence_id)

    checked = evaluation_finding_verify(RUN_ID, finding.finding_id)

    assert checked["mechanical_passed"] is True, checked["checks"]
