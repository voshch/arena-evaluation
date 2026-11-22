"""Regression tests for re-verifying a finding after its dataset changed."""

from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_evaluation.mcp import query
from arena_evaluation.mcp.findings import (
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_finding_verify,
    evaluation_findings,
)

from mcp_fixtures import RUN_ID, arena_data, build_runs

EPISODE_IDS_SQL = "SELECT episode_id FROM dim_episodes"


@pytest.fixture
def run_dir(tmp_path):
    path = build_runs(tmp_path / "data" / "benchmarks")
    assert path.name == RUN_ID, path.name
    with arena_data(tmp_path):
        yield path


def test_reverifying_after_the_dataset_changed_needs_work(run_dir):
    """A snapshot change fails re-verification and keeps the snapshot the finding was verified at."""
    pinned = query.evaluation_query(RUN_ID, EPISODE_IDS_SQL, pin_evidence=True)
    finding = evaluation_finding_create(RUN_ID, "episode ids are recorded", evidence_ids=[pinned["evidence_id"]])
    evaluation_finding_verify(RUN_ID, finding.finding_id)
    evaluation_finding_status(RUN_ID, finding.finding_id, "verified", by="verifier")
    verified_at = evaluation_findings(RUN_ID).findings[0].snapshot_id

    episode = run_dir / "episodes" / "episode_000" / "metrics" / "fact_robots.parquet"
    pl.read_parquet(episode).with_columns(pl.lit(999.0).alias("energy_total_wh")).write_parquet(episode)

    again = evaluation_finding_verify(RUN_ID, finding.finding_id)
    stored = evaluation_findings(RUN_ID).findings[0]

    assert again["mechanical_passed"] is False
    assert again["status"] == "needs_work"
    assert stored.snapshot_id == verified_at
    change = next(c for c in again["checks"] if c.get("mode") == "snapshot_changed")
    assert change["passed"] is False
