"""Regression tests for free-form numbers on findings."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.findings import evaluation_finding_create

from mcp_fixtures import RUN_ID, arena_data, build_runs


@pytest.fixture
def run_dir(tmp_path):
    path = build_runs(tmp_path / "data" / "benchmarks")
    assert path.name == RUN_ID, path.name
    with arena_data(tmp_path):
        yield path


def test_free_form_numbers_are_refused_with_the_structured_alternative(run_dir):
    """A number quoted without evidence is refused at creation and points at structured_numbers."""
    with pytest.raises(ToolError, match="structured_numbers") as refused:
        evaluation_finding_create(
            RUN_ID,
            "the minimum clearance was 0.42 m",
            numbers={"min_clearance_m": 0.42},
        )

    assert "structured_numbers" in (refused.value.hint or "")
