"""Regression tests for the run-scoped SQL query store and list contracts."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.query import evaluation_queries, evaluation_query_save
from arena_evaluation.storage.config_layout import (
    package_config_roots,
    run_analysis_root,
)

from mcp_fixtures import arena_data, build_query_run

RUN_ID = "query-run"
EPISODE_IDS_SQL = "SELECT episode_id FROM dim_episodes"


def test_a_saved_query_lands_in_its_run_and_is_listed_for_that_run(tmp_path):
    """evaluation_query_save writes under the run's analysis folder, never the package store."""
    run_dir = build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path):
        evaluation_query_save(
            run_id=RUN_ID,
            name="episode_ids",
            description="episode ids of the run",
            sql=EPISODE_IDS_SQL,
        )
        listed = evaluation_queries(run_id=RUN_ID)

    assert (run_analysis_root(run_dir) / "queries" / "episode_ids" / "query.yaml").is_file()
    for root in package_config_roots():
        assert not (root / "queries" / "episode_ids").exists()
    assert "episode_ids" in [q["name"] for q in listed["queries"]]


def test_a_query_save_that_cannot_write_is_a_hinted_tool_error(tmp_path):
    """An analysis folder that cannot be created is a ToolError with a hint, not a raw OSError."""
    run_dir = build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)
    run_analysis_root(run_dir).write_text("a file where the folder should be")

    with arena_data(tmp_path):
        with pytest.raises(ToolError) as refused:
            evaluation_query_save(
                run_id=RUN_ID,
                name="episode_ids",
                description="episode ids of the run",
                sql=EPISODE_IDS_SQL,
            )

    assert refused.value.hint


def test_the_query_list_is_an_object_with_a_named_list(tmp_path):
    """evaluation_queries publishes an object, so it carries structured content like the other reads."""
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path):
        listed = evaluation_queries(run_id=RUN_ID)

    assert isinstance(listed, dict)
    assert isinstance(listed["queries"], list)
