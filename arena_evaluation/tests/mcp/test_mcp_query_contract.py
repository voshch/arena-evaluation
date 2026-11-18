from __future__ import annotations

import polars as pl
import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_mcp.core import ToolError

from arena_evaluation.mcp.query import evaluation_evidence, evaluation_query

from mcp_fixtures import arena_data, build_query_run

RUN_ID = "query-run"


def test_query_caps_the_inline_preview_and_reports_the_total(tmp_path):
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path):
        result = evaluation_query(RUN_ID, "SELECT episode_id FROM dim_episodes")

    assert result["row_count"] == 10_000
    assert result["returned_rows"] == 200
    assert result["truncated"] is True
    assert result["limit"] == 200
    assert result["pin_status"] == "transient"
    assert result["columns"] == ["episode_id"]
    assert len(result["data"]) == 200


def test_query_pins_the_full_result_behind_the_capped_preview(tmp_path):
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path):
        result = evaluation_query(
            RUN_ID,
            "SELECT episode_id FROM dim_episodes",
            pin_evidence=True,
            purpose="ten thousand rows",
        )
        evidence = evaluation_evidence(RUN_ID, result["evidence_id"])

    assert result["pin_status"] == "pinned"
    assert result["returned_rows"] == 200
    assert result["truncated"] is True
    assert evidence["row_count"] == 10_000
    assert evidence["returned_rows"] == 200
    assert evidence["truncated"] is True
    assert evidence["snapshot_id"]
    assert len(evidence["data"]) == 200
    run_dir = tmp_path / "data" / "benchmarks" / RUN_ID
    artifact = run_dir / str(evidence["artifact_path"])
    assert pl.read_parquet(artifact).height == 10_000


def test_query_below_and_above_a_caller_limit(tmp_path):
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)
    query = "SELECT episode_id FROM dim_episodes LIMIT 5"

    with arena_data(tmp_path):
        small = evaluation_query(RUN_ID, query)
        capped = evaluation_query(RUN_ID, query, limit=3)

    assert (small["row_count"], small["returned_rows"], small["truncated"]) == (
        5,
        5,
        False,
    )
    assert (capped["row_count"], capped["returned_rows"], capped["truncated"]) == (
        5,
        3,
        True,
    )


@pytest.mark.parametrize(
    ("sql", "missing", "suggestion"),
    (
        ("SELECT * FROM dim_episodez", "dim_episodez", "dim_episodes"),
        ("SELECT plannr FROM dim_episodes", "plannr", "planner"),
    ),
)
def test_an_unknown_name_names_the_closest_valid_one(
    tmp_path, sql, missing, suggestion
):
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path), pytest.raises(ToolError) as caught:
        evaluation_query(RUN_ID, sql)

    assert missing in caught.value.message
    assert suggestion in caught.value.message
    assert caught.value.hint


def test_query_keeps_the_read_only_guard(tmp_path):
    build_query_run(tmp_path / "data" / "benchmarks", RUN_ID)

    with arena_data(tmp_path), pytest.raises(ToolError, match="Forbidden SQL keyword"):
        evaluation_query(RUN_ID, "DROP TABLE dim_episodes")
