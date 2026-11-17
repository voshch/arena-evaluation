from __future__ import annotations

import shutil

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from arena_evaluation.mcp.runs import evaluation_runs, evaluation_status

from mcp_fixtures import EMPTY_RUN_ID, RUN_ID, arena_data, build_run, build_runs


def test_runs_report_the_folder_name_and_hide_a_copied_folder(tmp_path):
    root = tmp_path / "data" / "benchmarks"
    run_path = build_runs(root)
    shutil.copytree(run_path, root / "zzz-copied")

    with arena_data(tmp_path):
        listed = evaluation_runs()
        copied = evaluation_status("zzz-copied")

    assert [row["run_id"] for row in listed["runs"]] == [RUN_ID, EMPTY_RUN_ID]
    assert copied["run_id"] == "zzz-copied"
    assert copied["manifest_run_id"] == RUN_ID


def test_runs_sort_by_created_at_not_by_folder_name(tmp_path):
    root = tmp_path / "data" / "benchmarks"
    build_runs(root)
    build_run(root, "aaa-newest", created_at="2026-10-20T00:00:00+00:00")

    with arena_data(tmp_path):
        listed = evaluation_runs()

    assert [row["run_id"] for row in listed["runs"]] == [
        "aaa-newest",
        RUN_ID,
        EMPTY_RUN_ID,
    ]
