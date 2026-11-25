from __future__ import annotations

from pathlib import Path

import pytest

from arena_evaluation.mcp.presentation import evaluation_read_plot_declaration, evaluation_save_plot_declaration
from arena_mcp.core.errors import ToolError


def _make_run_dir(tmp_path: Path) -> str:
    run_id = "test_plot_validation_run"
    (tmp_path / "benchmarks" / run_id).mkdir(parents=True)
    return run_id


def test_save_plot_declaration_rejects_unknown_option_for_plot_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    bad_yaml = "id: bad_box\ntitle: Bad Box\ntype: box\ndata:\n  source: fact_robots\nchannels:\n  x: planner\n  y: energy_total_wh\noptions:\n  not_a_real_option: true\n"
    with pytest.raises(ToolError, match="unknown option.*not_a_real_option"):
        evaluation_save_plot_declaration("bad_box", bad_yaml, run_id=run_id)


def test_save_plot_declaration_rejects_unknown_plot_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    bad_yaml = "id: unknown_kind\ntitle: Unknown Kind\ntype: nonexistent_chart\ndata:\n  source: fact_robots\nchannels:\n  x: planner\n  y: energy_total_wh\n"
    with pytest.raises(ToolError, match="unknown plot type.*nonexistent_chart"):
        evaluation_save_plot_declaration("unknown_kind", bad_yaml, run_id=run_id)


def test_save_plot_declaration_accepts_supported_box_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    good_yaml = "id: valid_box\ntitle: Valid Box\ntype: box\ndata:\n  source: fact_robots\nchannels:\n  x: planner\n  y: energy_total_wh\noptions:\n  show_outliers: false\n"
    res = evaluation_save_plot_declaration("valid_box", good_yaml, run_id=run_id)
    assert res["status"] == "saved"
    assert res["type"] == "box"


def test_save_and_read_plot_declaration_resolve_type_through_uses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    derived_yaml = "uses: scorecard_table\ntitle: Derived Scorecard\noptions:\n  note: derived\n"
    res = evaluation_save_plot_declaration("derived_scorecard", derived_yaml, run_id=run_id)
    assert res["status"] == "saved"
    read = evaluation_read_plot_declaration("derived_scorecard", run_id=run_id)
    assert read["spec"]["uses"] == "scorecard_table"


def test_save_plot_declaration_rejects_option_unknown_to_the_used_recipe_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    derived_yaml = "uses: scorecard_table\noptions:\n  show_outliers: false\n"
    with pytest.raises(ToolError, match="unknown option.*show_outliers.*'table'"):
        evaluation_save_plot_declaration("derived_bad", derived_yaml, run_id=run_id)


def test_save_plot_declaration_rejects_uses_of_a_missing_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    run_id = _make_run_dir(tmp_path)
    with pytest.raises(ToolError, match="not_a_recipe.*not found"):
        evaluation_save_plot_declaration("derived_missing", "uses: not_a_recipe\n", run_id=run_id)
