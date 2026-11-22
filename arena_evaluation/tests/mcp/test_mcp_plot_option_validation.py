from __future__ import annotations

import os
from pathlib import Path

import pytest

from arena_evaluation.mcp.presentation import evaluation_save_plot_declaration
from arena_mcp.core.errors import ToolError


def _make_run_dir(tmp_path: Path) -> str:
    run_id = "test_plot_validation_run"
    (tmp_path / "benchmarks" / run_id).mkdir(parents=True)
    return run_id


def test_save_plot_declaration_rejects_unknown_option_for_plot_type(tmp_path: Path) -> None:
    prev = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    try:
        run_id = _make_run_dir(tmp_path)
        bad_yaml = (
            "id: bad_box\n"
            "title: Bad Box\n"
            "type: box\n"
            "data:\n"
            "  source: fact_robots\n"
            "channels:\n"
            "  x: planner\n"
            "  y: energy_total_wh\n"
            "options:\n"
            "  not_a_real_option: true\n"
        )
        with pytest.raises(ToolError, match="unknown option.*not_a_real_option"):
            evaluation_save_plot_declaration("bad_box", bad_yaml, run_id=run_id)
    finally:
        if prev is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = prev


def test_save_plot_declaration_rejects_unknown_plot_type(tmp_path: Path) -> None:
    prev = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    try:
        run_id = _make_run_dir(tmp_path)
        bad_yaml = (
            "id: unknown_kind\n"
            "title: Unknown Kind\n"
            "type: nonexistent_chart\n"
            "data:\n"
            "  source: fact_robots\n"
            "channels:\n"
            "  x: planner\n"
            "  y: energy_total_wh\n"
        )
        with pytest.raises(ToolError, match="unknown plot type.*nonexistent_chart"):
            evaluation_save_plot_declaration("unknown_kind", bad_yaml, run_id=run_id)
    finally:
        if prev is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = prev


def test_save_plot_declaration_accepts_supported_box_option(tmp_path: Path) -> None:
    prev = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    try:
        run_id = _make_run_dir(tmp_path)
        good_yaml = (
            "id: valid_box\n"
            "title: Valid Box\n"
            "type: box\n"
            "data:\n"
            "  source: fact_robots\n"
            "channels:\n"
            "  x: planner\n"
            "  y: energy_total_wh\n"
            "options:\n"
            "  show_outliers: false\n"
        )
        res = evaluation_save_plot_declaration("valid_box", good_yaml, run_id=run_id)
        assert res["status"] == "saved"
        assert res["type"] == "box"
    finally:
        if prev is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = prev
