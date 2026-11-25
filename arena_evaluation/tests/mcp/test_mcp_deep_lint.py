"""Deep declaration lint: every plot binding is resolved, not only its schema."""

from __future__ import annotations

import os
import pathlib

import pytest

from mcp_fixtures import build_runs

from arena_evaluation.mcp import schema


def _write_plot(root: pathlib.Path, name: str, body: str) -> None:
    bundle = root / "plots" / name
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "plot.yaml").write_text(body, encoding="utf-8")


def _clean_root(root: pathlib.Path) -> None:
    _write_plot(
        root,
        "ok_plot",
        "id: ok_plot\ntitle: Ok\ntype: bar\ndata:\n  source: v_episode_summary\nchannels:\n  x: planner\n  y: time_to_goal\n",
    )


def test_deep_lint_resolves_bindings_and_flags_each_broken_binding(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deep lint passes a runnable plot and fails query, channel, and computation mistakes."""
    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    config_root = tmp_path / "configs"
    _clean_root(config_root)
    monkeypatch.setattr("arena_evaluation.mcp.lint.package_config_roots", lambda: [config_root])

    shallow = schema.evaluation_lint_declarations(run_id=run_dir.name)
    assert shallow["valid"] is True
    assert shallow["plots"] == []
    assert shallow["deep"] is False

    deep = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    assert deep["deep"] is True
    assert deep["valid"] is True
    checked = [plot for plot in deep["plots"] if plot["name"] == "plot 'ok_plot'"]
    assert checked and checked[0]["passed"] is True
    assert checked[0]["row_count"] is not None

    wrong_column_root = tmp_path / "wrong_column"
    _write_plot(
        wrong_column_root,
        "missing_channel",
        "id: missing_channel\ntitle: Missing\ntype: bar\ndata:\n  source: v_episode_summary\nchannels:\n  x: planner\n  y: not_a_column\n",
    )
    monkeypatch.setattr("arena_evaluation.mcp.lint.package_config_roots", lambda: [wrong_column_root])
    missing_channel = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    assert missing_channel["valid"] is False
    verdict = next(plot for plot in missing_channel["plots"] if plot["name"] == "plot 'missing_channel'")
    assert verdict["passed"] is False
    assert "not_a_column" in verdict["reason"]

    broken_query_root = tmp_path / "broken_query"
    _write_plot(
        broken_query_root,
        "unresolved_query",
        "id: unresolved_query\ntitle: Query\ntype: bar\ndata:\n  query: SELECT not_a_column FROM v_episode_summary\nchannels:\n  x: planner\n",
    )
    monkeypatch.setattr("arena_evaluation.mcp.lint.package_config_roots", lambda: [broken_query_root])
    broken_query = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    assert broken_query["valid"] is False
    query_verdict = next(plot for plot in broken_query["plots"] if plot["name"] == "plot 'unresolved_query'")
    assert query_verdict["passed"] is False
    assert "not_a_column" in query_verdict["reason"]

    invented_metric_root = tmp_path / "invented_metric"
    _write_plot(
        invented_metric_root,
        "invented_metric",
        "id: invented_metric\ntitle: Radar\ntype: radar\ndata:\n  computation: radar_profile\n  params:\n    metrics: [invented_metric_xyz]\nchannels:\n  x: metric\n  y: normalized_score\n",
    )
    monkeypatch.setattr(
        "arena_evaluation.mcp.lint.package_config_roots",
        lambda: [invented_metric_root],
    )
    invented = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    assert invented["valid"] is False
    invented_verdict = next(plot for plot in invented["plots"] if plot["name"] == "plot 'invented_metric'")
    assert invented_verdict["passed"] is False
    assert "invented_metric_xyz" in invented_verdict["reason"]


def test_deep_lint_skips_evidence_bindings_instead_of_failing_them(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plot bound to a pinned evidence id is reported as unchecked, not as a false failure."""
    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    config_root = tmp_path / "configs"
    _write_plot(
        config_root,
        "evidence_plot",
        "id: evidence_plot\ntitle: Evidence\ntype: bar\ndata:\n  evidence_id: ev-0123456789ab\nchannels:\n  x: planner\n",
    )
    monkeypatch.setattr("arena_evaluation.mcp.lint.package_config_roots", lambda: [config_root])

    report = schema.evaluation_lint_declarations(run_id=run_dir.name, deep=True)
    assert report["valid"] is True
    verdict = report["plots"][0]
    assert verdict["passed"] is True
    assert verdict["checked"] is False
    assert "evidence" in verdict["reason"]
