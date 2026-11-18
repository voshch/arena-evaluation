"""Unit tests for technical quality-gate checks in presentation/checks.py."""

from __future__ import annotations

import polars as pl

from arena_evaluation.presentation.checks import (
    BuildChecks,
    CheckStage,
    DiagnosticInspector,
    PlotCheck,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_inspector_passes_valid_frame_without_advisory_warnings() -> None:
    spec = PlotSpec(
        id="p_ok",
        type="bar",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="time_to_goal"),
        checks={"require_columns": ["planner", "time_to_goal"], "min_rows": 2},
    )
    # Constant values (zero variance) must still pass technical checks without advisory warnings.
    df = pl.DataFrame({"planner": ["dwb", "teb"], "time_to_goal": [10.0, 10.0]})
    diag = DiagnosticInspector.inspect(df, spec)
    assert diag.status == "PASSED"
    assert diag.issues == []
    assert diag.row_count == 2


def test_inspector_fails_empty_and_missing_required_columns() -> None:
    spec = PlotSpec(
        id="p_bad",
        type="bar",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="time_to_goal"),
        checks={"require_columns": ["planner", "missing_col"], "min_rows": 3},
    )
    diag = DiagnosticInspector.inspect(pl.DataFrame({"planner": []}), spec)
    assert diag.status == "FAILED"
    assert any("0 rows" in issue for issue in diag.issues)
    assert any("missing_col" in issue for issue in diag.issues)
    assert any("min_rows=3" in issue for issue in diag.issues)


def test_build_checks_counts_unique_failed_plots() -> None:
    failures = (
        PlotCheck(plot_id="a", title="A", stage=CheckStage.STATIC, detail="no png"),
        PlotCheck(
            plot_id="a", title="A", stage=CheckStage.INTERACTIVE, detail="no html"
        ),
        PlotCheck(
            plot_id="b", title="B", stage=CheckStage.LOAD, detail="missing source"
        ),
    )
    checks = BuildChecks(plots_total=4, failures=failures)
    assert checks.failed_count == 2
    assert checks.passed_count == 2
    assert len(checks.for_plot("a")) == 2
