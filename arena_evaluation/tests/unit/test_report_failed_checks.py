"""Unit tests verifying failed/empty plots are recorded as failed PlotChecks."""

from __future__ import annotations

import pathlib

from star_benchmark import write_benchmark

from arena_evaluation.presentation.checks import CheckStage
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    VizManifest,
)


def test_report_builder_records_failed_checks_in_appendix(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_fail"
    write_benchmark(
        bench_dir,
        [
            {"episode_id": 0, "planner": "dwb", "success": True, "time_to_goal": 10.0},
            {"episode_id": 1, "planner": "teb", "success": True, "time_to_goal": 9.0},
        ],
    )
    out_dir = tmp_path / "out_fail"
    manifest = VizManifest(
        name="failure_appendix",
        title="Failure Appendix Report",
        plots=[
            PlotSpec(
                id="valid_plot",
                type="box",
                title="Valid Plot",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
            PlotSpec(
                id="empty_plot",
                type="bar",
                title="Empty Filtered Plot",
                data={
                    "source": "v_episode_summary",
                    "where": "planner == '__nonexistent__'",
                },
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
            PlotSpec(
                id="missing_col_plot",
                type="bar",
                title="Missing Column Plot",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="nonexistent_metric"),
            ),
        ],
    )
    builder = ReportBuilder(bench_dir, output_dir=out_dir, manifest=manifest)
    report_path = builder.build()

    assert report_path is not None and report_path.exists()
    checks = builder.build_checks
    assert checks.plots_total == 3
    assert checks.passed_count == 1
    assert checks.failed_count == 2
    assert any(c.stage == CheckStage.QUALITY for c in checks.for_plot("empty_plot"))
    assert checks.for_plot("missing_col_plot")

    html = report_path.read_text(encoding="utf-8")
    assert "Appendix: Failed Checks" in html
    assert "empty_plot" in html
    assert "missing_col_plot" in html
