"""Regression test: an all-NULL metric is a failed quality check, not a rendered figure."""

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


def test_report_builder_rejects_an_all_null_metric(tmp_path: pathlib.Path) -> None:
    bench_dir = tmp_path / "bench_all_null"
    write_benchmark(
        bench_dir,
        [
            {"episode_id": 0, "planner": "dwb", "success": None},
            {"episode_id": 1, "planner": "teb", "success": None},
        ],
    )
    manifest = VizManifest(
        name="all_null_report",
        title="All NULL Report",
        plots=[
            PlotSpec(
                id="box_all_null",
                type="box",
                title="All NULL box",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
            PlotSpec(
                id="valid_table",
                type="table",
                title="Planner table",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="episode_id"),
            ),
        ],
    )
    builder = ReportBuilder(bench_dir, output_dir=tmp_path / "out", manifest=manifest)

    report_path = builder.build()

    assert report_path is not None and report_path.exists()
    checks = builder.build_checks
    assert checks.plots_total == 2
    assert checks.failed_count == 1
    assert checks.passed_count == 1
    quality = checks.for_plot("box_all_null")
    assert quality and quality[0].stage == CheckStage.QUALITY
    assert "100% NULL" in quality[0].detail
    plots_dir = tmp_path / "out" / "reports" / "all_null_report" / "plots"
    assert not (plots_dir / "box_all_null.png").exists()
    assert (plots_dir / "valid_table.png").exists()

    html = report_path.read_text(encoding="utf-8")
    assert "box_all_null" in html
    assert "100% NULL" in html
