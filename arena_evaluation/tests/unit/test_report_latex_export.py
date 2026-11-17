"""LaTeX export: the report's approved tables and figures, escaped and content-checked."""

from __future__ import annotations

import pathlib

from star_benchmark import write_benchmark

from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    VizManifest,
)


def _manifest() -> VizManifest:
    return VizManifest(
        name="latex_report",
        title="LaTeX Report",
        plots=[
            PlotSpec(
                id="table_metrics",
                type="table",
                title="Metrics by planner",
                data={
                    "computation": "group_summary",
                    "params": {
                        "metrics": ["success", "energy_total_wh"],
                        "group_cols": ["planner"],
                    },
                },
                channels=PlotChannels(x="planner", y=["metric", "n", "mean", "std"]),
            ),
            PlotSpec(
                id="bar_success",
                type="bar",
                title="Success by planner",
                data={
                    "computation": "group_summary",
                    "params": {"metrics": ["success"], "group_cols": ["planner"]},
                },
                channels=PlotChannels(x="planner", y="mean"),
            ),
            PlotSpec(
                id="broken_plot",
                type="bar",
                title="Broken",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="not_a_metric"),
            ),
        ],
    )


def test_latex_export_writes_only_approved_blocks(tmp_path: pathlib.Path) -> None:
    bench_dir = tmp_path / "bench"
    write_benchmark(
        bench_dir,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "success": True,
                "energy_total_wh": 12.5,
            },
            {
                "episode_id": 1,
                "planner": "teb",
                "success": False,
                "energy_total_wh": 20.0,
            },
        ],
    )
    builder = ReportBuilder(
        bench_dir, output_dir=tmp_path / "out", manifest=_manifest()
    )
    builder.build()

    exported = builder.export_latex(tmp_path / "report.tex")

    assert exported.path.is_file()
    assert exported.tables == ("table_metrics",)
    assert exported.figures == ("bar_success",)
    tex = exported.path.read_text(encoding="utf-8")
    assert tex.startswith("\\documentclass")
    assert tex.rstrip().endswith("\\end{document}")
    assert "\\usepackage{booktabs}" in tex
    assert "\\toprule" in tex and "\\bottomrule" in tex
    assert "energy\\_total\\_wh" in tex
    assert "energy_total_wh" not in tex
    assert "\\includegraphics" in tex
    assert "bar_success.png" in tex
    assert "\\caption{Success by planner}" in tex
    assert "broken" not in tex
