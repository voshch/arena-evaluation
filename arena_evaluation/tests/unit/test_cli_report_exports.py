"""The report CLI exports each approved figure as SVG/PDF and the report as LaTeX."""

from __future__ import annotations

import pathlib

from star_benchmark import write_benchmark

_ROWS: list[dict[str, object]] = [
    {"episode_id": 0, "planner": "dwb", "success": True, "energy_total_wh": 12.5},
    {"episode_id": 1, "planner": "teb", "success": False, "energy_total_wh": 20.0},
]

_MANIFEST = """\
name: export_check
title: Export Check
plots:
  - id: success_by_planner
    title: Success by planner
    type: bar
    data:
      computation: group_summary
      params:
        metrics: [success]
        group_cols: [planner]
    channels:
      x: planner
      y: mean
"""


def test_report_cli_exports_svg_pdf_and_latex(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.cli import _main_impl

    benchmark = tmp_path / "bench"
    write_benchmark(benchmark, _ROWS)
    manifest = tmp_path / "export_check.yaml"
    manifest.write_text(_MANIFEST, encoding="utf-8")
    output_dir = tmp_path / "out"

    rc = _main_impl(
        [
            "report",
            "--benchmark-dir",
            str(benchmark),
            "--output-dir",
            str(output_dir),
            "--report-manifest",
            str(manifest),
            "--export",
            "svg",
            "--export",
            "pdf",
            "--export",
            "latex",
            "--generate-gifs",
        ]
    )

    assert rc == 0
    report_dir = output_dir / "reports" / "export_check"
    assert (report_dir / "plots" / "success_by_planner.png").is_file()
    assert (report_dir / "plots" / "success_by_planner.svg").is_file()
    assert (report_dir / "plots" / "success_by_planner.pdf").is_file()
    latex = (report_dir / "report.tex").read_text(encoding="utf-8")
    assert r"\includegraphics" in latex
    assert "Success by planner" in latex
