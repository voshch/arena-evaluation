"""Unit tests for ReportBuilder single-run and multi-run report generation."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from report_page import parse_page
from star_benchmark import write_benchmark

from arena_evaluation.presentation.report_builder import (
    ReportBuilder,
    _summary_table_data,
)
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    SummarySpec,
    VizManifest,
)

_ROWS = [
    {
        "episode_id": 0,
        "planner": "dwb",
        "success": True,
        "time_to_goal": 10.0,
        "path_length": 12.0,
    },
    {
        "episode_id": 1,
        "planner": "dwb",
        "success": True,
        "time_to_goal": 12.0,
        "path_length": 13.0,
    },
    {
        "episode_id": 2,
        "planner": "teb",
        "success": False,
        "time_to_goal": 18.0,
        "path_length": 16.0,
    },
    {
        "episode_id": 3,
        "planner": "teb",
        "success": True,
        "time_to_goal": 9.0,
        "path_length": 11.0,
    },
]


def _sample_manifest() -> VizManifest:
    return VizManifest(
        name="builder_test",
        title="Builder Test Report",
        summary_group_by=["planner"],
        summary=[
            SummarySpec(metric="success", label="Success Rate", format="{:.1%}"),
            SummarySpec(
                metric="time_to_goal", label="Time to Goal [s]", format="{:.2f}"
            ),
        ],
        plots=[
            PlotSpec(
                id="planner_success",
                type="bar",
                title="Success Rate by Planner",
                layout_group="overview",
                data={
                    "computation": "group_summary",
                    "params": {"metrics": ["success"], "group_cols": ["planner"]},
                },
                channels=PlotChannels(x="planner", y="mean"),
            ),
            PlotSpec(
                id="episode_ttg",
                type="box",
                title="Time to Goal Distribution",
                layout_group="performance",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
        ],
    )


def test_report_builder_single_run_records_provenance_and_artifacts(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_single"
    write_benchmark(bench_dir, _ROWS)
    out_dir = tmp_path / "out_single"
    builder = ReportBuilder(bench_dir, output_dir=out_dir, manifest=_sample_manifest())
    report_path = builder.build()

    assert report_path is not None and report_path.exists()
    assert report_path.parent == out_dir / "reports" / "builder_test"
    report_html = report_path.read_text(encoding="utf-8")
    assert "plotly-graph-div" in report_html
    assert "plotly.js v" in report_html
    assert not (report_path.parent / "plotly.min.js").exists()
    assert (report_path.parent / "plots" / "planner_success.png").exists()
    assert (report_path.parent / "plots" / "episode_ttg.png").exists()

    assert builder.build_checks.plots_total == 2
    assert builder.build_checks.failed_count == 0
    assert builder.build_checks.passed_count == 2

    result = builder.build_result
    assert result is not None
    assert len(result.plots) == 2
    assert all(rec.passed for rec in result.plots)
    assert any(rec.evidence_id for rec in result.plots)
    assert all(rec.snapshot_id for rec in result.plots)

    html = report_path.read_text(encoding="utf-8")
    assert "Builder Test Report" in html
    assert "Success Rate" in html and "Time to Goal [s]" in html
    assert "100.0%" in html
    assert "evidence:" not in html
    assert "snapshot:" not in html


def test_report_builder_from_dirs_merges_multiple_catalogs(
    tmp_path: pathlib.Path,
) -> None:
    b1 = tmp_path / "bench_a"
    b2 = tmp_path / "bench_b"
    rows_b = [
        {
            **row,
            "run_id": "run_b",
            "time_to_goal": float(row["time_to_goal"]) + 10.0,
            "success": False,
        }
        for row in _ROWS
    ]
    write_benchmark(b1, _ROWS)
    write_benchmark(b2, rows_b)
    out_dir = tmp_path / "out_multi"
    manifest = _sample_manifest()
    manifest.plots[0] = manifest.plots[0].model_copy(
        update={"channels": PlotChannels(x="planner", y="mean", color="run")}
    )
    builder = ReportBuilder.from_dirs([b1, b2], out_dir, manifest=manifest)
    report_path = builder.build()

    assert report_path is not None and report_path.exists()
    assert builder.build_checks.failed_count == 0
    assert builder.build_checks.passed_count == 2
    result = builder.build_result
    assert result is not None
    assert all(len(rec.snapshot_ids) == 2 for rec in result.plots)
    assert len(result.plots[0].evidence_ids) == 2
    assert len(set(result.plots[0].evidence_ids)) == 2
    html = report_path.read_text(encoding="utf-8")
    assert "bench_a" in html and "bench_b" in html


def test_report_builder_records_missing_source_table_in_multirun_summary(
    tmp_path: pathlib.Path,
) -> None:
    b1 = tmp_path / "bench_a"
    b2 = tmp_path / "bench_b"
    write_benchmark(b1, _ROWS)
    write_benchmark(b2, _ROWS)
    pl.DataFrame({"t": [0.0], "x": [1.0]}).write_parquet(
        b1 / "episodes" / "episode_000" / "metrics" / "series_custom.parquet"
    )
    manifest = _sample_manifest().model_copy(
        update={
            "data_source": "v_series_custom",
            "summary": [SummarySpec(metric="x", label="X")],
            "summary_group_by": ["planner"],
        }
    )
    builder = ReportBuilder.from_dirs(
        [b1, b2], tmp_path / "out_missing", manifest=manifest
    )
    builder.build()
    assert any(
        c.plot_id == "summary" and "v_series_custom" in c.detail
        for c in builder.build_checks.failures
    )


def test_report_builder_records_missing_recipe_as_load_failure_and_nonnegative_passed(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_bad_recipe"
    write_benchmark(bench_dir, _ROWS)
    manifest = VizManifest(
        name="bad_recipe_test",
        title="Bad Recipe Test",
        summary=[SummarySpec(metric="nonexistent_metric", label="Missing")],
        summary_group_by=["planner"],
        plots=[
            PlotSpec(
                id="missing_recipe_plot",
                type="bar",
                uses="does_not_exist_recipe",
            )
        ],
    )
    builder = ReportBuilder(
        bench_dir, output_dir=tmp_path / "out_bad_recipe", manifest=manifest
    )
    builder.build()
    assert builder.build_checks.passed_count == 0
    assert any(
        f.plot_id == "missing_recipe_plot" and f.stage.value == "load"
        for f in builder.build_checks.failures
    )


def test_report_builder_summary_uses_manifest_source_and_fails_on_missing_metric(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_sum"
    write_benchmark(bench_dir, _ROWS)

    missing_manifest = VizManifest(
        name="missing_summary_metric",
        title="Missing Summary Metric",
        summary=[SummarySpec(metric="nonexistent_metric", label="Missing")],
        summary_group_by=["planner"],
        plots=[],
    )
    out_missing = tmp_path / "out_missing"
    b_missing = ReportBuilder(
        bench_dir, output_dir=out_missing, manifest=missing_manifest
    )
    b_missing.build()
    # The summary is a page-level feature, not a declared plot: it never inflates
    # failed_count past plots_total, but it is still a reported, non-passing check.
    assert b_missing.build_checks.plots_total == 0
    assert b_missing.build_checks.failed_count == 0
    assert b_missing.build_checks.has_failures is True
    assert b_missing.build_checks.failures[0].plot_id == "summary"
    assert "nonexistent_metric" in b_missing.build_checks.failures[0].detail

    empty_manifest = VizManifest(
        name="no_summary", title="No Summary", summary=[], plots=[]
    )
    out_empty = tmp_path / "out_empty"
    b_empty = ReportBuilder(bench_dir, output_dir=out_empty, manifest=empty_manifest)
    p_empty = b_empty.build()
    assert p_empty is not None
    assert b_empty.build_checks.failed_count == 0
    assert "Performance Summary" not in p_empty.read_text(encoding="utf-8")


def test_report_builder_escapes_html_and_sanitizes_markdown_links(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_xss"
    write_benchmark(bench_dir, _ROWS)
    manifest = VizManifest(
        name="xss_test",
        title="<script>alert('xss')</script>",
        summary=[],
        plots=[
            PlotSpec(
                id="p1",
                type="box",
                title="Box <b>Title</b>",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
                options={
                    "note": "[bad](javascript:alert(1)) and [ok](https://example.com)"
                },
            )
        ],
    )
    out_dir = tmp_path / "out_xss"
    builder = ReportBuilder(bench_dir, output_dir=out_dir, manifest=manifest)
    report_path = builder.build()
    assert report_path is not None
    html = report_path.read_text(encoding="utf-8")
    assert "<script>alert('xss')</script>" not in html
    assert (
        "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;" in html
        or "&lt;script&gt;" in html
    )
    assert 'href="javascript:' not in html
    assert 'href="https://example.com"' in html


def test_report_builder_embeds_bundle_and_static_render_for_every_plot(
    tmp_path: pathlib.Path,
) -> None:
    """One one-shot report: one bundle, one Plotly figure and PNG per plot, no sibling file."""
    bench_dir = tmp_path / "bench_blocks"
    write_benchmark(bench_dir, _ROWS)
    manifest = VizManifest(
        name="blocks_test",
        title="Plot Blocks",
        plots=[
            PlotSpec(
                id="bar_ok",
                type="bar",
                title="Bar",
                data={
                    "computation": "group_summary",
                    "params": {"metrics": ["time_to_goal"], "group_cols": ["planner"]},
                },
                channels=PlotChannels(x="planner", y="mean"),
            ),
            PlotSpec(
                id="box_ok",
                type="box",
                title="Box",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
            PlotSpec(
                id="bar_plotly_fails",
                type="bar",
                title="Plotly Fails",
                data={
                    "computation": "group_summary",
                    "params": {"metrics": ["time_to_goal"], "group_cols": ["planner"]},
                },
                channels=PlotChannels(x="planner", y="mean"),
                options={"orientation": "diagonal"},
            ),
        ],
    )
    builder = ReportBuilder(
        bench_dir, output_dir=tmp_path / "out_blocks", manifest=manifest
    )
    report_path = builder.build()
    assert report_path is not None
    page = report_path.read_text(encoding="utf-8")
    parts = parse_page(page)
    assert parts.bundle_count == 1
    assert parts.plotly_figures == 2
    assert parts.png_images == 2
    assert parts.references == []
    assert not (report_path.parent / "plotly.min.js").exists()
    assert builder.build_checks.failed_count == 1
    assert builder.build_checks.failures[0].plot_id == "bar_plotly_fails"
    assert "bar_plotly_fails" in page


def _summary_frame(pairs: list[tuple[str, str, float]]) -> pl.DataFrame:
    """The long ``group_summary`` shape: one mean per (group, metric)."""
    return pl.DataFrame(
        {
            "planner": [planner for planner, _, _ in pairs],
            "metric": [metric for _, metric, _ in pairs],
            "n": [2] * len(pairs),
            "mean": [mean for _, _, mean in pairs],
            "std": [1.0] * len(pairs),
        }
    )


def test_summary_table_pivots_one_mean_per_group_and_metric() -> None:
    """One row per group, one column per declared metric, in the manifest's metric order."""
    table = _summary_frame(
        [
            ("teb", "success", 0.5),
            ("dwb", "success", 1.0),
            ("teb", "time_to_goal", 18.0),
            ("dwb", "time_to_goal", 11.0),
        ]
    )
    specs = [
        SummarySpec(metric="success", label="Success", format="{:.0%}"),
        SummarySpec(metric="time_to_goal", label="Time [s]", format="{:.1f}"),
    ]

    data = _summary_table_data(table, ["planner"], specs)

    assert data["headers"] == ["planner", "Success", "Time [s]"]
    assert data["rows"] == [["teb", "50%", "18.0"], ["dwb", "100%", "11.0"]]


def test_summary_table_rejects_a_repeated_group_and_metric_cell() -> None:
    """Two means for one cell must be reported, never silently resolved to the last row."""
    table = _summary_frame([("teb", "success", 0.5), ("teb", "success", 0.75)])
    specs = [SummarySpec(metric="success", label="Success", format="{:.0%}")]

    with pytest.raises(ValueError, match=r"repeats 1 \(group, metric\) cell"):
        _summary_table_data(table, ["planner"], specs)
