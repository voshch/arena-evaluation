"""Contract tests for PR5/H5: reviewed report loop, new plot types, appendix failures, and exports."""

from __future__ import annotations

import base64
import io
import os
import pathlib

import polars as pl
import pytest
import yaml
from PIL import Image
from mcp.types import ImageContent
from report_page import parse_page
from synthetic_benchmark import write_synthetic_benchmark

from arena_evaluation.mcp import analysis, findings, query, reporting
from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY


def test_h5_new_plot_types_registered() -> None:
    """All four analysis plot types (interval, rank_swap, trajectory_slice, breakdown) are registered."""
    for ptype in ("interval", "rank_swap", "trajectory_slice", "breakdown"):
        assert ptype in RENDERER_REGISTRY


def test_h5_draft_plot_view_review_assemble_with_exports_and_appendix(
    tmp_path: pathlib.Path,
) -> None:
    """Full reviewed report loop: approved plots + verified findings assemble; failed/dropped go to appendix; SVG/PDF/LaTeX exports have sidecars."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h5_run"
    write_synthetic_benchmark(run_dir, include_swap=True)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    # Pin query & create verified finding
    q_res = query.evaluation_query(
        run_id="h5_run",
        query="SELECT planner, AVG(time_to_goal) AS mean_ttg FROM v_episode_summary GROUP BY planner ORDER BY planner",
        pin_evidence=True,
        save_as="planner_ttg",
    )
    ev_id = q_res["evidence_id"]
    pa_val = next(r["mean_ttg"] for r in q_res["data"] if r["planner"] == "planner_a")
    f_created = findings.evaluation_finding_create(
        run_id="h5_run",
        claim="planner_a has mean_ttg recorded in derived_planner_ttg",
        evidence_ids=[ev_id],
        structured_numbers=[{"evidence_id": ev_id, "column": "mean_ttg", "value": float(pa_val)}],
        author="analyst",
    )
    findings.evaluation_finding_verify(run_id="h5_run", finding_id=f_created.finding_id)
    findings.evaluation_finding_status(
        run_id="h5_run",
        finding_id=f_created.finding_id,
        status="verified",
        by="verifier",
    )

    # Also leave an unverified proposed finding (must NOT appear in assembled report)
    findings.evaluation_finding_create(
        run_id="h5_run",
        claim="Unverified claim",
        evidence_ids=[ev_id],
        structured_numbers=[{"evidence_id": ev_id, "column": "mean_ttg", "value": float(pa_val)}],
        author="analyst",
    )

    # Store the comparison frame and its interval summary as derived tables
    comparison = analysis.evaluation_analysis(
        procedure="compare",
        run_id="h5_run",
        params={
            "planner_a": "planner_a",
            "planner_b": "planner_b",
            "metric": "time_to_goal",
        },
        save_as="cmp_ab",
    )
    assert comparison["derived_summary_table"] == "derived_cmp_ab_summary"

    # 1. Draft
    draft = reporting.evaluation_report_draft(run_id="h5_run", title="H5 Reviewed Report")
    draft_id = draft["draft_id"]

    # 2. Plot 1: interval (approved)
    p1 = reporting.evaluation_report_plot(
        run_id="h5_run",
        draft_id=draft_id,
        plot={
            "id": "cmp_interval",
            "title": "Planner A vs B",
            "type": "interval",
            "data": {"source": "derived_cmp_ab_summary"},
            "channels": {"x": "metric", "y": "mean_diff"},
            "options": {"low": "ci_low", "high": "ci_high"},
        },
    )
    assert p1["status"] == "rendered", p1

    # View returns ImageContent + check report
    view_res = reporting.evaluation_report_view(run_id="h5_run", draft_id=draft_id, plot_id="cmp_interval")
    assert len(view_res.content) >= 2

    img_blocks = [c for c in view_res.content if isinstance(c, ImageContent)]
    assert img_blocks, "expected ImageContent block"
    im = Image.open(io.BytesIO(base64.b64decode(img_blocks[0].data)))
    colors = im.convert("RGB").getcolors(maxcolors=256)
    assert colors is None or len(colors) > 1, "rendered PNG is blank"

    reporting.evaluation_report_review(
        run_id="h5_run",
        draft_id=draft_id,
        plot_id="cmp_interval",
        verdict="approved",
        reason="Clear separation",
        by="reporter",
    )

    # 3. Plot 2: dropped plot
    reporting.evaluation_report_plot(
        run_id="h5_run",
        draft_id=draft_id,
        plot={
            "id": "dropped_bar",
            "title": "Dropped Bar",
            "type": "bar",
            "data": {"source": "derived_planner_ttg"},
            "channels": {"x": "planner", "y": "mean_ttg"},
        },
    )
    reporting.evaluation_report_review(
        run_id="h5_run",
        draft_id=draft_id,
        plot_id="dropped_bar",
        verdict="dropped",
        reason="Redundant with interval",
        by="reporter",
    )

    # 4. Plot 3: bound to missing column -> failed check, appendix entry
    p3 = reporting.evaluation_report_plot(
        run_id="h5_run",
        draft_id=draft_id,
        plot={
            "id": "broken_col",
            "title": "Broken Column",
            "type": "bar",
            "data": {"source": "derived_planner_ttg"},
            "channels": {"x": "planner", "y": "nonexistent_col_xyz"},
        },
    )
    assert p3["status"] == "failed"

    # 5. Assemble with SVG, PDF, LaTeX exports
    assembled = reporting.evaluation_report_assemble(
        run_id="h5_run",
        draft_id=draft_id,
        name="h5_final",
        exports=["svg", "pdf", "latex"],
    )
    assert assembled["plots_used"] == ["cmp_interval"]
    assert assembled["plots_failed"] == ["broken_col"]
    assert assembled["plots_dropped"] == ["dropped_bar"]
    assert assembled["findings_used"] == [f_created.finding_id]

    html_path = pathlib.Path(assembled["report_path"])
    assert html_path.is_file()
    html_text = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert f_created.finding_id in html_text
    assert "Unverified claim" not in html_text
    assert "broken_col" in html_text
    assert "dropped_bar" in html_text

    # Check exports and sidecars
    exports = assembled["export_paths"]
    assert pathlib.Path(exports["svg"][0]).is_file()
    assert pathlib.Path(exports["svg"][0] + ".meta.yaml").is_file()
    assert pathlib.Path(exports["pdf"][0]).is_file()
    assert pathlib.Path(exports["pdf"][0] + ".meta.yaml").is_file()
    assert pathlib.Path(exports["latex"][0]).is_file()
    assert pathlib.Path(exports["latex"][0] + ".meta.yaml").is_file()


def test_h5_rejects_analysis_prefix_and_preserves_renderer_diagnostics(
    tmp_path: pathlib.Path,
) -> None:
    """validate_data_source rejects 'analysis:' and renderers propagate notices into PlotDiagnostics."""
    from arena_mcp.core import ToolError
    from arena_evaluation.presentation.data_sources import validate_data_source
    from arena_evaluation.mcp import presentation

    with pytest.raises(ValueError, match="unknown data_source"):
        validate_data_source("analysis:compare")

    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h5_diag"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    draft = reporting.evaluation_report_draft(run_id="h5_diag", title="Diag Draft")
    plot_res = reporting.evaluation_report_plot(
        run_id="h5_diag",
        draft_id=draft["draft_id"],
        plot={
            "id": "acoustic_diag",
            "title": "Acoustic",
            "type": "acoustic_field",
            "data": {"source": "v_episode_summary"},
            "channels": {"x": "planner", "y": "time_to_goal"},
        },
    )
    assert plot_res["status"] == "failed"
    assert plot_res["error"]
    assert plot_res["checks"]["issues"]

    plot_dir = run_dir / "analysis" / "plots" / "acoustic_diag"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "plot.yaml").write_text(
        "id: acoustic_diag\ntitle: Acoustic\ntype: acoustic_field\ndata:\n  source: v_episode_summary\nchannels:\n  x: planner\n  y: time_to_goal\n",
        encoding="utf-8",
    )
    with pytest.raises(ToolError) as exc_info:
        presentation.evaluation_render_plot(run_id="h5_diag", declaration="acoustic_diag", output_format="png")
    assert plot_res["checks"]["issues"][0] in str(exc_info.value)


def test_h5_reviewed_report_embeds_the_bundle_and_both_plot_renders(
    tmp_path: pathlib.Path,
) -> None:
    """The reviewed report carries one inline bundle, one Plotly figure and PNG per plot."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h5_blocks"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    query.evaluation_query(
        run_id="h5_blocks",
        query=("SELECT planner, AVG(time_to_goal) AS mean_ttg FROM v_episode_summary GROUP BY planner ORDER BY planner"),
        pin_evidence=True,
        save_as="planner_ttg",
    )
    draft_id = reporting.evaluation_report_draft(run_id="h5_blocks", title="plot blocks")["draft_id"]
    for plot_id, plot_type in (("bar_ttg", "bar"), ("box_ttg", "box")):
        shown = reporting.evaluation_report_plot(
            run_id="h5_blocks",
            draft_id=draft_id,
            plot={
                "id": plot_id,
                "title": plot_id,
                "type": plot_type,
                "data": {"source": "derived_planner_ttg"},
                "channels": {"x": "planner", "y": "mean_ttg"},
            },
        )
        assert shown["status"] == "rendered", shown
        assert shown["html_path"]
        assert pathlib.Path(shown["html_path"]).is_file()
        reporting.evaluation_report_review(
            run_id="h5_blocks",
            draft_id=draft_id,
            plot_id=plot_id,
            verdict="approved",
            reason="plot blocks",
            by="reporter",
        )

    failed = reporting.evaluation_report_plot(
        run_id="h5_blocks",
        draft_id=draft_id,
        plot={
            "id": "bar_plotly_fails",
            "title": "Plotly Fails",
            "type": "bar",
            "data": {"source": "derived_planner_ttg"},
            "channels": {"x": "planner", "y": "mean_ttg"},
            "options": {"orientation": "diagonal"},
        },
    )
    assert failed["status"] == "failed"

    assembled = reporting.evaluation_report_assemble(run_id="h5_blocks", draft_id=draft_id, name="plot_blocks")
    report_path = pathlib.Path(assembled["report_path"])
    page = report_path.read_text(encoding="utf-8")
    parts = parse_page(page)
    assert parts.bundle_count == 1
    assert parts.plotly_figures == 2
    assert parts.png_images == 2
    assert parts.references == []
    assert not (report_path.parent / "plotly.min.js").exists()
    assert assembled["plots_used"] == ["bar_ttg", "box_ttg"]
    assert assembled["plots_failed"] == ["bar_plotly_fails"]
    assert "bar_plotly_fails" in page


def test_h5_analysis_save_as_stores_and_refreshes_the_summary_table(
    tmp_path: pathlib.Path,
) -> None:
    """save_as stores the frame and the summary; refreshing the table refreshes both."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "h5_summary"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    saved = analysis.evaluation_analysis(
        procedure="compare",
        run_id="h5_summary",
        params={
            "planner_a": "planner_a",
            "planner_b": "planner_b",
            "metric": "time_to_goal",
        },
        save_as="cmp_pair",
    )
    assert saved["derived_table"] == "derived_cmp_pair"
    assert saved["derived_summary_table"] == "derived_cmp_pair_summary"
    listed = {table["table_name"] for table in analysis.evaluation_tables(run_id="h5_summary")["tables"]}
    assert {"derived_cmp_pair", "derived_cmp_pair_summary"} <= listed

    refreshed = analysis.evaluation_refresh_table(run_id="h5_summary", name="cmp_pair")
    assert refreshed["version"] == 2
    derived = run_dir / "analysis" / "derived"
    summary_sidecar = yaml.safe_load((derived / "cmp_pair_summary.yaml").read_text(encoding="utf-8"))
    assert summary_sidecar["version"] == 2
    summary_frame = pl.read_parquet(derived / "cmp_pair_summary.parquet")
    assert "mean_diff" in summary_frame.columns
