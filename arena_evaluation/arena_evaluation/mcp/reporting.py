"""Reviewed report loop: draft, plot, view (image+checks), review, and assemble with exports."""

from __future__ import annotations

import base64
import dataclasses
import datetime
import html
import json
import pathlib
import uuid
from typing import Annotated, Any, Literal

import polars.exceptions
import matplotlib

matplotlib.use("Agg")
import pydantic
import yaml
from arena_mcp.core import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from .base import (
    RunId,
    append_audit_event,
    assert_safe_analysis_write,
    atomic_write_text,
    component,
    domain,
    file_lock,
    run_path,
)
from arena_evaluation.analytics.findings import Finding, FindingStatus, load_findings
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.report_plot_block import (
    plot_block,
    plot_block_styles,
    plotly_bundle_script,
)


def _drafts_root(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / "analysis" / "report_drafts"


def _draft_dir(run_dir: pathlib.Path, draft_id: str) -> pathlib.Path:
    return _drafts_root(run_dir) / component(draft_id, "draft_id")


def _load_draft(draft_dir: pathlib.Path) -> dict[str, Any]:
    path = draft_dir / "draft.yaml"
    if not path.is_file():
        raise ToolError(
            f"report draft {draft_dir.name!r} not found",
            hint="evaluation_report_draft creates a new draft",
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ToolError(
            f"corrupt draft.yaml in {draft_dir}",
            hint="create a fresh draft with evaluation_report_draft",
        )
    return data


def _save_draft(
    run_dir: pathlib.Path, draft_dir: pathlib.Path, data: dict[str, Any]
) -> None:
    path = draft_dir / "draft.yaml"
    assert_safe_analysis_write(run_dir, path)
    atomic_write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_report_draft(
    run_id: RunId,
    from_manifest: Annotated[
        str | None,
        pydantic.Field(
            description="Optional existing manifest name to seed plots from"
        ),
    ] = None,
    title: Annotated[
        str, pydantic.Field(description="Report title")
    ] = "Evaluation Report",
) -> dict[str, Any]:
    """Create a new report draft under ``<run>/analysis/report_drafts/<id>/``."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.presentation.viz_manifest import VizManifest
    from arena_evaluation.storage.config_layout import (
        bundle_path,
        package_config_roots,
        run_analysis_root,
    )
    from arena_evaluation.storage.star_schema import has_star_schema

    run_dir = run_path(run_id)
    if not has_star_schema(run_dir):
        raise ToolError(
            f"run {run_id!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )
    catalog = BenchmarkCatalog.open(run_dir)
    draft_id = f"draft-{uuid.uuid4().hex[:8]}"
    ddir = _draft_dir(run_dir, draft_id)
    ddir.mkdir(parents=True, exist_ok=True)

    plots: list[dict[str, Any]] = []
    if from_manifest is not None:
        stem = component(from_manifest.removesuffix(".yaml"), "from_manifest")
        candidates = [
            bundle_path(run_analysis_root(run_dir), "report", stem),
            *(bundle_path(root, "report", stem) for root in package_config_roots()),
        ]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise ToolError(
                f"manifest {from_manifest!r} not found",
                hint="evaluation_manifests lists available report manifests",
            )
        manifest = VizManifest.model_validate(
            yaml.safe_load(found.read_text(encoding="utf-8"))
        )
        plots = [p.model_dump(mode="json", exclude_none=True) for p in manifest.plots]

    data = {
        "draft_id": draft_id,
        "run_id": run_dir.name,
        "snapshot_id": catalog.snapshot_id,
        "title": title,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "plots": plots,
        "reviews": {},
    }
    _save_draft(run_dir, ddir, data)
    append_audit_event(
        run_dir,
        role="reporter",
        tool="evaluation_report_draft",
        args={"run_id": run_id, "from_manifest": from_manifest},
        produced_ids=[draft_id],
    )
    return {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "snapshot_id": catalog.snapshot_id,
        "n_plots": len(plots),
    }


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_report_plot(
    run_id: RunId,
    draft_id: str,
    plot: Annotated[
        dict[str, Any],
        pydantic.Field(
            description="PlotSpec mapping with id, type, title, data, channels"
        ),
    ],
) -> dict[str, Any]:
    """Add or replace one plot in a report draft and render it to PNG and HTML plus checks."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.evidence import EvidenceStore
    from arena_evaluation.analytics.queries import QueryCatalog
    from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
    from arena_evaluation.presentation.static_renderer import StaticRenderer
    from arena_evaluation.presentation.resolver import DataResolver
    from arena_evaluation.presentation.viz_manifest import PlotSpec

    run_dir = run_path(run_id)
    ddir = _draft_dir(run_dir, draft_id)
    plots_dir = ddir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    spec = PlotSpec.model_validate(plot)
    if not spec.id:
        raise ToolError(
            "plot.id must be non-empty",
            hint="set plot['id'] to a short identifier",
        )

    catalog = BenchmarkCatalog.open(run_dir)
    resolver = DataResolver(
        catalog=catalog,
        sql_engine=PolarsSQLEngine(catalog),
        evidence_store=EvidenceStore(run_dir),
        query_catalog=QueryCatalog(run_dir=run_dir),
        benchmark_dir=run_dir,
    )

    out_png = plots_dir / f"{spec.id}.png"
    out_html = plots_dir / f"{spec.id}.html"
    assert_safe_analysis_write(run_dir, out_png)
    assert_safe_analysis_write(run_dir, out_html)

    status = "rendered"
    error_message: str | None = None
    checks_dict: dict[str, Any] = {}
    evidence_id = ""
    snapshot_id = catalog.snapshot_id

    try:
        df, diagnostics = resolver.resolve(spec)
        evidence_id = diagnostics.evidence_id
        snapshot_id = diagnostics.snapshot_id or snapshot_id
        if df.is_empty():
            status = "failed"
            error_message = f"query for plot {spec.id!r} returned 0 rows"
        else:
            StaticRenderer(units={}).render(
                spec, df, out_png, run_dir=run_dir, diagnostics=diagnostics
            )
            plotly_body = PlotlyRenderer(units={}).render(
                spec, df, run_dir=run_dir, diagnostics=diagnostics
            )
            if isinstance(plotly_body, list):
                plotly_body = "\n".join(plotly_body)
            checks_dict = dataclasses.asdict(diagnostics)
            if not plotly_body:
                status = "failed"
                error_message = (
                    "; ".join(diagnostics.issues)
                    or f"interactive renderer produced no output for plot {spec.id!r}"
                )
            elif not out_png.is_file() or out_png.stat().st_size == 0:
                status = "failed"
                error_message = (
                    "; ".join(diagnostics.issues)
                    or f"static renderer produced no output for plot {spec.id!r}"
                )
            else:
                atomic_write_text(out_html, plotly_body)
    except (
        KeyError,
        ValueError,
        OSError,
        RuntimeError,
        pydantic.ValidationError,
        polars.exceptions.PolarsError,
    ) as exc:
        status = "failed"
        error_message = str(exc)

    with file_lock(ddir / "draft.yaml"):
        data = _load_draft(ddir)
        existing_plots = [p for p in data.get("plots", []) if p.get("id") != spec.id]
        existing_plots.append(spec.model_dump(mode="json", exclude_none=True))
        data["plots"] = existing_plots
        data.setdefault("reviews", {})[spec.id] = {
            "verdict": "failed" if status == "failed" else "pending",
            "status": status,
            "error": error_message,
            "reason": error_message or "",
            "note": "",
            "by": "",
            "image_path": str(out_png)
            if out_png.is_file() and status == "rendered"
            else None,
            "html_path": str(out_html)
            if out_html.is_file() and status == "rendered"
            else None,
            "checks": checks_dict,
            "evidence_id": evidence_id,
            "snapshot_id": snapshot_id,
        }
        _save_draft(run_dir, ddir, data)

    append_audit_event(
        run_dir,
        role="reporter",
        tool="evaluation_report_plot",
        args={"run_id": run_id, "draft_id": draft_id, "plot_id": spec.id},
        produced_ids=[spec.id],
    )
    return {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "plot_id": spec.id,
        "status": status,
        "error": error_message,
        "image_path": str(out_png) if status == "rendered" else None,
        "html_path": str(out_html) if status == "rendered" else None,
        "checks": checks_dict,
        "snapshot_id": snapshot_id,
        "evidence_id": evidence_id,
    }


@domain.tool(read_only=True)
def evaluation_report_view(
    run_id: RunId,
    draft_id: str,
    plot_id: str,
) -> CallToolResult:
    """Return the rendered PNG image and structured check report for one draft plot."""
    run_dir = run_path(run_id)
    ddir = _draft_dir(run_dir, draft_id)
    data = _load_draft(ddir)
    review = data.get("reviews", {}).get(plot_id)
    if not isinstance(review, dict):
        raise ToolError(
            f"plot {plot_id!r} not found in draft {draft_id!r}",
            hint="add the plot first with evaluation_report_plot",
        )

    summary_payload = {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "plot_id": plot_id,
        "status": review.get("status"),
        "verdict": review.get("verdict"),
        "error": review.get("error"),
        "checks": review.get("checks", {}),
        "snapshot_id": review.get("snapshot_id", ""),
        "evidence_id": review.get("evidence_id", ""),
    }
    content: list[Any] = [TextContent(type="text", text=json.dumps(summary_payload))]
    img_path_str = review.get("image_path")
    if img_path_str and pathlib.Path(img_path_str).is_file():
        encoded = base64.b64encode(pathlib.Path(img_path_str).read_bytes()).decode(
            "ascii"
        )
        content.append(ImageContent(type="image", data=encoded, mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=summary_payload)


@domain.tool(read_only=False, destructive=False, idempotent=True)
def evaluation_report_review(
    run_id: RunId,
    draft_id: str,
    plot_id: str,
    verdict: Literal["approved", "revise", "dropped"],
    reason: str = "",
    note: str = "",
    by: str = "reporter",
) -> dict[str, Any]:
    """Record a reviewer verdict (``approved``, ``revise``, or ``dropped``) on a draft plot."""
    run_dir = run_path(run_id)
    ddir = _draft_dir(run_dir, draft_id)
    with file_lock(ddir / "draft.yaml"):
        data = _load_draft(ddir)
        reviews = data.setdefault("reviews", {})
        if plot_id not in reviews:
            raise ToolError(
                f"plot {plot_id!r} not found in draft {draft_id!r}",
                hint="add the plot first with evaluation_report_plot",
            )
        if reviews[plot_id].get("status") == "failed" and verdict == "approved":
            raise ToolError(
                f"plot {plot_id!r} failed rendering and cannot be approved",
                hint="fix the plot spec and re-render with evaluation_report_plot",
            )
        reviews[plot_id].update(
            {
                "verdict": verdict,
                "reason": reason,
                "note": note,
                "by": by,
                "reviewed_at": datetime.datetime.now(datetime.UTC).isoformat(
                    timespec="seconds"
                ),
            }
        )
        _save_draft(run_dir, ddir, data)

    append_audit_event(
        run_dir,
        role=by,
        tool="evaluation_report_review",
        args={
            "run_id": run_id,
            "draft_id": draft_id,
            "plot_id": plot_id,
            "verdict": verdict,
        },
        produced_ids=[plot_id],
    )
    return {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "plot_id": plot_id,
        "verdict": verdict,
        "reason": reason,
    }


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_report_assemble(
    run_id: RunId,
    draft_id: str,
    name: str = "reviewed_report",
    exports: list[Literal["svg", "pdf", "latex"]] | None = None,
) -> dict[str, Any]:
    """Assemble self-contained HTML from approved plots and verified findings, with appendix and exports."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.analytics.derived import list_derived_tables
    from arena_evaluation.analytics.evidence import EvidenceStore
    from arena_evaluation.analytics.queries import QueryCatalog
    from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
    from arena_evaluation.presentation.static_renderer import StaticRenderer
    from arena_evaluation.presentation.resolver import DataResolver
    from arena_evaluation.presentation.viz_manifest import PlotSpec

    run_dir = run_path(run_id)
    ddir = _draft_dir(run_dir, draft_id)
    data = _load_draft(ddir)
    out_dir = run_dir / "reports" / component(name, "name")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_safe_analysis_write(run_dir, out_dir / "report.html")

    catalog = BenchmarkCatalog.open(run_dir)
    snapshot_id = catalog.snapshot_id
    reviews = data.get("reviews", {})
    plots_by_id = {p["id"]: p for p in data.get("plots", []) if "id" in p}

    plots_used: list[str] = []
    plots_failed: list[str] = []
    plots_dropped: list[str] = []
    for pid, rev in reviews.items():
        if rev.get("status") == "failed":
            plots_failed.append(pid)
        elif rev.get("verdict") == "approved":
            plots_used.append(pid)
        elif rev.get("verdict") == "dropped":
            plots_dropped.append(pid)

    all_findings = load_findings(run_dir)
    verified_findings: list[Finding] = []
    findings_left_out: list[dict[str, str]] = []
    for finding in all_findings:
        if finding.status is not FindingStatus.VERIFIED:
            continue
        if finding.mechanical_passed is not True:
            findings_left_out.append(
                {
                    "finding_id": finding.finding_id,
                    "reason": "latest mechanical verification did not pass",
                }
            )
        elif finding.snapshot_id != snapshot_id:
            findings_left_out.append(
                {
                    "finding_id": finding.finding_id,
                    "reason": (
                        f"verified at snapshot {finding.snapshot_id!r}, "
                        f"the run is at {snapshot_id!r}"
                    ),
                }
            )
        else:
            verified_findings.append(finding)

    export_paths: dict[str, list[str]] = {"svg": [], "pdf": [], "latex": []}
    wanted_exports = set(exports or [])
    if wanted_exports & {"svg", "pdf"}:
        resolver = DataResolver(
            catalog=catalog,
            sql_engine=PolarsSQLEngine(catalog),
            evidence_store=EvidenceStore(run_dir),
            query_catalog=QueryCatalog(run_dir=run_dir),
            benchmark_dir=run_dir,
        )
        exports_dir = out_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        for pid in plots_used:
            spec = PlotSpec.model_validate(plots_by_id[pid])
            df, diag = resolver.resolve(spec)
            for fmt in sorted(wanted_exports & {"svg", "pdf"}):
                epath = exports_dir / f"{pid}.{fmt}"
                StaticRenderer(units={}).render(spec, df, epath)
                meta_path = pathlib.Path(f"{epath}.meta.yaml")
                atomic_write_text(
                    meta_path,
                    yaml.safe_dump(
                        {
                            "plot_id": pid,
                            "format": fmt,
                            "snapshot_id": snapshot_id,
                            "evidence_id": diag.evidence_id,
                        },
                        sort_keys=False,
                    ),
                )
                export_paths[fmt].append(str(epath))

    if "latex" in wanted_exports:
        exports_dir = out_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        derived = list_derived_tables(run_dir, current_snapshot_id=snapshot_id)
        for tbl in derived[:5]:
            tname = tbl["table_name"]
            df = catalog.get_table(tname).collect().head(20)
            tex_path = exports_dir / f"{tname}.tex"
            header = " & ".join(df.columns) + r" \\"
            body_lines = [
                " & ".join(str(v) for v in row.values()) + r" \\"
                for row in df.to_dicts()
            ]
            tex = "\n".join(
                [
                    r"\begin{tabular}{" + "l" * len(df.columns) + "}",
                    r"\hline",
                    header,
                    r"\hline",
                    *body_lines,
                    r"\hline",
                    r"\end{tabular}",
                ]
            )
            atomic_write_text(tex_path, tex)
            atomic_write_text(
                pathlib.Path(f"{tex_path}.meta.yaml"),
                yaml.safe_dump(
                    {
                        "table_name": tname,
                        "snapshot_id": snapshot_id,
                        "evidence_ids": [
                            f.evidence_id for f in verified_findings if f.evidence_id
                        ],
                    },
                    sort_keys=False,
                ),
            )
            export_paths["latex"].append(str(tex_path))

    # Build self-contained HTML
    sections: list[str] = [
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(str(data.get('title', 'Report')))}</title>",
        "<style>body{font-family:sans-serif;max-width:960px;margin:2rem auto;line-height:1.5} img{max-width:100%} .card{border:1px solid #ddd;padding:1rem;margin:1rem 0}</style>",
        plot_block_styles(),
        plotly_bundle_script(),
        f"</head><body><h1>{html.escape(str(data.get('title', 'Report')))}</h1>",
        f"<p>Run: <code>{html.escape(run_dir.name)}</code> | Snapshot: <code>{html.escape(snapshot_id)}</code></p>",
        "<h2>Verified Findings</h2><ul>",
    ]
    for f in verified_findings:
        evs = ", ".join(f.evidence_ids or ([f.evidence_id] if f.evidence_id else []))
        sections.append(
            f"<li><strong>{html.escape(f.finding_id)}</strong> [{html.escape(f.status.value)}]: "
            f"{html.escape(f.claim)} (evidence: <code>{html.escape(evs)}</code>)</li>"
        )
    sections.append("</ul><h2>Approved Plots</h2>")
    rendered_plots: list[str] = []
    for pid in plots_used:
        rev = reviews[pid]
        img_path = rev.get("image_path")
        html_path = rev.get("html_path")
        if not isinstance(img_path, str) or not isinstance(html_path, str):
            plots_failed.append(pid)
            continue
        image_file = pathlib.Path(img_path)
        fragment_file = pathlib.Path(html_path)
        if not image_file.is_file() or not fragment_file.is_file():
            plots_failed.append(pid)
            continue
        rendered_plots.append(pid)
        block = plot_block(
            plot_id=pid,
            title=str(plots_by_id.get(pid, {}).get("title") or pid),
            interactive_html=fragment_file.read_text(encoding="utf-8"),
            png_path=image_file,
        )
        sections.append(
            f"<div class='card'><h3>{html.escape(pid)}</h3>"
            f"<p>Reviewer: {html.escape(str(rev.get('by') or ''))} | Evidence: <code>{html.escape(str(rev.get('evidence_id') or ''))}</code></p>"
            f"{block}</div>"
        )
    sections.append("<h2>Appendix: Dropped &amp; Failed Plots</h2><ul>")
    for pid in [*plots_dropped, *plots_failed]:
        rev = reviews[pid]
        sections.append(
            f"<li><strong>{html.escape(pid)}</strong> ({html.escape(str(rev.get('verdict') or rev.get('status')))}): "
            f"{html.escape(str(rev.get('reason') or rev.get('error') or ''))}</li>"
        )
    sections.append("</ul></body></html>")

    report_path = out_dir / "report.html"
    atomic_write_text(report_path, "\n".join(sections))
    append_audit_event(
        run_dir,
        role="reporter",
        tool="evaluation_report_assemble",
        args={"run_id": run_id, "draft_id": draft_id, "name": name},
        produced_ids=[name],
    )
    return {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "snapshot_id": snapshot_id,
        "report_path": str(report_path),
        "plots_used": rendered_plots,
        "plots_failed": plots_failed,
        "plots_dropped": plots_dropped,
        "findings_used": [f.finding_id for f in verified_findings],
        "findings_left_out": findings_left_out,
        "export_paths": export_paths,
    }
