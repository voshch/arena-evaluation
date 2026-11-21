"""Reviewed report loop: draft, plot, view (image+checks), review, and assemble with exports."""

from __future__ import annotations

import base64
import dataclasses
import datetime
import json
import pathlib
import uuid
from typing import Annotated, Any, Final, Literal

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
from arena_evaluation.storage.exceptions import ArenaEvaluationError

_EXPORT_FORMATS: Final[tuple[str, str]] = ("svg", "pdf")
_DRAFT_FILE: Final[str] = "draft.yaml"
_RENDER_FAILURES: Final[tuple[type[BaseException], ...]] = (
    KeyError,
    ValueError,
    OSError,
    RuntimeError,
    ArenaEvaluationError,
    pydantic.ValidationError,
    polars.exceptions.PolarsError,
)


def _draft_dir(run_dir: pathlib.Path, draft_id: str) -> pathlib.Path:
    return run_dir / "analysis" / "report_drafts" / component(draft_id, "draft_id")


def _load_draft(draft_dir: pathlib.Path) -> dict[str, Any]:
    path = draft_dir / _DRAFT_FILE
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


def _write_yaml(path: pathlib.Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    )


def _save_draft(
    run_dir: pathlib.Path, draft_dir: pathlib.Path, data: dict[str, Any]
) -> None:
    path = draft_dir / _DRAFT_FILE
    assert_safe_analysis_write(run_dir, path)
    _write_yaml(path, data)


def _plot_not_found(plot_id: str, draft_id: str) -> ToolError:
    return ToolError(
        f"plot {plot_id!r} not found in draft {draft_id!r}",
        hint="add the plot first with evaluation_report_plot",
    )


def _resolver(run_dir: pathlib.Path, catalog: Any) -> Any:
    """The one DataResolver wiring shared by plot rendering and the export renderers."""
    from arena_evaluation.analytics.evidence import EvidenceStore
    from arena_evaluation.analytics.queries import QueryCatalog
    from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
    from arena_evaluation.presentation.resolver import DataResolver

    return DataResolver(
        catalog=catalog,
        sql_engine=PolarsSQLEngine(catalog),
        evidence_store=EvidenceStore(run_dir),
        query_catalog=QueryCatalog(run_dir=run_dir),
        benchmark_dir=run_dir,
    )


def _derived_table_provenance(
    run_dir: pathlib.Path, source: str | None
) -> dict[str, str]:
    """The snapshot recorded by a ``derived_*`` table's sidecar, for the report card (H-022)."""
    if not source or not source.startswith("derived_"):
        return {}
    stem = component(source.removeprefix("derived_"), "derived table name")
    sidecar = run_dir / "analysis" / "derived" / f"{stem}.yaml"
    raw = (
        yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        if sidecar.is_file()
        else None
    )
    if not isinstance(raw, dict):
        return {}
    return {
        "snapshot_id": str(raw.get("snapshot_id") or ""),
        "table_name": str(raw.get("table_name") or source),
    }


def _render_outcome(
    body: str, out_png: pathlib.Path, plot_id: str, diagnostics: Any
) -> tuple[str, str | None]:
    """The rendered/failed status of one plot, naming the first render defect it finds."""
    from arena_evaluation.presentation.checks import image_is_blank

    issues = "; ".join(diagnostics.issues)
    if not body:
        return (
            "failed",
            issues or f"interactive renderer produced no output for plot {plot_id!r}",
        )
    if not out_png.is_file() or out_png.stat().st_size == 0:
        return (
            "failed",
            issues or f"static renderer produced no output for plot {plot_id!r}",
        )
    if image_is_blank(out_png):
        return "failed", f"static render of plot {plot_id!r} is blank"
    return "rendered", None


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
    from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
    from arena_evaluation.presentation.static_renderer import StaticRenderer
    from arena_evaluation.presentation.viz_manifest import PlotSpec

    run_dir = run_path(run_id)
    ddir = _draft_dir(run_dir, draft_id)
    plots_dir = ddir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    spec = PlotSpec.model_validate(plot)
    if not spec.id:
        raise ToolError(
            "plot.id must be non-empty", hint="set plot['id'] to a short identifier"
        )

    catalog = BenchmarkCatalog.open(run_dir)
    resolver = _resolver(run_dir, catalog)
    out_png = plots_dir / f"{spec.id}.png"
    out_html = plots_dir / f"{spec.id}.html"
    assert_safe_analysis_write(run_dir, out_png)
    assert_safe_analysis_write(run_dir, out_html)

    status = "rendered"
    error_message: str | None = None
    checks: dict[str, Any] = {}
    evidence_id = ""
    snapshot_id = catalog.snapshot_id
    diagnostics: Any = None

    try:
        df, diagnostics = resolver.resolve(spec)
        derived = _derived_table_provenance(
            run_dir, spec.data.source if spec.data is not None else None
        )
        if derived.get("snapshot_id") and not diagnostics.snapshot_id:
            diagnostics.snapshot_id = derived["snapshot_id"]
        evidence_id = diagnostics.evidence_id
        snapshot_id = diagnostics.snapshot_id or snapshot_id
        if diagnostics.status == "FAILED":
            status = "failed"
            error_message = "; ".join(diagnostics.issues) or "quality gate rejected"
        else:
            StaticRenderer(units={}).render(
                spec, df, out_png, run_dir=run_dir, diagnostics=diagnostics
            )
            body = PlotlyRenderer(units={}).render(
                spec, df, run_dir=run_dir, diagnostics=diagnostics
            )
            if isinstance(body, list):
                body = "\n".join(body)
            status, error_message = _render_outcome(body, out_png, spec.id, diagnostics)
            if status == "rendered":
                atomic_write_text(out_html, body)
    except _RENDER_FAILURES as exc:
        status = "failed"
        error_message = str(exc)
        checks = {"issues": [str(exc)]}

    if diagnostics is not None:
        checks = dataclasses.asdict(diagnostics)
        if status == "failed":
            issues = list(checks.get("issues") or ())
            if error_message and error_message not in issues:
                issues.append(error_message)
            checks["issues"] = issues
            if issues:
                error_message = "; ".join(issues)

    image_path = str(out_png) if status == "rendered" and out_png.is_file() else None
    html_path = str(out_html) if status == "rendered" and out_html.is_file() else None

    with file_lock(ddir / _DRAFT_FILE):
        data = _load_draft(ddir)
        data["plots"] = [p for p in data.get("plots", []) if p.get("id") != spec.id]
        data["plots"].append(spec.model_dump(mode="json", exclude_none=True))
        data.setdefault("reviews", {})[spec.id] = {
            "verdict": "failed" if status == "failed" else "pending",
            "status": status,
            "error": error_message,
            "reason": error_message or "",
            "note": "",
            "by": "",
            "image_path": image_path,
            "html_path": html_path,
            "checks": checks,
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
        "image_path": image_path,
        "html_path": html_path,
        "checks": checks,
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
    review = _load_draft(_draft_dir(run_dir, draft_id)).get("reviews", {}).get(plot_id)
    if not isinstance(review, dict):
        raise _plot_not_found(plot_id, draft_id)

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
    with file_lock(ddir / _DRAFT_FILE):
        data = _load_draft(ddir)
        reviews = data.setdefault("reviews", {})
        if plot_id not in reviews:
            raise _plot_not_found(plot_id, draft_id)
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


def _verified_findings(
    run_dir: pathlib.Path, snapshot_id: str
) -> tuple[list[Finding], list[dict[str, str]], list[dict[str, str]]]:
    """Split findings into used / left out (mechanical, stale) / not used (status)."""
    verified: list[Finding] = []
    left_out: list[dict[str, str]] = []
    not_used: list[dict[str, str]] = []
    for finding in load_findings(run_dir):
        left_out_reason: str | None = None
        if finding.status is not FindingStatus.VERIFIED:
            reason = "not verified"
        elif finding.mechanical_passed is not True:
            reason = left_out_reason = "latest mechanical verification did not pass"
        elif finding.snapshot_id != snapshot_id:
            reason = left_out_reason = (
                f"verified at snapshot {finding.snapshot_id!r}, the run is at {snapshot_id!r}"
            )
        else:
            verified.append(finding)
            continue
        if left_out_reason is not None:
            left_out.append(
                {"finding_id": finding.finding_id, "reason": left_out_reason}
            )
        not_used.append(
            {
                "finding_id": finding.finding_id,
                "status": finding.status.value,
                "reason": reason,
            }
        )
    return verified, left_out, not_used


def _classify_reviews(
    reviews: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
    """Approved / dropped / failed plot ids plus the excluded-plot cards for the appendix."""
    approved: list[str] = []
    dropped: list[str] = []
    failed: list[str] = []
    excluded: list[dict[str, str]] = []
    for pid, rev in reviews.items():
        status: str | None = None
        reason = ""
        if rev.get("verdict") == "dropped":
            status, reason = "dropped", str(rev.get("reason") or "dropped in review")
            dropped.append(pid)
        elif rev.get("status") == "failed":
            status = "failed"
            reason = str(rev.get("error") or rev.get("reason") or "render failed")
            failed.append(pid)
        elif rev.get("verdict") == "approved":
            approved.append(pid)
        if status is not None:
            excluded.append({"plot_id": pid, "status": status, "reason": reason})
    return approved, dropped, failed, excluded


def _write_exports(
    *,
    builder: Any,
    run_dir: pathlib.Path,
    reports_root: pathlib.Path,
    report_name: str,
    catalog: Any,
    plots_used: list[str],
    plots_by_id: dict[str, dict[str, Any]],
    snapshot_id: str,
    findings_used: list[Finding],
    wanted: set[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Render the requested SVG/PDF/LaTeX exports, verify every artifact, and write its sidecar."""
    from arena_evaluation.presentation.static_renderer import StaticRenderer
    from arena_evaluation.presentation.viz_manifest import PlotSpec

    paths: dict[str, list[str]] = {"svg": [], "pdf": [], "latex": []}
    warnings: list[str] = []
    if not wanted:
        return paths, warnings
    exports_dir = reports_root / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    if wanted & set(_EXPORT_FORMATS):
        resolver = _resolver(run_dir, catalog)
        for pid in plots_used:
            spec = PlotSpec.model_validate(plots_by_id[pid])
            try:
                df, diagnostics = resolver.resolve(spec)
            except Exception as exc:  # noqa: BLE001 - every resolver failure is an export warning
                warnings.append(f"{pid}: cannot resolve for export ({exc})")
                continue
            if diagnostics.status == "FAILED":
                warnings.append(
                    f"{pid}: quality gate rejected the export frame "
                    f"({'; '.join(diagnostics.issues)})"
                )
                continue
            for fmt in sorted(wanted & set(_EXPORT_FORMATS)):
                target = exports_dir / f"{pid}.{fmt}"
                try:
                    StaticRenderer(units={}).render(spec, df, target)
                except Exception as exc:  # noqa: BLE001 - a failed export is reported, not fatal
                    warnings.append(f"{pid}: {fmt} export failed ({exc})")
                    continue
                if not target.is_file() or target.stat().st_size == 0:
                    warnings.append(f"{pid}: {fmt} export produced no file")
                    continue
                _write_yaml(
                    pathlib.Path(f"{target}.meta.yaml"),
                    {
                        "plot_id": pid,
                        "format": fmt,
                        "snapshot_id": snapshot_id,
                        "evidence_id": diagnostics.evidence_id,
                        "row_count": diagnostics.row_count,
                    },
                )
                paths[fmt].append(str(target))

    if "latex" in wanted:
        try:
            latex = builder.export_latex(exports_dir / f"{report_name}.tex")
        except RuntimeError as exc:
            warnings.append(f"latex export produced no document ({exc})")
        else:
            _write_yaml(
                pathlib.Path(f"{latex.path}.meta.yaml"),
                {
                    "report": report_name,
                    "snapshot_id": snapshot_id,
                    "tables": list(latex.tables),
                    "figures": list(latex.figures),
                    "findings": [f.finding_id for f in findings_used],
                },
            )
            paths["latex"].append(str(latex.path))

    for fmt in sorted(wanted):
        if not paths.get(fmt):
            warnings.append(f"requested {fmt!r} export produced no artifact")
    return paths, warnings


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_report_assemble(
    run_id: RunId,
    draft_id: str,
    name: str = "reviewed_report",
    exports: list[Literal["svg", "pdf", "latex"]] | None = None,
) -> dict[str, Any]:
    """Build the reviewed report through ReportBuilder: approved plots, findings, appendix, exports."""
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.presentation.report_builder import ReportBuilder
    from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest

    run_dir = run_path(run_id)
    data = _load_draft(_draft_dir(run_dir, draft_id))
    report_name = component(name, "name")
    reports_root = run_dir / "reports" / report_name
    reports_root.mkdir(parents=True, exist_ok=True)
    assert_safe_analysis_write(run_dir, reports_root / "report.html")

    catalog = BenchmarkCatalog.open(run_dir)
    snapshot_id = catalog.snapshot_id
    plots_by_id = {p["id"]: p for p in data.get("plots", []) if "id" in p}
    approved_ids, plots_dropped, draft_failed, excluded_plots = _classify_reviews(
        data.get("reviews", {})
    )

    findings_used, findings_left_out, findings_not_used = _verified_findings(
        run_dir, snapshot_id
    )
    findings_payload = [
        {
            "finding_id": finding.finding_id,
            "status": finding.status.value,
            "claim": finding.claim,
            "evidence": ", ".join(
                finding.evidence_ids
                or ([finding.evidence_id] if finding.evidence_id else [])
            ),
        }
        for finding in findings_used
    ]

    approved_specs = [
        PlotSpec.model_validate(plots_by_id[pid])
        for pid in approved_ids
        if pid in plots_by_id
    ]
    manifest = VizManifest(
        name=report_name,
        title=str(data.get("title") or report_name),
        description="Reviewed report assembled from an approved draft.",
        plots=approved_specs,
    )
    builder = ReportBuilder(
        run_dir,
        output_dir=run_dir,
        manifest=manifest,
        findings=findings_payload,
        findings_not_used=findings_not_used,
        excluded_plots=excluded_plots,
    )
    report_path = builder.build()
    build = builder.build_result
    assert build is not None and report_path is not None

    build_failed = {
        failure.plot_id
        for failure in builder.build_checks.failures
        if failure.plot_id != "summary"
    }
    plots_used = [record.plot_id for record in build.plots if record.passed]
    plots_failed = [
        *draft_failed,
        *(
            pid
            for pid in approved_ids
            if pid in build_failed and pid not in draft_failed
        ),
    ]

    export_paths, export_warnings = _write_exports(
        builder=builder,
        run_dir=run_dir,
        reports_root=reports_root,
        report_name=report_name,
        catalog=catalog,
        plots_used=plots_used,
        plots_by_id=plots_by_id,
        snapshot_id=snapshot_id,
        findings_used=findings_used,
        wanted=set(exports or []),
    )

    append_audit_event(
        run_dir,
        role="reporter",
        tool="evaluation_report_assemble",
        args={"run_id": run_id, "draft_id": draft_id, "name": report_name},
        produced_ids=[report_name],
    )
    return {
        "run_id": run_dir.name,
        "draft_id": draft_id,
        "snapshot_id": snapshot_id,
        "report_path": str(report_path),
        "complete": bool(plots_used),
        "plots_used": plots_used,
        "plots_failed": plots_failed,
        "plots_failed_count": len(plots_failed),
        "plots_dropped": plots_dropped,
        "findings_used": [f.finding_id for f in findings_used],
        "findings_left_out": findings_left_out,
        "findings_not_used": findings_not_used,
        "export_paths": export_paths,
        "export_warnings": export_warnings,
    }
