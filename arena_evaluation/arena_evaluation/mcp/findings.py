"""MCP WP5 finding tools: thin gateway over :mod:`arena_evaluation.analytics.findings`."""

from __future__ import annotations

import pathlib
from typing import Annotated, Any, Final, Literal

import pydantic
from arena_mcp.core import ToolError

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.findings import (
    Finding,
    FindingAttachResponse,
    FindingError,
    FindingStatus,
    FindingsResponse,
    create_finding,
    findings_dir,
    findings_lock_path,
    load_finding,
    load_findings,
    save_finding,
    transition_finding_status,
    verify_finding,
)
from arena_evaluation.storage.config_layout import (
    bundle_path,
    package_config_roots,
    run_analysis_root,
)

from .analysis import rerun_sensitivity_params
from .base import (
    RunId,
    append_audit_event,
    assert_safe_analysis_write,
    component,
    domain,
    file_lock,
    run_path,
)
from .runs import FINDING_NOTE_PREFIX, Note, write_notes

__all__ = [
    "Finding",
    "FindingAttachResponse",
    "FindingStatus",
    "FindingsResponse",
    "evaluation_finding_attach",
    "evaluation_finding_create",
    "evaluation_finding_status",
    "evaluation_finding_verify",
    "evaluation_findings",
]

DEFAULT_REPORT: Final[str] = "report.html"


def _note_text(finding: Finding) -> str:
    """One self-contained notes.yaml entry for a finding attached to a report."""
    parts = [finding.claim]
    if finding.numbers:
        formatted = ", ".join(
            f"{key}={value}" for key, value in sorted(finding.numbers.items())
        )
        parts.append(f"numbers: {formatted}")
    if finding.evidence_id:
        parts.append(f"evidence: {finding.evidence_id}")
    elif finding.stored_query:
        parts.append(f"query: {finding.stored_query}")
    if finding.figure_path:
        parts.append(f"figure: {finding.figure_path}")
    return "; ".join(parts)


def _report_token(run_dir: pathlib.Path, report: str) -> str:
    """Resolve the report file or run-local/package manifest name."""
    stem = component(report.removesuffix(".yaml"), "report")
    report_file = run_dir / report
    manifest = run_dir / f"{stem}.yaml"
    run_bundle = bundle_path(run_analysis_root(run_dir), "report", stem)
    if report_file.is_file():
        return report_file.name
    if manifest.is_file():
        return manifest.name
    if run_bundle.is_file():
        return stem
    for root in package_config_roots():
        if bundle_path(root, "report", stem).is_file():
            return stem
    raise ToolError(
        f"run {run_dir.name!r} has no report {report!r}",
        hint="evaluation_report writes report.html; evaluation_create_manifest writes <name>/report.yaml",
    )


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_finding_create(
    run_id: RunId,
    claim: Annotated[
        str, pydantic.Field(description="empirical claim in one sentence")
    ],
    episode_id: Annotated[
        int | None, pydantic.Field(description="episode the claim is about")
    ] = None,
    encounter_id: Annotated[
        str | None, pydantic.Field(description="encounter id from fact_encounters")
    ] = None,
    contact_id: Annotated[
        str | None, pydantic.Field(description="contact id from fact_contacts")
    ] = None,
    pedestrian_id: Annotated[
        str | None, pydantic.Field(description="pedestrian id the claim is about")
    ] = None,
    query: Annotated[
        str | None,
        pydantic.Field(description="SELECT query that establishes the claim"),
    ] = None,
    stored_query: Annotated[
        str | None,
        pydantic.Field(description="stored query name from evaluation_queries"),
    ] = None,
    evidence_id: Annotated[
        str | None,
        pydantic.Field(description="pinned evidence id from evaluation_query"),
    ] = None,
    evidence_ids: Annotated[
        list[str] | None,
        pydantic.Field(description="pinned evidence ids supporting the claim"),
    ] = None,
    numbers: Annotated[
        dict[str, float] | None,
        pydantic.Field(description="named numeric values quoted in the claim"),
    ] = None,
    structured_numbers: Annotated[
        list[dict[str, Any]] | None,
        pydantic.Field(
            description="structured numeric assertions verified against EvidenceStore: [{evidence_id, column, value, tolerance}]"
        ),
    ] = None,
    author: Annotated[
        str, pydantic.Field(description="role authoring the finding")
    ] = "analyst",
    figure_path: Annotated[
        str | None,
        pydantic.Field(description="path to a plot artifact supporting the claim"),
    ] = None,
) -> Finding:
    """Record one empirical claim as a ``proposed`` finding under ``analysis/findings/<id>.yaml``."""
    run_dir = run_path(run_id)
    fdir = findings_dir(run_dir)
    assert_safe_analysis_write(run_dir, fdir / "finding.yaml")
    try:
        finding = create_finding(
            run_dir,
            claim,
            episode_id=episode_id,
            encounter_id=encounter_id,
            contact_id=contact_id,
            pedestrian_id=pedestrian_id,
            query=query,
            stored_query=stored_query,
            evidence_id=evidence_id,
            evidence_ids=evidence_ids,
            numbers=numbers,
            structured_numbers=structured_numbers,
            author=author,
            figure_path=figure_path,
        )
    except FindingError as exc:
        raise ToolError(str(exc), hint=exc.hint or "check finding parameters") from exc
    append_audit_event(
        run_dir,
        role=author or "analyst",
        tool="evaluation_finding_create",
        args={"run_id": run_id, "claim": finding.claim},
        produced_ids=[finding.finding_id],
    )
    return finding


@domain.tool(read_only=True)
def evaluation_findings(run_id: RunId) -> FindingsResponse:
    """Every finding recorded for a run under ``analysis/findings/``."""
    run_dir = run_path(run_id)
    try:
        findings = load_findings(run_dir)
    except FindingError as exc:
        raise ToolError(
            str(exc), hint=exc.hint or "fix or remove corrupt finding YAML"
        ) from exc
    return FindingsResponse(run_id=run_id, n_findings=len(findings), findings=findings)


@domain.tool(read_only=False, destructive=False, idempotent=True)
def evaluation_finding_verify(run_id: RunId, finding_id: str) -> dict[str, Any]:
    """Mechanically verify a finding's evidence IDs and structured numbers against EvidenceStore."""
    run_dir = run_path(run_id)
    try:
        result = verify_finding(
            run_dir,
            finding_id,
            computation_runners={"evaluation_sensitivity": rerun_sensitivity_params},
        )
    except FindingError as exc:
        raise ToolError(
            str(exc), hint=exc.hint or "evaluation_findings lists findings"
        ) from exc
    append_audit_event(
        run_dir,
        role="verifier",
        tool="evaluation_finding_verify",
        args={"run_id": run_id, "finding_id": finding_id},
        produced_ids=[finding_id],
    )
    previous_status = str(result.get("previous_status") or "")
    status = str(result.get("status") or "")
    if previous_status and previous_status != status:
        append_audit_event(
            run_dir,
            role="verifier",
            tool="evaluation_finding_status",
            args={"run_id": run_id, "finding_id": finding_id, "status": status},
            produced_ids=[finding_id],
        )
    return result


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_finding_status(
    run_id: RunId,
    finding_id: Annotated[
        str, pydantic.Field(description="finding to review, from evaluation_findings")
    ],
    status: Annotated[
        Literal["proposed", "verified", "refuted", "needs_work"],
        pydantic.Field(description="new review state"),
    ],
    by: Annotated[
        str, pydantic.Field(description="role adjudicating the finding status")
    ] = "verifier",
    note: Annotated[
        str | None, pydantic.Field(description="optional review note")
    ] = None,
) -> Finding:
    """Move one finding between review states with mechanical and author-separation gates."""
    run_dir = run_path(run_id)
    try:
        reviewed = transition_finding_status(
            run_dir, finding_id, status, by=by, note=note
        )
    except FindingError as exc:
        raise ToolError(
            str(exc), hint=exc.hint or "check finding status transition"
        ) from exc
    append_audit_event(
        run_dir,
        role=by,
        tool="evaluation_finding_status",
        args={"run_id": run_id, "finding_id": finding_id, "status": status},
        produced_ids=[finding_id],
    )
    return reviewed


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_finding_attach(
    run_id: RunId,
    finding_id: Annotated[
        str, pydantic.Field(description="finding to attach, from evaluation_findings")
    ],
    report: Annotated[
        str,
        pydantic.Field(
            description="report.html or the run-local manifest name the finding belongs to"
        ),
    ] = DEFAULT_REPORT,
) -> FindingAttachResponse:
    """Attach a verified finding to a report, locking findings and notes together."""
    run_dir = run_path(run_id)
    report_token = _report_token(run_dir, report)
    notes_path = run_dir / "notes.yaml"
    note_label = f"{FINDING_NOTE_PREFIX}{finding_id}"

    with file_lock(findings_lock_path(run_dir)):
        try:
            finding = load_finding(run_dir, finding_id)
        except FindingError as exc:
            raise ToolError(
                str(exc), hint=exc.hint or "evaluation_findings lists them"
            ) from exc
        if finding.status is not FindingStatus.VERIFIED:
            raise ToolError(
                f"finding {finding_id} is {finding.status.value}, only a verified finding can be attached",
                hint=f"run evaluation_finding_verify and evaluation_finding_status({run_id!r}, {finding_id!r}, 'verified') first",
            )
        if finding.mechanical_passed is not True:
            raise ToolError(
                f"finding {finding_id} has not passed its latest mechanical verification",
                hint=f"run evaluation_finding_verify and evaluation_finding_status({run_id!r}, {finding_id!r}, 'verified') again",
            )
        current_snapshot = BenchmarkCatalog.open(run_dir).snapshot_id
        if finding.snapshot_id != current_snapshot:
            raise ToolError(
                f"finding {finding_id} was verified at snapshot {finding.snapshot_id!r}, the run is at {current_snapshot!r}",
                hint=f"run evaluation_finding_verify and evaluation_finding_status({run_id!r}, {finding_id!r}, 'verified') again",
            )
        attached = finding.model_copy(update={"report": report_token})
        save_finding(run_dir, attached)
        write_notes(
            notes_path,
            [Note(label=note_label, value=_note_text(attached))],
            mode="merge",
        )

    append_audit_event(
        run_dir,
        role="reporter",
        tool="evaluation_finding_attach",
        args={"run_id": run_id, "finding_id": finding_id, "report": report_token},
        produced_ids=[finding_id],
    )
    return FindingAttachResponse(
        run_id=run_id,
        finding=attached,
        report=report_token,
        note_label=note_label,
    )
