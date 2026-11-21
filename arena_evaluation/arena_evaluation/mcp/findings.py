"""MCP WP5 finding tools: thin gateway over :mod:`arena_evaluation.analytics.findings`."""

from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Any, Final, Literal

import polars as pl
from arena_mcp.core import ToolError
from pydantic import Field

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import COMPUTATIONS, run_computation
from arena_evaluation.analytics.findings import (
    Finding,
    FindingAttachResponse,
    FindingError,
    FindingsResponse,
    FindingStatus,
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

DEFAULT_REPORT: Final[str] = "reviewed_report"  # the name evaluation_report_assemble writes by default


def _described(kind: Any, description: str) -> Any:  # noqa: ANN401 - runtime Annotated type-alias factory for MCP tool schemas
    """Annotate a tool parameter with its published description."""
    return Annotated[kind, Field(description=description)]


@contextlib.contextmanager
def _finding_errors(hint: str) -> Iterator[None]:
    """Translate a finding-domain error into the tool error contract."""
    try:
        yield
    except FindingError as exc:
        raise ToolError(str(exc), hint=exc.hint or hint) from exc


def _attach_note(finding: Finding) -> Note:
    """One self-contained notes.yaml entry for a finding attached to a report."""
    parts = [finding.claim]
    if finding.structured_numbers:
        formatted = ", ".join(f"{entry['column']}={entry['value']}" for entry in finding.structured_numbers)
        parts.append(f"numbers: {formatted}")
    if finding.evidence_id:
        parts.append(f"evidence: {finding.evidence_id}")
    elif finding.stored_query:
        parts.append(f"query: {finding.stored_query}")
    if finding.figure_path:
        parts.append(f"figure: {finding.figure_path}")
    return Note(label=f"{FINDING_NOTE_PREFIX}{finding.finding_id}", value="; ".join(parts))


def _report_token(run_dir: pathlib.Path, report: str) -> str:
    """Resolve the report file or run-local/package manifest name."""
    stem = component(report.removesuffix(".yaml"), "report")
    named = (
        (run_dir / "reports" / stem / "report.html", stem),
        (run_dir / report, pathlib.Path(report).name),
        (run_dir / f"{stem}.yaml", f"{stem}.yaml"),
        (bundle_path(run_analysis_root(run_dir), "report", stem), stem),
    )
    for candidate, token in named:
        if candidate.is_file():
            return token
    for root in package_config_roots():
        if bundle_path(root, "report", stem).is_file():
            return stem
    raise ToolError(
        f"run {run_dir.name!r} has no report {report!r}",
        hint="evaluation_report_assemble writes reports/<name>/report.html: assemble the draft, then attach; evaluation_create_manifest writes <name>/report.yaml",
    )


ComputationRunner = Callable[[pathlib.Path, Mapping[str, Any]], pl.DataFrame]


def _computation_runners() -> dict[str, ComputationRunner]:
    """Rerun each registered L3 computation, and the sensitivity rerun, keyed by evidence operation."""

    def runner_for(name: str) -> ComputationRunner:
        def rerun(run_dir: pathlib.Path, params: Mapping[str, Any]) -> pl.DataFrame:
            catalog = BenchmarkCatalog.open(run_dir)
            return run_computation(name, catalog, params).summary

        return rerun

    return {
        **{name: runner_for(name) for name in COMPUTATIONS},
        "evaluation_sensitivity": rerun_sensitivity_params,
    }


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_finding_create(
    run_id: RunId,
    claim: _described(str, "empirical claim in one sentence"),
    episode_id: _described(int | None, "episode the claim is about") = None,
    encounter_id: _described(str | None, "encounter id from fact_encounters") = None,
    contact_id: _described(str | None, "contact id from fact_contacts") = None,
    pedestrian_id: _described(str | None, "pedestrian id the claim is about") = None,
    query: _described(str | None, "SELECT query that establishes the claim") = None,
    stored_query: _described(str | None, "stored query name from evaluation_queries") = None,
    evidence_id: _described(str | None, "pinned evidence id from evaluation_query") = None,
    evidence_ids: _described(list[str] | None, "pinned evidence ids supporting the claim") = None,
    numbers: _described(
        dict[str, float] | None,
        "refused: free-form numbers are not checked against evidence; quote them in structured_numbers",
    ) = None,
    structured_numbers: _described(
        list[dict[str, Any]] | None,
        "structured numeric assertions verified against EvidenceStore: [{evidence_id, column, value, tolerance}]",
    ) = None,
    author: _described(str, "role authoring the finding") = "analyst",
    figure_path: _described(str | None, "path to a plot artifact supporting the claim") = None,
) -> Finding:
    """Record one empirical claim as a ``proposed`` finding under ``analysis/findings/<id>.yaml``."""
    run_dir = run_path(run_id)
    fdir = findings_dir(run_dir)
    assert_safe_analysis_write(run_dir, fdir / "finding.yaml")
    with _finding_errors("check finding parameters"):
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
    with _finding_errors("fix or remove corrupt finding YAML"):
        findings = load_findings(run_dir)
    return FindingsResponse(run_id=run_id, n_findings=len(findings), findings=findings)


@domain.tool(read_only=False, destructive=False, idempotent=True)
def evaluation_finding_verify(run_id: RunId, finding_id: str) -> dict[str, Any]:
    """Mechanically verify a finding's evidence IDs and structured numbers against EvidenceStore."""
    run_dir = run_path(run_id)
    with _finding_errors("evaluation_findings lists findings"):
        result = verify_finding(
            run_dir,
            finding_id,
            computation_runners=_computation_runners(),
        )
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
    finding_id: _described(str, "finding to review, from evaluation_findings"),
    status: _described(Literal["proposed", "verified", "refuted", "needs_work"], "new review state"),
    by: _described(str, "role adjudicating the finding status") = "verifier",
    note: _described(str | None, "optional review note") = None,
) -> Finding:
    """Move one finding between review states with mechanical and author-separation gates."""
    run_dir = run_path(run_id)
    with _finding_errors("check finding status transition"):
        reviewed = transition_finding_status(run_dir, finding_id, status, by=by, note=note)
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
    finding_id: _described(str, "finding to attach, from evaluation_findings"),
    report: _described(
        str,
        "assembled report name: reports/<name>/report.html written by evaluation_report_assemble",
    ) = DEFAULT_REPORT,
) -> FindingAttachResponse:
    """Attach a verified finding to a report, locking findings and notes together."""
    run_dir = run_path(run_id)
    notes_path = run_dir / "notes.yaml"
    with (
        file_lock(findings_lock_path(run_dir)),
        _finding_errors("evaluation_findings lists them"),
    ):
        finding = load_finding(run_dir, finding_id)
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
        report_token = _report_token(run_dir, report)
        attached = finding.model_copy(update={"report": report_token})
        save_finding(run_dir, attached)
        note = _attach_note(attached)
        write_notes(notes_path, [note], mode="merge")

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
        note_label=note.label,
    )
