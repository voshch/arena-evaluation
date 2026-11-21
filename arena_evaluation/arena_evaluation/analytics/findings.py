"""Layer 3 finding lifecycle: single store under ``analysis/findings/<id>.yaml``."""

from __future__ import annotations

import datetime
import pathlib
import tempfile
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Final

import polars as pl
import pydantic
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore, compute_frame_hash
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.storage.atomic import atomic_write_text, file_lock
from arena_evaluation.storage.config_layout import run_analysis_root

__all__ = [
    "Finding",
    "FindingAttachResponse",
    "FindingError",
    "FindingStatus",
    "FindingsResponse",
    "findings_dir",
    "load_finding",
    "load_findings",
    "save_finding",
    "transition_finding_status",
    "verify_finding",
]

FINDINGS_SUBDIR: Final[str] = "findings"

_ComputationRunners = Mapping[
    str, Callable[[pathlib.Path, Mapping[str, Any]], pl.DataFrame]
]


class FindingError(ValueError):
    """Domain error with an optional remediation hint for MCP callers."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class FindingStatus(StrEnum):
    """Review state of one recorded empirical claim."""

    PROPOSED = "proposed"
    VERIFIED = "verified"
    REFUTED = "refuted"
    NEEDS_WORK = "needs_work"


ALLOWED_TRANSITIONS: Final[dict[FindingStatus, frozenset[FindingStatus]]] = {
    FindingStatus.PROPOSED: frozenset(
        {FindingStatus.VERIFIED, FindingStatus.REFUTED, FindingStatus.NEEDS_WORK}
    ),
    FindingStatus.NEEDS_WORK: frozenset(
        {FindingStatus.PROPOSED, FindingStatus.VERIFIED, FindingStatus.REFUTED}
    ),
    FindingStatus.VERIFIED: frozenset(
        {FindingStatus.REFUTED, FindingStatus.NEEDS_WORK}
    ),
    FindingStatus.REFUTED: frozenset(),
}

#: Status a failed verification forces a finding out of.
_FAILED_VERIFICATION_STATUS: Final[dict[FindingStatus, FindingStatus]] = {
    FindingStatus.VERIFIED: FindingStatus.NEEDS_WORK
}


class Finding(pydantic.BaseModel):
    """One empirical claim, the evidence it rests on, and its review state."""

    finding_id: str
    claim: str
    status: FindingStatus = FindingStatus.PROPOSED
    created_at: str
    episode_id: int | None = None
    encounter_id: str | None = None
    contact_id: str | None = None
    pedestrian_id: str | None = None
    query: str | None = None
    stored_query: str | None = None
    evidence_id: str | None = None
    evidence_ids: list[str] = pydantic.Field(default_factory=list)
    structured_numbers: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    author: str
    figure_path: str | None = None
    report: str | None = None
    note: str | None = None
    mechanical_passed: bool | None = None
    verification_checks: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    snapshot_id: str = ""


class FindingsResponse(pydantic.BaseModel):
    """All findings recorded for one benchmark run."""

    run_id: str
    n_findings: int
    findings: list[Finding]


class FindingAttachResponse(pydantic.BaseModel):
    """Confirmation of a finding attached to a report."""

    run_id: str
    finding: Finding
    report: str
    note_label: str


#: Entry gates of one target status: (violation predicate, message, remediation hint).
_StatusGate = tuple[Callable[[Finding, str], bool], str, str]
_ENTRY_GATES: Final[dict[FindingStatus, tuple[_StatusGate, ...]]] = {
    FindingStatus.VERIFIED: (
        (
            lambda f, by: f.mechanical_passed is not True,
            "finding {finding_id} has not passed mechanical verification",
            "run evaluation_finding_verify first",
        ),
        (
            lambda f, by: by == f.author,
            "author role {author!r} cannot verify its own finding {finding_id}",
            "use a different verifier role",
        ),
    ),
}


def findings_dir(run_dir: pathlib.Path) -> pathlib.Path:
    """Return ``<run>/analysis/findings`` directory."""
    return run_analysis_root(run_dir) / FINDINGS_SUBDIR


def findings_lock_path(run_dir: pathlib.Path) -> pathlib.Path:
    """Return lock file path inside ``<run>/analysis/findings``."""
    return findings_dir(run_dir) / ".findings.lock"


def _finding_file(run_dir: pathlib.Path, finding_id: str) -> pathlib.Path:
    return findings_dir(run_dir) / f"{finding_id}.yaml"


def _read_finding(path: pathlib.Path, *, label: str, hint: str) -> Finding:
    """Parse one finding YAML file, mapping every malformed document to FindingError."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:
        raise FindingError(
            f"corrupt finding file {path.name}: {exc}", hint=hint
        ) from exc
    if not isinstance(raw, dict):
        raise FindingError(f"{label} is not a YAML mapping", hint=hint)
    try:
        return Finding.model_validate(raw)
    except pydantic.ValidationError as exc:
        raise FindingError(
            f"invalid finding schema in {path.name}: {exc}", hint=hint
        ) from exc


def load_findings(run_dir: pathlib.Path) -> list[Finding]:
    """Load all findings from ``<run>/analysis/findings/*.yaml`` sorted by filename."""
    fdir = findings_dir(run_dir)
    if not fdir.is_dir():
        return []
    return [
        _read_finding(
            path, label=f"finding file {path.name}", hint=f"fix or remove {path}"
        )
        for path in sorted(fdir.glob("*.yaml"))
    ]


def load_finding(run_dir: pathlib.Path, finding_id: str) -> Finding:
    """Load one finding by id, raising FindingError when absent."""
    path = _finding_file(run_dir, finding_id)
    if not path.is_file():
        raise FindingError(
            f"no finding {finding_id!r} in this run",
            hint="evaluation_findings lists them",
        )
    return _read_finding(
        path,
        label=f"finding {finding_id!r}",
        hint="re-create the finding with evaluation_finding_create",
    )


def save_finding(run_dir: pathlib.Path, finding: Finding) -> pathlib.Path:
    """Atomically write ``finding`` to ``<run>/analysis/findings/<id>.yaml``."""
    path = _finding_file(run_dir, finding.finding_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        finding.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    atomic_write_text(path, payload)
    return path


def _new_finding_id(existing: list[Finding]) -> str:
    taken = {item.finding_id for item in existing}
    for _ in range(100):
        candidate = f"fnd-{uuid.uuid4().hex[:8]}"
        if candidate not in taken:
            return candidate
    raise FindingError(
        "could not allocate a finding id",
        hint="the run holds an implausible number of findings",
    )


#: Reference kinds a finding may carry: (missing message, remediation hint).
_REFERENCE_ERRORS: Final[dict[str, tuple[str, str]]] = {
    "evidence": (
        "no pinned evidence {value!r} in this run",
        "evaluation_query pin_evidence=true pins one, evaluation_evidence lists them",
    ),
    "stored_query": (
        "no stored query {value!r} in this run",
        "evaluation_queries lists the run's query declarations",
    ),
    "figure": (
        "no figure at {value!r}",
        "evaluation_render_plot writes plot artifacts into analysis/plots",
    ),
}


def _checked_reference(run_dir: pathlib.Path, kind: str, value: str) -> str:
    """Validate one evidence id, stored query name or figure path a finding references."""
    match kind:
        case "evidence":
            index = EvidenceStore(run_dir).load_index()
            exists = not index.filter(pl.col("evidence_id") == value).is_empty()
        case "stored_query":
            exists = QueryCatalog(run_dir=run_dir).has_query(value)
        case "figure":
            candidate = pathlib.Path(value)
            resolved = candidate if candidate.is_absolute() else run_dir / candidate
            exists = resolved.is_file()
            if exists:
                value = (
                    str(resolved.relative_to(run_dir))
                    if resolved.is_relative_to(run_dir)
                    else str(resolved)
                )
        case _:
            raise AssertionError(f"unknown finding reference kind {kind!r}")
    if not exists:
        message, hint = _REFERENCE_ERRORS[kind]
        raise FindingError(message.format(value=value), hint=hint)
    return value


def _number_match(
    series: pl.Series, value: float, tolerance: float
) -> tuple[bool, float | None]:
    """Return whether ``series`` holds ``value`` within ``tolerance`` and its closest entry."""
    series = series.drop_nulls().cast(pl.Float64)
    if series.is_empty():
        return False, None
    diffs = (series - value).abs()
    closest_idx = diffs.arg_min()
    closest = None if closest_idx is None else float(series[closest_idx])
    return bool((diffs <= tolerance).any()), closest


def _validate_structured_numbers(
    run_dir: pathlib.Path, entries: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if not entries:
        return []
    store = EvidenceStore(run_dir)
    verified: list[dict[str, Any]] = []
    for entry in entries:
        ev = str(entry.get("evidence_id") or "")
        col = str(entry.get("column") or "")
        val = float(entry["value"])
        tol = float(entry.get("tolerance", 1e-6))
        try:
            frame = store.read_frame(ev)
        except KeyError as exc:
            raise FindingError(
                f"evidence {ev!r} not found in EvidenceStore",
                hint="pin evidence first with evaluation_query(pin_evidence=True) or evaluation_analysis",
            ) from exc
        if col not in frame.columns:
            raise FindingError(
                f"column {col!r} not found in evidence {ev!r}",
                hint=f"columns: {', '.join(frame.columns)}",
            )
        if not _number_match(frame[col], val, tol)[0]:
            raise FindingError(
                f"number {val} for {col!r} not found in evidence {ev!r} within tolerance {tol}",
                hint="quote a value actually present in the pinned evidence frame",
            )
        verified.append(
            {"evidence_id": ev, "column": col, "value": val, "tolerance": tol}
        )
    return verified


def create_finding(
    run_dir: pathlib.Path,
    claim: str,
    *,
    episode_id: int | None = None,
    encounter_id: str | None = None,
    contact_id: str | None = None,
    pedestrian_id: str | None = None,
    query: str | None = None,
    stored_query: str | None = None,
    evidence_id: str | None = None,
    evidence_ids: list[str] | None = None,
    numbers: dict[str, float] | None = None,
    structured_numbers: list[dict[str, Any]] | None = None,
    author: str,
    figure_path: str | None = None,
) -> Finding:
    """Validate and persist a new finding in ``proposed`` status."""
    text = claim.strip()
    if not text:
        raise FindingError(
            "a finding needs a claim", hint="pass the empirical claim as one sentence"
        )
    author_clean = author.strip()
    if not author_clean:
        raise FindingError(
            "a finding needs an author", hint="pass the authoring role or identifier"
        )
    if query is not None and stored_query is not None:
        raise FindingError(
            "pass the SQL or a stored query name, not both",
            hint="evaluation_queries lists stored queries",
        )
    if numbers:
        raise FindingError(
            "free-form numbers are not checked against evidence",
            hint="quote each number as a structured_numbers entry {evidence_id, column, value, tolerance} that the pinned evidence contains",
        )
    checked = {
        kind: None if raw is None else _checked_reference(run_dir, kind, raw)
        for kind, raw in (
            ("stored_query", stored_query),
            ("evidence", evidence_id),
            ("figure", figure_path),
        )
    }
    verified_structured = _validate_structured_numbers(run_dir, structured_numbers)
    all_ev = list(
        dict.fromkeys(
            [
                *(evidence_ids or []),
                *([checked["evidence"]] if checked["evidence"] else []),
            ]
        )
    )

    fdir = findings_dir(run_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    with file_lock(findings_lock_path(run_dir)):
        existing = load_findings(run_dir)
        finding = Finding(
            finding_id=_new_finding_id(existing),
            claim=text,
            status=FindingStatus.PROPOSED,
            created_at=datetime.datetime.now(tz=datetime.UTC).isoformat(
                timespec="seconds"
            ),
            episode_id=episode_id,
            encounter_id=encounter_id,
            contact_id=contact_id,
            pedestrian_id=pedestrian_id,
            query=query,
            stored_query=checked["stored_query"],
            evidence_id=checked["evidence"] or (all_ev[0] if all_ev else None),
            evidence_ids=all_ev,
            structured_numbers=verified_structured,
            author=author_clean,
            figure_path=checked["figure"],
        )
        save_finding(run_dir, finding)
    return finding


def _rerun_evidence(
    run_dir: pathlib.Path,
    catalog: BenchmarkCatalog,
    record: Any,
    stored_frame: pl.DataFrame,
    *,
    computation_runners: _ComputationRunners | None = None,
) -> tuple[pl.DataFrame, set[object]]:
    """Rerun one evidence record and return its frame with the seeds it covers."""
    if record.kind == "sql":
        sql = str(record.sql_text or "").strip()
        if not sql:
            raise ValueError(
                f"SQL evidence {record.evidence_id!r} has empty sql_text and cannot be rerun; re-pin via evaluation_query"
            )
        frame = PolarsSQLEngine(catalog).query(sql)
    elif record.kind == "computation":
        if computation_runners and record.operation in computation_runners:
            frame = computation_runners[record.operation](run_dir, record.params)
        else:
            from arena_evaluation.analytics.analyses.procedures import (
                list_procedures,
                run_procedure,
            )

            if record.operation not in {p["name"] for p in list_procedures()}:
                raise ValueError(
                    f"Unsupported computation evidence operation {record.operation!r}; cannot rerun"
                )
            frame = run_procedure(record.operation, catalog, record.params).frame
    else:
        raise ValueError(f"Unsupported evidence kind {record.kind!r}; cannot rerun")

    raw = record.params.get("seeds") if isinstance(record.params, dict) else None
    if isinstance(raw, (list, tuple, set)):
        seeds = set(raw)
    elif stored_frame.schema.get("seed") is not None:
        seeds = set(stored_frame.get_column("seed").drop_nulls())
    else:
        seeds = set()
    return frame, seeds


def _check(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    """One verification check record: its name, outcome and details."""
    return {"check": name, "passed": passed, **details}


def verify_finding(
    run_dir: pathlib.Path,
    finding_id: str,
    *,
    computation_runners: _ComputationRunners | None = None,
) -> dict[str, Any]:
    """Mechanically verify a finding's evidence integrity, rerun consistency, and structured numbers."""
    store = EvidenceStore(run_dir)
    catalog = BenchmarkCatalog.open(run_dir)

    with file_lock(findings_lock_path(run_dir)):
        finding = load_finding(run_dir, finding_id)
        checks: list[dict[str, Any]] = []
        passed = True
        rerun_frames: dict[str, pl.DataFrame] = {}
        seed_sets: list[tuple[str, set[object]]] = []
        evidence_list = finding.evidence_ids or (
            [finding.evidence_id] if finding.evidence_id else []
        )
        for ev in evidence_list:
            try:
                record = store.get(ev)
                stored_frame = store.read_frame(ev)
            except KeyError:
                passed = False
                checks.append(_check("evidence_exists", False, evidence_id=ev))
                continue
            checks.append(
                _check(
                    "evidence_exists", True, evidence_id=ev, rows=stored_frame.height
                )
            )
            with tempfile.TemporaryDirectory() as tmp_dir:
                recomputed = EvidenceStore(pathlib.Path(tmp_dir)).record(
                    stored_frame,
                    kind=record.kind,
                    operation=record.operation,
                    snapshot_id=record.snapshot_id,
                    purpose=record.purpose,
                    params=record.params,
                    sql_text=record.sql_text,
                )
            integrity_ok = recomputed.evidence_id == ev
            checks.append(
                _check(
                    "evidence_integrity",
                    integrity_ok,
                    evidence_id=ev,
                    stored_id=ev,
                    recomputed_id=recomputed.evidence_id,
                )
            )
            if not integrity_ok:
                passed = False
                continue
            try:
                rerun_frame, seeds = _rerun_evidence(
                    run_dir,
                    catalog,
                    record,
                    stored_frame,
                    computation_runners=computation_runners,
                )
            except (KeyError, ValueError, OSError, pl.exceptions.PolarsError) as exc:
                passed = False
                checks.append(
                    _check("evidence_rerun", False, evidence_id=ev, hint=str(exc))
                )
                continue
            rerun_frames[ev] = rerun_frame
            seed_sets.append((ev, seeds))
            if record.snapshot_id != catalog.snapshot_id:
                passed = False
                checks.append(
                    _check(
                        "evidence_rerun",
                        False,
                        evidence_id=ev,
                        mode="snapshot_changed",
                        stored_snapshot=record.snapshot_id,
                        current_snapshot=catalog.snapshot_id,
                        rerun_rows=rerun_frame.height,
                    )
                )
                continue
            stored_hash = compute_frame_hash(stored_frame)
            rerun_hash = compute_frame_hash(rerun_frame)
            hash_ok = stored_hash == rerun_hash
            if not hash_ok:
                passed = False
            checks.append(
                _check(
                    "evidence_rerun",
                    hash_ok,
                    evidence_id=ev,
                    mode="hash",
                    stored_hash=stored_hash,
                    rerun_hash=rerun_hash,
                )
            )
        if len(seed_sets) >= 2:
            overlaps = [
                {"evidence_ids": [ev_a, ev_b], "seeds": common}
                for idx, (ev_a, seeds_a) in enumerate(seed_sets)
                if seeds_a
                for ev_b, seeds_b in seed_sets[idx + 1 :]
                if (common := sorted(seeds_a & seeds_b, key=str))
            ]
            checks.append(
                _check(
                    "same_seed_overlap",
                    True,
                    flagged=bool(overlaps),
                    **({"overlaps": overlaps} if overlaps else {}),
                )
            )
        for entry in finding.structured_numbers:
            ev = str(entry.get("evidence_id") or "")
            col = str(entry.get("column") or "")
            val = float(entry["value"])
            tol = float(entry.get("tolerance", 1e-6))
            target_frame = rerun_frames.get(ev)
            ok, rerun_val = (
                (False, None)
                if target_frame is None or col not in target_frame.columns
                else _number_match(target_frame[col], val, tol)
            )
            if not ok:
                passed = False
            checks.append(
                _check(
                    "structured_number",
                    ok,
                    evidence_id=ev,
                    column=col,
                    value=val,
                    rerun_value=rerun_val,
                )
            )
        status = (
            finding.status
            if passed
            else _FAILED_VERIFICATION_STATUS.get(finding.status, finding.status)
        )
        updated = finding.model_copy(
            update={
                "status": status,
                "mechanical_passed": passed,
                "verification_checks": checks,
                "snapshot_id": catalog.snapshot_id if passed else finding.snapshot_id,
            }
        )
        save_finding(run_dir, updated)

    return {
        "run_id": run_dir.name,
        "finding_id": finding_id,
        "snapshot_id": catalog.snapshot_id,
        "mechanical_passed": passed,
        "status": updated.status.value,
        "previous_status": finding.status.value,
        "checks": checks,
    }


def transition_finding_status(
    run_dir: pathlib.Path,
    finding_id: str,
    status: str,
    *,
    by: str = "verifier",
    note: str | None = None,
) -> Finding:
    """Move one finding between review states with mechanical and author-separation gates."""
    try:
        target = FindingStatus(status)
    except ValueError as exc:
        raise FindingError(
            f"unknown finding status {status!r}",
            hint="proposed, verified, refuted or needs_work",
        ) from exc

    with file_lock(findings_lock_path(run_dir)):
        finding = load_finding(run_dir, finding_id)
        if (
            finding.status is not target
            and target not in ALLOWED_TRANSITIONS[finding.status]
        ):
            raise FindingError(
                f"finding {finding_id} is {finding.status.value}, it cannot become {target.value}",
                hint="proposed -> verified|refuted|needs_work; needs_work -> proposed|verified|refuted; verified -> refuted|needs_work",
            )
        for violated, message, hint in _ENTRY_GATES.get(target, ()):
            if violated(finding, by):
                raise FindingError(
                    message.format(
                        finding_id=finding.finding_id, author=finding.author
                    ),
                    hint=hint,
                )
        if finding.status is target and (note is None or note == finding.note):
            return finding
        updates: dict[str, Any] = {"status": target}
        if note is not None:
            updates["note"] = note
        reviewed = finding.model_copy(update=updates)
        save_finding(run_dir, reviewed)
    return reviewed
