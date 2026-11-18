"""Layer 3 finding lifecycle: single store under ``analysis/findings/<id>.yaml``."""

from __future__ import annotations

import datetime
import math
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
    numbers: dict[str, float] = pydantic.Field(default_factory=dict)
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


def findings_dir(run_dir: pathlib.Path) -> pathlib.Path:
    """Return ``<run>/analysis/findings`` directory."""
    return run_analysis_root(run_dir) / FINDINGS_SUBDIR


def findings_lock_path(run_dir: pathlib.Path) -> pathlib.Path:
    """Return lock file path inside ``<run>/analysis/findings``."""
    return findings_dir(run_dir) / ".findings.lock"


def _finding_file(run_dir: pathlib.Path, finding_id: str) -> pathlib.Path:
    return findings_dir(run_dir) / f"{finding_id}.yaml"


def load_findings(run_dir: pathlib.Path) -> list[Finding]:
    """Load all findings from ``<run>/analysis/findings/*.yaml`` sorted by filename."""
    fdir = findings_dir(run_dir)
    if not fdir.is_dir():
        return []
    out: list[Finding] = []
    for path in sorted(fdir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as exc:
            raise FindingError(
                f"corrupt finding file {path.name}: {exc}",
                hint=f"fix or remove {path}",
            ) from exc
        if not isinstance(raw, dict):
            raise FindingError(
                f"finding file {path.name} is not a YAML mapping",
                hint=f"fix or remove {path}",
            )
        try:
            out.append(Finding.model_validate(raw))
        except pydantic.ValidationError as exc:
            raise FindingError(
                f"invalid finding schema in {path.name}: {exc}",
                hint=f"fix or remove {path}",
            ) from exc
    return out


def load_finding(run_dir: pathlib.Path, finding_id: str) -> Finding:
    """Load one finding by id, raising FindingError when absent."""
    path = _finding_file(run_dir, finding_id)
    if not path.is_file():
        raise FindingError(
            f"no finding {finding_id!r} in this run",
            hint="evaluation_findings lists them",
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FindingError(
            f"finding {finding_id!r} is not a YAML mapping",
            hint="re-create the finding with evaluation_finding_create",
        )
    return Finding.model_validate(raw)


def save_finding(run_dir: pathlib.Path, finding: Finding) -> pathlib.Path:
    """Atomically write ``finding`` to ``<run>/analysis/findings/<id>.yaml``."""
    fdir = findings_dir(run_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    path = _finding_file(run_dir, finding.finding_id)
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


def _checked_numbers(numbers: dict[str, float] | None) -> dict[str, float]:
    if not numbers:
        return {}
    non_finite = sorted(k for k, v in numbers.items() if not math.isfinite(v))
    if non_finite:
        raise FindingError(
            f"finding numbers must be finite, got {', '.join(non_finite)}",
            hint="pass finite numbers, e.g. {'min_clearance_m': 0.42}",
        )
    return dict(numbers)


def _checked_evidence(run_dir: pathlib.Path, evidence_id: str) -> str:
    store = EvidenceStore(run_dir)
    index = store.load_index().filter(pl.col("evidence_id") == evidence_id)
    if len(index) == 0:
        raise FindingError(
            f"no pinned evidence {evidence_id!r} in this run",
            hint="evaluation_query pin_evidence=true pins one, evaluation_evidence lists them",
        )
    return evidence_id


def _checked_stored_query(run_dir: pathlib.Path, name: str) -> str:
    if not QueryCatalog(run_dir=run_dir).has_query(name):
        raise FindingError(
            f"no stored query {name!r} in this run",
            hint="evaluation_queries lists the run's query declarations",
        )
    return name


def _checked_figure(run_dir: pathlib.Path, figure_path: str) -> str:
    candidate = pathlib.Path(figure_path)
    resolved = candidate if candidate.is_absolute() else run_dir / candidate
    if not resolved.is_file():
        raise FindingError(
            f"no figure at {figure_path!r}",
            hint="evaluation_render_plot writes plot artifacts into analysis/plots",
        )
    try:
        return str(resolved.relative_to(run_dir))
    except ValueError:
        return str(resolved)


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
        series = frame[col].drop_nulls().cast(float)
        if series.is_empty() or not bool(((series - val).abs() <= tol).any()):
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
    checked_stored_query = (
        None if stored_query is None else _checked_stored_query(run_dir, stored_query)
    )
    checked_evidence = (
        None if evidence_id is None else _checked_evidence(run_dir, evidence_id)
    )
    checked_figure = (
        None if figure_path is None else _checked_figure(run_dir, figure_path)
    )
    checked_numbers = _checked_numbers(numbers)
    verified_structured = _validate_structured_numbers(run_dir, structured_numbers)
    all_ev = list(
        dict.fromkeys(
            [*(evidence_ids or []), *([checked_evidence] if checked_evidence else [])]
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
            stored_query=checked_stored_query,
            evidence_id=checked_evidence or (all_ev[0] if all_ev else None),
            evidence_ids=all_ev,
            numbers=checked_numbers,
            structured_numbers=verified_structured,
            author=author_clean,
            figure_path=checked_figure,
        )
        save_finding(run_dir, finding)
    return finding


def _rerun_evidence(
    run_dir: pathlib.Path,
    catalog: BenchmarkCatalog,
    record: Any,
    *,
    computation_runners: Mapping[
        str, Callable[[pathlib.Path, Mapping[str, Any]], pl.DataFrame]
    ]
    | None = None,
) -> pl.DataFrame:
    """Rerun one EvidenceRecord's SQL query or computation against ``catalog``."""
    if record.kind == "sql":
        sql = str(record.sql_text or "").strip()
        if not sql:
            raise ValueError(
                f"SQL evidence {record.evidence_id!r} has empty sql_text and cannot be rerun; re-pin via evaluation_query"
            )
        return PolarsSQLEngine(catalog).query(sql)
    if record.kind == "computation":
        if computation_runners and record.operation in computation_runners:
            return computation_runners[record.operation](run_dir, record.params)
        from arena_evaluation.analytics.analyses.procedures import (
            list_procedures,
            run_procedure,
        )

        if record.operation in {p["name"] for p in list_procedures()}:
            return run_procedure(record.operation, catalog, record.params).frame
        raise ValueError(
            f"Unsupported computation evidence operation {record.operation!r}; cannot rerun"
        )
    raise ValueError(f"Unsupported evidence kind {record.kind!r}; cannot rerun")


def _extract_seeds(record: Any, frame: pl.DataFrame) -> set[object]:
    """Extract seed identifiers from an evidence record's params or frame."""
    raw = record.params.get("seeds") if isinstance(record.params, dict) else None
    if isinstance(raw, (list, tuple, set)):
        return set(raw)
    if frame.schema.get("seed") is not None:
        return set(frame.get_column("seed").drop_nulls())
    return set()


def verify_finding(
    run_dir: pathlib.Path,
    finding_id: str,
    *,
    computation_runners: Mapping[
        str, Callable[[pathlib.Path, Mapping[str, Any]], pl.DataFrame]
    ]
    | None = None,
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
                checks.append(
                    {"check": "evidence_exists", "evidence_id": ev, "passed": False}
                )
                continue

            checks.append(
                {
                    "check": "evidence_exists",
                    "evidence_id": ev,
                    "passed": True,
                    "rows": stored_frame.height,
                }
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
            if not integrity_ok:
                passed = False
            checks.append(
                {
                    "check": "evidence_integrity",
                    "evidence_id": ev,
                    "passed": integrity_ok,
                    "stored_id": ev,
                    "recomputed_id": recomputed.evidence_id,
                }
            )
            if not integrity_ok:
                continue

            try:
                rerun_frame = _rerun_evidence(
                    run_dir,
                    catalog,
                    record,
                    computation_runners=computation_runners,
                )
            except (KeyError, ValueError, OSError, pl.exceptions.PolarsError) as exc:
                passed = False
                checks.append(
                    {
                        "check": "evidence_rerun",
                        "evidence_id": ev,
                        "passed": False,
                        "hint": str(exc),
                    }
                )
                continue

            rerun_frames[ev] = rerun_frame
            seed_sets.append((ev, _extract_seeds(record, stored_frame)))

            if record.snapshot_id == catalog.snapshot_id:
                stored_hash = compute_frame_hash(stored_frame)
                rerun_hash = compute_frame_hash(rerun_frame)
                hash_ok = stored_hash == rerun_hash
                if not hash_ok:
                    passed = False
                checks.append(
                    {
                        "check": "evidence_rerun",
                        "evidence_id": ev,
                        "mode": "hash",
                        "stored_hash": stored_hash,
                        "rerun_hash": rerun_hash,
                        "passed": hash_ok,
                    }
                )
            else:
                checks.append(
                    {
                        "check": "evidence_rerun",
                        "evidence_id": ev,
                        "mode": "snapshot_changed",
                        "stored_snapshot": record.snapshot_id,
                        "current_snapshot": catalog.snapshot_id,
                        "rerun_rows": rerun_frame.height,
                        "passed": True,
                    }
                )

        if len(seed_sets) >= 2:
            overlaps: list[dict[str, Any]] = []
            for idx, (ev_a, seeds_a) in enumerate(seed_sets):
                if not seeds_a:
                    continue
                for ev_b, seeds_b in seed_sets[idx + 1 :]:
                    common = sorted(seeds_a & seeds_b, key=str)
                    if common:
                        overlaps.append({"evidence_ids": [ev_a, ev_b], "seeds": common})
            checks.append(
                {
                    "check": "same_seed_overlap",
                    "flagged": bool(overlaps),
                    "passed": True,
                    **({"overlaps": overlaps} if overlaps else {}),
                }
            )

        for entry in finding.structured_numbers:
            ev = str(entry.get("evidence_id") or "")
            col = str(entry.get("column") or "")
            val = float(entry["value"])
            tol = float(entry.get("tolerance", 1e-6))
            rerun_val: float | None = None
            target_frame = rerun_frames.get(ev)
            if target_frame is None or col not in target_frame.columns:
                ok = False
            else:
                series = target_frame[col].drop_nulls().cast(pl.Float64)
                if series.is_empty():
                    ok = False
                else:
                    diffs = (series - val).abs()
                    ok = bool((diffs <= tol).any())
                    min_idx = diffs.arg_min()
                    if min_idx is not None:
                        rerun_val = float(series[min_idx])
            if not ok:
                passed = False
            checks.append(
                {
                    "check": "structured_number",
                    "evidence_id": ev,
                    "column": col,
                    "value": val,
                    "rerun_value": rerun_val,
                    "passed": ok,
                }
            )

        status = finding.status
        if not passed and finding.status is FindingStatus.VERIFIED:
            status = FindingStatus.NEEDS_WORK
        updated = finding.model_copy(
            update={
                "status": status,
                "mechanical_passed": passed,
                "verification_checks": checks,
                "snapshot_id": catalog.snapshot_id,
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
        if target is FindingStatus.VERIFIED:
            if finding.mechanical_passed is not True:
                raise FindingError(
                    f"finding {finding_id} has not passed mechanical verification",
                    hint="run evaluation_finding_verify first",
                )
            if by == finding.author:
                raise FindingError(
                    f"author role {finding.author!r} cannot verify its own finding {finding_id}",
                    hint="use a different verifier role",
                )
        if finding.status is target and (note is None or note == finding.note):
            return finding
        updates: dict[str, Any] = {"status": target}
        if note is not None:
            updates["note"] = note
        reviewed = finding.model_copy(update=updates)
        save_finding(run_dir, reviewed)
    return reviewed
