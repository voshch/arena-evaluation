"""Content-Addressed Evidence Store and AnalysisResult (Layer 3).

Records analytical SQL and computation results under ``analysis/evidence/`` with
deterministic ``ev-<sha256[:12]>`` identifiers, snapshot identity, and a single flat
Parquet index at ``analysis/evidence_index.parquet`` (no JSON sidecars).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Final

import polars as pl

from arena_evaluation.storage.atomic import file_lock, staged_temp_path
from arena_evaluation.storage.registry import (
    FACT_TABLES,
    dimensions,
    registered_tables,
)
from arena_evaluation.storage.config_layout import run_evidence_dir
from arena_evaluation.storage.registry.analysis import EvidenceIndex

__all__ = [
    "EVIDENCE_INDEX_SCHEMA",
    "VALID_EVIDENCE_KINDS",
    "AnalysisResult",
    "EvidenceRecord",
    "EvidenceStore",
    "compute_frame_hash",
    "compute_schema_signature",
    "compute_snapshot_id",
]

VALID_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset({"sql", "computation"})

EVIDENCE_INDEX_SCHEMA: Final[pl.Schema] = EvidenceIndex.polars_schema()


def compute_schema_signature(frame: pl.DataFrame | pl.LazyFrame) -> str:
    """Compute a canonical string signature ``col1:Type1,col2:Type2,...`` for a frame schema."""
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return ",".join(f"{name}:{dtype}" for name, dtype in schema.items())


def compute_frame_hash(frame: pl.DataFrame) -> str:
    """Compute a row-order-insensitive, Polars-version-independent SHA-256 hex digest of a DataFrame."""
    hasher = hashlib.sha256()
    hasher.update(compute_schema_signature(frame).encode())
    hasher.update(f"|rows:{frame.height}|".encode())
    if frame.height > 0 and frame.width > 0:
        sorted_frame = frame.sort(frame.columns, nulls_last=True)
        hasher.update(sorted_frame.write_csv().encode("utf-8"))
    return hasher.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _registry_signature() -> str:
    parts = [
        f"{tbl.table_name}:{','.join(f'{c}:{d}' for c, d in tbl.polars_schema().items())}"
        for tbl in sorted(registered_tables(), key=lambda t: t.table_name)
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _file_fingerprint(path: pathlib.Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def compute_snapshot_id(benchmark_dir: pathlib.Path) -> str:
    """Compute a deterministic ``snap-<sha256[:12]>`` dataset and registry snapshot identifier."""
    file_fps: dict[str, str] = {}
    if benchmark_dir.is_dir():
        for tbl in dimensions.DIMENSION_TABLES.values():
            dim_path = benchmark_dir / tbl.relative_path()
            if dim_path.is_file():
                file_fps[tbl.relative_path().as_posix()] = _file_fingerprint(dim_path)
        episodes_root = benchmark_dir / "episodes"
        if episodes_root.is_dir():
            for ep_dir in sorted(p for p in episodes_root.iterdir() if p.is_dir()):
                for fact_tbl in FACT_TABLES.values():
                    fpath = ep_dir / fact_tbl.relative_path()
                    if fpath.is_file():
                        rel = fpath.relative_to(benchmark_dir).as_posix()
                        file_fps[rel] = _file_fingerprint(fpath)

    payload = {
        "registry_signature": _registry_signature(),
        "files": dict(sorted(file_fps.items())),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return f"snap-{digest[:12]}"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult:
    """Immutable result of an analytical computation or procedure (D-18)."""

    frame: pl.DataFrame
    summary: pl.DataFrame
    excluded_rows: Mapping[str, int] = dataclasses.field(default_factory=dict)
    seeds_used: tuple[int, ...] = ()
    snapshot_id: str = ""
    evidence_id: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceRecord:
    """Immutable metadata record for a content-addressed analytical evidence artifact."""

    evidence_id: str
    kind: str
    operation: str
    params: Mapping[str, object]
    row_count: int
    schema_signature: str
    created_at_ns: int
    snapshot_id: str
    artifact_path: pathlib.Path
    sql_text: str = ""
    purpose: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize record metadata to a JSON-compatible dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "operation": self.operation,
            "purpose": self.purpose,
            "sql_text": self.sql_text,
            "snapshot_id": self.snapshot_id,
            "params": dict(self.params),
            "params_json": _canonical_json(self.params),
            "row_count": self.row_count,
            "schema_signature": self.schema_signature,
            "created_at_ns": self.created_at_ns,
            "artifact_path": self.artifact_path.as_posix(),
        }

    @classmethod
    def from_index_row(cls, row: Mapping[str, object]) -> EvidenceRecord:
        """Construct an ``EvidenceRecord`` from a row of ``evidence_index.parquet``."""
        raw_params = row.get("params_json") or "{}"
        parsed_params = (
            json.loads(str(raw_params)) if isinstance(raw_params, str) else {}
        )
        raw_artifact = row.get("artifact_path")
        if not raw_artifact:
            raise ValueError("Evidence index row is missing artifact_path.")
        return cls(
            evidence_id=str(row["evidence_id"]),
            kind=str(row["kind"]),
            operation=str(row["operation"]),
            purpose=str(row.get("purpose") or ""),
            sql_text=str(row.get("sql_text") or ""),
            snapshot_id=str(row["snapshot_id"]),
            params=parsed_params if isinstance(parsed_params, dict) else {},
            row_count=int(str(row["row_count"])),
            schema_signature=str(row["schema_signature"]),
            created_at_ns=int(str(row["created_at_ns"])),
            artifact_path=pathlib.Path(str(raw_artifact)),
        )


class EvidenceStore:
    """Content-addressed store for analytical results indexed at ``analysis/evidence_index.parquet``."""

    def __init__(self, benchmark_dir: pathlib.Path) -> None:
        self.benchmark_dir = pathlib.Path(benchmark_dir)
        self.evidence_dir = run_evidence_dir(self.benchmark_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    @property
    def index_path(self) -> pathlib.Path:
        """Path to the authoritative ``analysis/evidence_index.parquet`` file."""
        return self.benchmark_dir / EvidenceIndex.relative_path()

    def record(
        self,
        frame: pl.DataFrame | pl.LazyFrame,
        *,
        kind: str,
        operation: str,
        snapshot_id: str,
        purpose: str = "",
        params: Mapping[str, object] | None = None,
        sql_text: str = "",
        created_at_ns: int | None = None,
    ) -> EvidenceRecord:
        """Record a DataFrame result and return its content-addressed ``EvidenceRecord``."""
        if kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(
                f"Unsupported evidence kind {kind!r}. Valid kinds: {sorted(VALID_EVIDENCE_KINDS)}"
            )
        if not operation or not operation.strip():
            raise ValueError("operation must be a non-empty string.")
        if not snapshot_id or not snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string.")

        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        nested = [
            name
            for name, dtype in df.schema.items()
            if isinstance(dtype, (pl.List, pl.Struct, pl.Array))
        ]
        if nested:
            raise ValueError(
                f"EvidenceStore only accepts flat frames; nested column(s) {nested} are not supported."
            )
        normalized_params = dict(params or {})
        schema_sig = compute_schema_signature(df)
        data_hash = compute_frame_hash(df)

        fingerprint_payload: dict[str, object] = {
            "kind": kind,
            "operation": operation.strip(),
            "params": normalized_params,
            "snapshot_id": snapshot_id,
            "sql_text": sql_text.strip(),
            "schema_signature": schema_sig,
            "data_hash": data_hash,
        }
        digest = hashlib.sha256(
            _canonical_json(fingerprint_payload).encode()
        ).hexdigest()
        evidence_id = f"ev-{digest[:12]}"

        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        abs_artifact = self.evidence_dir / f"{evidence_id}.parquet"
        with staged_temp_path(abs_artifact) as temporary:
            df.write_parquet(temporary)
        rel_artifact = abs_artifact.relative_to(self.benchmark_dir)

        stamp_ns = int(created_at_ns) if created_at_ns is not None else time.time_ns()
        rec = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,
            operation=operation.strip(),
            purpose=purpose.strip(),
            sql_text=sql_text.strip(),
            snapshot_id=snapshot_id,
            params=normalized_params,
            row_count=df.height,
            schema_signature=schema_sig,
            created_at_ns=stamp_ns,
            artifact_path=rel_artifact,
        )

        self._upsert_index(rec)
        return rec

    def _row_dict(self, record: EvidenceRecord) -> dict[str, object]:
        return {
            "evidence_id": record.evidence_id,
            "kind": record.kind,
            "operation": record.operation,
            "purpose": record.purpose,
            "sql_text": record.sql_text,
            "snapshot_id": record.snapshot_id,
            "params_json": _canonical_json(record.params),
            "row_count": record.row_count,
            "schema_signature": record.schema_signature,
            "created_at_ns": record.created_at_ns,
            "artifact_path": record.artifact_path.as_posix(),
        }

    def _upsert_index(self, record: EvidenceRecord) -> None:
        """Upsert a single record into the authoritative ``analysis/evidence_index.parquet``."""
        index_path = self.index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)
        row = pl.DataFrame([self._row_dict(record)], schema=EVIDENCE_INDEX_SCHEMA)
        with file_lock(index_path):
            if index_path.is_file():
                existing = pl.read_parquet(index_path)
                filtered = existing.filter(pl.col("evidence_id") != record.evidence_id)
                updated = pl.concat([filtered, row], how="vertical_relaxed")
            else:
                updated = row

            with staged_temp_path(index_path) as temporary:
                updated.write_parquet(temporary)

    def load_index(self) -> pl.DataFrame:
        """Load the evidence index as a DataFrame, or an empty index frame if none exists."""
        if self.index_path.is_file():
            return pl.read_parquet(self.index_path)
        return pl.DataFrame(schema=EVIDENCE_INDEX_SCHEMA)

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Retrieve an ``EvidenceRecord`` by ID from ``evidence_index.parquet``."""
        matched = self.load_index().filter(pl.col("evidence_id") == evidence_id)
        if matched.is_empty():
            raise KeyError(f"Evidence record '{evidence_id}' not found.")
        return EvidenceRecord.from_index_row(matched.row(0, named=True))

    def read_frame(self, evidence_id: str) -> pl.DataFrame:
        """Read the persisted Parquet frame for ``evidence_id`` directly from disk."""
        rec = self.get(evidence_id)
        abs_path = self.benchmark_dir / rec.artifact_path
        if abs_path.is_file():
            return pl.read_parquet(abs_path)
        raise KeyError(f"Evidence frame '{evidence_id}' not found at {abs_path}.")

    def list_records(self) -> tuple[EvidenceRecord, ...]:
        """Return all registered ``EvidenceRecord`` entries from the index sorted by ID."""
        df = self.load_index()
        if df.is_empty():
            return ()
        rows = df.sort("evidence_id").to_dicts()
        return tuple(EvidenceRecord.from_index_row(row) for row in rows)
