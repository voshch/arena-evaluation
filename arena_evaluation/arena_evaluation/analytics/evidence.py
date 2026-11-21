"""Content-addressed evidence store with ``ev-<sha256[:12]>`` ids and a flat Parquet index."""

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
from arena_evaluation.storage.config_layout import run_evidence_dir
from arena_evaluation.storage.registry import registered_tables
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
_SNAPSHOT_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset({"analysis", "reports", "plots", "hhi"})


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(*chunks: str) -> str:
    return hashlib.sha256("".join(chunks).encode()).hexdigest()


def compute_schema_signature(frame: pl.DataFrame | pl.LazyFrame) -> str:
    """Compute a canonical ``col1:Type1,col2:Type2,...`` frame-schema signature."""
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    return ",".join(f"{name}:{dtype}" for name, dtype in schema.items())


def compute_frame_hash(frame: pl.DataFrame) -> str:
    """Compute a row-order-insensitive, Polars-version-independent SHA-256 digest of a DataFrame."""
    csv = frame.sort(frame.columns, nulls_last=True).write_csv() if frame.height and frame.width else ""
    return _digest(compute_schema_signature(frame), f"|rows:{frame.height}|", csv)


def compute_snapshot_id(benchmark_dir: pathlib.Path) -> str:
    """Compute a deterministic ``snap-<sha256[:12]>`` dataset and registry snapshot identifier."""
    file_fps: dict[str, str] = {}
    if benchmark_dir.is_dir():
        for path in sorted(benchmark_dir.rglob("*.parquet")):
            relative = path.relative_to(benchmark_dir)
            if relative.parts and relative.parts[0] in _SNAPSHOT_EXCLUDED_DIRS:
                continue
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(65536):
                    hasher.update(chunk)
            file_fps[relative.as_posix()] = hasher.hexdigest()[:16]
    registry = _digest("|".join(f"{table.table_name}:{','.join(f'{column}:{dtype}' for column, dtype in table.polars_schema().items())}" for table in sorted(registered_tables(), key=lambda table: table.table_name)))[:16]
    payload = {"registry_signature": registry, "files": dict(sorted(file_fps.items()))}
    return f"snap-{_digest(_canonical_json(payload))[:12]}"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult:
    """Immutable result of an analytical computation or procedure (D-18)."""

    frame: pl.DataFrame
    summary: pl.DataFrame
    excluded_rows: Mapping[str, int] = dataclasses.field(default_factory=dict)
    seeds_used: tuple[int, ...] = ()
    snapshot_id: str = ""
    evidence_id: str = ""
    diagnostics: tuple[str, ...] = ()


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
        return dataclasses.asdict(self) | {
            "params_json": _canonical_json(self.params),
            "artifact_path": self.artifact_path.as_posix(),
        }

    @classmethod
    def from_index_row(cls, row: Mapping[str, object]) -> EvidenceRecord:
        """Construct an ``EvidenceRecord`` from a row of ``evidence_index.parquet``."""
        artifact = row.get("artifact_path")
        if not artifact:
            raise ValueError("Evidence index row is missing artifact_path.")
        raw = row.get("params_json") or "{}"
        parsed = json.loads(raw) if isinstance(raw, str) else {}
        return cls(
            evidence_id=str(row["evidence_id"]),
            kind=str(row["kind"]),
            operation=str(row["operation"]),
            purpose=str(row.get("purpose") or ""),
            sql_text=str(row.get("sql_text") or ""),
            snapshot_id=str(row["snapshot_id"]),
            params=parsed if isinstance(parsed, dict) else {},
            row_count=int(str(row["row_count"])),
            schema_signature=str(row["schema_signature"]),
            created_at_ns=int(str(row["created_at_ns"])),
            artifact_path=pathlib.Path(str(artifact)),
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
            raise ValueError(f"Unsupported evidence kind {kind!r}. Valid kinds: {sorted(VALID_EVIDENCE_KINDS)}")
        if not operation or not operation.strip():
            raise ValueError("operation must be a non-empty string.")
        if not snapshot_id or not snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string.")

        df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
        nested = [name for name, dtype in df.schema.items() if isinstance(dtype, (pl.List, pl.Struct, pl.Array))]
        if nested:
            raise ValueError(f"EvidenceStore only accepts flat frames; nested column(s) {nested} are not supported.")
        normalized_params = dict(params or {})
        schema_signature = compute_schema_signature(df)
        fingerprint = _canonical_json(
            {
                "kind": kind,
                "operation": operation.strip(),
                "params": normalized_params,
                "snapshot_id": snapshot_id,
                "sql_text": sql_text.strip(),
                "schema_signature": schema_signature,
                "data_hash": compute_frame_hash(df),
            }
        )
        evidence_id = f"ev-{_digest(fingerprint)[:12]}"

        artifact = self.evidence_dir / f"{evidence_id}.parquet"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        with staged_temp_path(artifact) as temporary:
            df.write_parquet(temporary)

        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,
            operation=operation.strip(),
            purpose=purpose.strip(),
            sql_text=sql_text.strip(),
            snapshot_id=snapshot_id,
            params=normalized_params,
            row_count=df.height,
            schema_signature=schema_signature,
            created_at_ns=int(created_at_ns) if created_at_ns is not None else time.time_ns(),
            artifact_path=artifact.relative_to(self.benchmark_dir),
        )
        record_dict = record.to_dict()
        row = pl.DataFrame(
            [{name: record_dict[name] for name in EVIDENCE_INDEX_SCHEMA.names()}],
            schema=EVIDENCE_INDEX_SCHEMA,
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.index_path):
            if self.index_path.is_file():
                existing = pl.read_parquet(self.index_path)
                row = pl.concat(
                    [existing.filter(pl.col("evidence_id") != evidence_id), row],
                    how="vertical_relaxed",
                )
            with staged_temp_path(self.index_path) as temporary:
                row.write_parquet(temporary)
        return record

    def load_index(self) -> pl.DataFrame:
        """Load the evidence index as a DataFrame, or an empty index frame if none exists."""
        return pl.read_parquet(self.index_path) if self.index_path.is_file() else pl.DataFrame(schema=EVIDENCE_INDEX_SCHEMA)

    def get(self, evidence_id: str) -> EvidenceRecord:
        """Retrieve an ``EvidenceRecord`` by ID from ``evidence_index.parquet``."""
        if (matched := self.load_index().filter(pl.col("evidence_id") == evidence_id)).is_empty():
            raise KeyError(f"Evidence record '{evidence_id}' not found.")
        return EvidenceRecord.from_index_row(matched.row(0, named=True))

    def read_frame(self, evidence_id: str) -> pl.DataFrame:
        """Read the persisted Parquet frame for ``evidence_id`` directly from disk."""
        absolute = self.benchmark_dir / self.get(evidence_id).artifact_path
        if not absolute.is_file():
            raise KeyError(f"Evidence frame '{evidence_id}' not found at {absolute}.")
        return pl.read_parquet(absolute)

    def list_records(self) -> tuple[EvidenceRecord, ...]:
        """Return all registered ``EvidenceRecord`` entries from the index sorted by ID."""
        return tuple(EvidenceRecord.from_index_row(row) for row in self.load_index().sort("evidence_id").to_dicts())
