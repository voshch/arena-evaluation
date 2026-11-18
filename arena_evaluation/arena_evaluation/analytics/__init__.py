"""Polars OLAP Analytics Core (Layer 3).

Exposes :class:`BenchmarkCatalog`, :class:`PolarsSQLEngine`, :class:`QueryCatalog`,
:class:`QueryDeclaration`, :class:`EvidenceStore`, and the computation registry.
"""

from __future__ import annotations

from .catalog import (
    CANONICAL_VIEW_NAMES,
    SERIES_TABLE_PREFIX,
    BenchmarkCatalog,
)
from .computations import (
    COMPUTATIONS,
    Computation,
    get_computation,
    list_computations,
    register_computation,
    run_computation,
)
from .evidence import (
    EVIDENCE_INDEX_SCHEMA,
    VALID_EVIDENCE_KINDS,
    AnalysisResult,
    EvidenceRecord,
    EvidenceStore,
    compute_snapshot_id,
)
from .queries import (
    QueryCatalog,
    QueryDeclaration,
)
from .sql_engine import (
    PolarsSQLEngine,
    SQLGuardError,
    validate_readonly_sql,
)

__all__ = [
    "AnalysisResult",
    "BenchmarkCatalog",
    "CANONICAL_VIEW_NAMES",
    "COMPUTATIONS",
    "Computation",
    "EVIDENCE_INDEX_SCHEMA",
    "EvidenceRecord",
    "EvidenceStore",
    "PolarsSQLEngine",
    "QueryCatalog",
    "QueryDeclaration",
    "SERIES_TABLE_PREFIX",
    "SQLGuardError",
    "VALID_EVIDENCE_KINDS",
    "compute_snapshot_id",
    "get_computation",
    "list_computations",
    "register_computation",
    "run_computation",
    "validate_readonly_sql",
]
