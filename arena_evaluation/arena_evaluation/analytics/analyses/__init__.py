"""Named statistical analysis procedures returning canonical AnalysisResult objects."""

from __future__ import annotations

from arena_evaluation.analytics.analyses.procedures import (
    AttributionSharesParams,
    CompareParams,
    LeadMeasuresParams,
    ProcedureSpec,
    RankingStabilityParams,
    TriageParams,
    VisibilityEffectParams,
    get_procedure,
    list_procedures,
    run_procedure,
)

__all__ = [
    "AttributionSharesParams",
    "CompareParams",
    "LeadMeasuresParams",
    "ProcedureSpec",
    "RankingStabilityParams",
    "TriageParams",
    "VisibilityEffectParams",
    "get_procedure",
    "list_procedures",
    "run_procedure",
]
