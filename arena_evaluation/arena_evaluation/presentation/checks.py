"""Technical quality-gate checks and diagnostics for report plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from arena_evaluation.presentation.viz_manifest import PlotSpec

__all__ = [
    "BuildChecks",
    "CheckStage",
    "DiagnosticInspector",
    "PlotCheck",
    "PlotDiagnostics",
]


@dataclass(slots=True, kw_only=True)
class PlotDiagnostics:
    """Technical diagnostic metadata collected when resolving and checking a plot frame."""

    status: str = "PASSED"
    issues: list[str] = field(default_factory=list)
    row_count: int = 0
    evidence_id: str = ""
    snapshot_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    snapshot_ids: tuple[str, ...] = ()


class CheckStage(StrEnum):
    """The build step that refused a plot."""

    LOAD = "load"
    QUALITY = "quality gate"
    STATIC = "static plot"
    INTERACTIVE = "interactive plot"


@dataclass(frozen=True, slots=True, kw_only=True)
class PlotCheck:
    """One plot that failed a build step, with the reason the build recorded."""

    plot_id: str
    title: str
    stage: CheckStage
    detail: str


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildChecks:
    """The failed checks of one report build against the plots its manifest asked for."""

    plots_total: int
    failures: tuple[PlotCheck, ...]

    @property
    def failed_count(self) -> int:
        """Plots that failed at least one build step."""
        return len({failure.plot_id for failure in self.failures})

    @property
    def passed_count(self) -> int:
        """Plots that rendered on every step of the build."""
        plot_failures = len(
            {
                failure.plot_id
                for failure in self.failures
                if failure.plot_id != "summary"
            }
        )
        return max(0, self.plots_total - plot_failures)

    def for_plot(self, plot_id: str) -> tuple[PlotCheck, ...]:
        """Every failed step of one plot, in build order."""
        return tuple(failure for failure in self.failures if failure.plot_id == plot_id)


class DiagnosticInspector:
    """Runs technical quality-gate checks on a resolved plot DataFrame."""

    @staticmethod
    def inspect(df: pl.DataFrame, spec: PlotSpec | None = None) -> PlotDiagnostics:
        """Verify technical preconditions (non-empty frame, required columns, min_rows)."""
        # differs from overhaul/source:checks.py:108-256: technical checks only, no advisory warnings
        issues: list[str] = []
        schema = df.schema
        row_count = df.height

        if row_count == 0:
            issues.append("Resolved DataFrame contains 0 rows")

        if spec is not None and spec.checks:
            req_cols = spec.checks.get("require_columns", [])
            if isinstance(req_cols, list):
                for col in req_cols:
                    if isinstance(col, str) and schema.get(col) is None:
                        issues.append(
                            f"Required column '{col}' not found in data frame"
                        )
            min_rows = spec.checks.get("min_rows")
            if isinstance(min_rows, int) and row_count < min_rows:
                issues.append(f"Row count {row_count} is below min_rows={min_rows}")

        return PlotDiagnostics(
            status="FAILED" if issues else "PASSED",
            issues=issues,
            row_count=row_count,
        )
