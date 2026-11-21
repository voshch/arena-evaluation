"""Technical quality-gate checks and diagnostics for report plots."""

from __future__ import annotations

import dataclasses
import pathlib
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from arena_evaluation.presentation.viz_manifest import PlotSpec

__all__ = [
    "BuildChecks",
    "CheckStage",
    "DiagnosticInspector",
    "PlotCheck",
    "PlotDiagnostics",
    "PlotRenderError",
    "image_is_blank",
    "null_reason_of",
]

# A rendered figure whose luma never leaves this band drew no axes, no text and no marks.
_BLANK_LUMA_SPREAD: Final[int] = 8
# The page-level summary table is a manifest feature, not a declared plot.
_SUMMARY_PLOT_ID: Final[str] = "summary"


class PlotRenderError(RuntimeError):
    """A renderer refused a frame it cannot draw; the builder records it as a failed check."""


@dataclasses.dataclass(slots=True, kw_only=True)
class PlotDiagnostics:
    """Technical diagnostic metadata collected when resolving and checking a plot frame."""

    status: str = "PASSED"
    issues: list[str] = dataclasses.field(default_factory=list)
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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PlotCheck:
    """One plot that failed a build step, with the reason the build recorded."""

    plot_id: str
    title: str
    stage: CheckStage
    detail: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BuildChecks:
    """The failed checks of one report build against the plots its manifest asked for."""

    plots_total: int
    failures: tuple[PlotCheck, ...]

    @property
    def failed_count(self) -> int:
        """Declared plots that failed at least one build step (the summary pseudo-plot excluded)."""
        return len({failure.plot_id for failure in self.failures if failure.plot_id != _SUMMARY_PLOT_ID})

    @property
    def passed_count(self) -> int:
        """Plots that rendered on every step of the build."""
        return max(0, self.plots_total - self.failed_count)

    @property
    def has_failures(self) -> bool:
        """Whether any build step failed, the page-level summary included."""
        return bool(self.failures)

    def for_plot(self, plot_id: str) -> tuple[PlotCheck, ...]:
        """Every failed step of one plot, in build order."""
        return tuple(failure for failure in self.failures if failure.plot_id == plot_id)


def null_reason_of(df: pl.DataFrame, column: str) -> str:
    """The recorded reason for a NULL value of ``column``, preferring ``<column>_null_reason``."""
    for candidate in (f"{column}_null_reason", "null_reason"):
        if candidate not in df.columns:
            continue
        for value in df[candidate].drop_nulls().head(5):
            reason = str(value).strip()
            if reason:
                return reason
    return ""


def _declared_value_columns(spec: PlotSpec) -> tuple[str, ...]:
    """Every column the spec binds to a value channel, in declaration order and deduplicated."""
    channels = spec.channels
    names: list[str] = []
    for value in (channels.x, channels.y, channels.color, channels.size):
        candidates = [value] if isinstance(value, str) else (value or ())
        names.extend(name.strip() for name in candidates if isinstance(name, str) and name.strip())
    return tuple(dict.fromkeys(names))


def image_is_blank(path: pathlib.Path) -> bool:
    """Whether a rendered image is one flat colour, i.e. the renderer drew nothing."""
    from PIL import Image

    with Image.open(path) as image:
        luma = image.convert("L")
        luma.thumbnail((128, 128))
        low, high = luma.getextrema()
    assert isinstance(low, (int, float)) and isinstance(high, (int, float))
    return int(high) - int(low) < _BLANK_LUMA_SPREAD


class DiagnosticInspector:
    """Runs technical quality-gate checks on a resolved plot DataFrame."""

    @staticmethod
    def inspect(df: pl.DataFrame, spec: PlotSpec | None = None) -> PlotDiagnostics:
        """Verify technical preconditions: rows, declared columns, all-NULL metrics, min_rows."""
        # differs from overhaul/source:checks.py:108-256: technical checks only, no advisory warnings
        issues: list[str] = []
        schema = df.schema
        row_count = df.height

        if row_count == 0:
            issues.append("Resolved DataFrame contains 0 rows")

        if spec is not None:
            checks = spec.checks
            declared = _declared_value_columns(spec)
            for column in dict.fromkeys([*checks.require_columns, *declared]):
                if schema.get(column) is None:
                    issues.append(f"Required column '{column}' not found in data frame")
            if row_count > 0:
                exempt = frozenset(checks.allow_all_null)
                for column in declared:
                    if column in exempt or schema.get(column) is None:
                        continue
                    if not (schema[column].is_numeric() or schema[column] == pl.Null):
                        continue
                    if df[column].null_count() != row_count:
                        continue
                    reason = null_reason_of(df, column)
                    suffix = f" (reason: {reason})" if reason else ""
                    issues.append(f"Declared metric '{column}' is 100% NULL{suffix}")
            if checks.min_rows is not None and row_count < checks.min_rows:
                issues.append(f"Row count {row_count} is below min_rows={checks.min_rows}")

        return PlotDiagnostics(
            status="FAILED" if issues else "PASSED",
            issues=issues,
            row_count=row_count,
        )
