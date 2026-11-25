"""Report rendering. The heavy members load on first access."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report_builder import ReportBuilder
    from .viz_manifest import VizManifest

__all__ = ["ReportBuilder", "VizManifest"]


def __getattr__(name: str) -> object:
    """Import one of the two declared heavy members on first access."""
    match name:
        case "ReportBuilder":
            from .report_builder import ReportBuilder

            return ReportBuilder
        case "VizManifest":
            from .viz_manifest import VizManifest

            return VizManifest
        case _:
            raise AttributeError(name)
