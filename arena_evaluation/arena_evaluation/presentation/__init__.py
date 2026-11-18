"""Report rendering. The heavy members load on first access."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .report_builder import ReportBuilder
    from .viz_manifest import VizManifest

__all__ = ["ReportBuilder", "VizManifest"]


def __getattr__(name: str) -> object:
    if name == "ReportBuilder":
        from .report_builder import ReportBuilder

        return ReportBuilder
    if name == "VizManifest":
        from .viz_manifest import VizManifest

        return VizManifest
    try:
        return importlib.import_module(f".{name}", __package__)
    except ModuleNotFoundError:
        raise AttributeError(name) from None
