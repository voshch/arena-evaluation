from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl

from .acoustic_field import AcousticFieldAnimationRenderer, AcousticFieldRenderer
from .bar import BarRenderer
from .base import BasePlotRenderer
from .box import BoxRenderer
from .breakdown import BreakdownRenderer
from .heatmap import HeatmapRenderer
from .hierarchy import HierarchyRenderer
from .histogram import HistogramRenderer
from .interval import IntervalRenderer
from .line import LineRenderer
from .radar import RadarRenderer
from .rank_swap import RankSwapRenderer
from .scatter import ScatterRenderer
from .table import TableRenderer
from .timeseries import TimeseriesRenderer
from .trajectory import TrajectoryRenderer
from .trajectory_slice import TrajectorySliceRenderer
from .violin import ViolinRenderer

RENDERER_REGISTRY: Mapping[str, type[BasePlotRenderer]] = {
    "acoustic_field": AcousticFieldRenderer,
    "acoustic_field_animation": AcousticFieldAnimationRenderer,
    "bar": BarRenderer,
    "box": BoxRenderer,
    "breakdown": BreakdownRenderer,
    "heatmap": HeatmapRenderer,
    "hierarchy": HierarchyRenderer,
    "histogram": HistogramRenderer,
    "interval": IntervalRenderer,
    "line": LineRenderer,
    "radar": RadarRenderer,
    "rank_swap": RankSwapRenderer,
    "scatter": ScatterRenderer,
    "table": TableRenderer,
    "timeseries": TimeseriesRenderer,
    "trajectory": TrajectoryRenderer,
    "trajectory_slice": TrajectorySliceRenderer,
    "violin": ViolinRenderer,
}

if TYPE_CHECKING:
    from ..checks import PlotDiagnostics
    from ..viz_manifest import PlotSpec


def prepare_renderer(
    spec: PlotSpec,
    df: pl.DataFrame,
    *,
    units: dict[str, str] | None = None,
    run_dir: pathlib.Path | None = None,
    world_geometry: pl.DataFrame | None = None,
    diagnostics: PlotDiagnostics | None = None,
) -> BasePlotRenderer:
    """Instantiate, configure, and validate the renderer registered for ``spec.type``."""
    from arena_evaluation.presentation.color_utils import set_global_color_palette

    set_global_color_palette()
    renderer_cls = RENDERER_REGISTRY.get(spec.type)
    if not renderer_cls:
        raise ValueError(f"Plot {spec.id!r}: unknown plot type {spec.type!r}.")
    renderer = renderer_cls(spec, units=units)
    renderer.run_dir = run_dir
    renderer.world_geometry = world_geometry
    renderer.diagnostics = diagnostics
    renderer.validate_frame(df)
    return renderer


__all__ = [
    "RENDERER_REGISTRY",
    "prepare_renderer",
    "AcousticFieldAnimationRenderer",
    "AcousticFieldRenderer",
    "BarRenderer",
    "BasePlotRenderer",
    "BoxRenderer",
    "BreakdownRenderer",
    "HeatmapRenderer",
    "HierarchyRenderer",
    "HistogramRenderer",
    "IntervalRenderer",
    "LineRenderer",
    "RadarRenderer",
    "RankSwapRenderer",
    "ScatterRenderer",
    "TableRenderer",
    "TimeseriesRenderer",
    "TrajectoryRenderer",
    "TrajectorySliceRenderer",
    "ViolinRenderer",
]
