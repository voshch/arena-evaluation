from __future__ import annotations

import pathlib
import polars as pl
from arena_evaluation.storage.schemas import PlotSpec

class PlotlyRenderer:
    """Dispatches plot rendering to the correct Plotly class."""

    def __init__(self, units: dict[str, str] | None = None):
        self.units = units or {}
        from .color_utils import set_global_color_palette
        set_global_color_palette()

        from .plot_types import (
            ViolinRenderer,
            BoxRenderer,
            BarRenderer,
            TrajectoryRenderer,
            RadarRenderer,
            ScatterRenderer,
            HistogramRenderer,
            HeatmapRenderer,
            TimeseriesRenderer,
            LineRenderer,
            TableRenderer,
            AcousticFieldRenderer,
            AcousticFieldAnimationRenderer,
        )

        self.renderers = {
            "violin": ViolinRenderer,
            "box": BoxRenderer,
            "bar": BarRenderer,
            "trajectory": TrajectoryRenderer,
            "radar": RadarRenderer,
            "scatter": ScatterRenderer,
            "histogram": HistogramRenderer,
            "heatmap": HeatmapRenderer,
            "timeseries": TimeseriesRenderer,
            "line": LineRenderer,
            "table": TableRenderer,
            "acoustic_field": AcousticFieldRenderer,
            "acoustic_field_animation": AcousticFieldAnimationRenderer,
        }

    def render(self, spec: PlotSpec, df: pl.DataFrame, run_dir: pathlib.Path | None = None) -> str | list[str] | None:
        renderer_cls = self.renderers.get(spec.type)
        if not renderer_cls:
            return None

        renderer = renderer_cls(spec, units=self.units)
        renderer.run_dir = run_dir
        if renderer_cls.__name__ == "TrajectoryRenderer":
            renderer.generate_gifs = False  # GIFs are seaborn-only
        return renderer.render_plotly(df)
