"""Interactive Plotly plot dispatcher."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import polars as pl

from arena_evaluation.presentation.viz_manifest import PlotSpec

from .plot_types import prepare_renderer

if TYPE_CHECKING:
    from .checks import PlotDiagnostics


class PlotlyRenderer:
    """Dispatches plot rendering to the correct Plotly class."""

    def __init__(self, units: dict[str, str] | None = None) -> None:
        self.units = units or {}

    def render(
        self,
        spec: PlotSpec,
        df: pl.DataFrame,
        run_dir: pathlib.Path | None = None,
        *,
        world_geometry: pl.DataFrame | None = None,
        diagnostics: PlotDiagnostics | None = None,
    ) -> str | list[str] | None:
        renderer = prepare_renderer(
            spec,
            df,
            units=self.units,
            run_dir=run_dir,
            world_geometry=world_geometry,
            diagnostics=diagnostics,
        )
        return renderer.render_plotly(df)
