"""Static plot dispatcher backed by Seaborn and Matplotlib renderers."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import polars as pl

from arena_evaluation.presentation.checks import PlotRenderError
from arena_evaluation.presentation.viz_manifest import PlotSpec

from .plot_types import prepare_renderer

if TYPE_CHECKING:
    from .checks import PlotDiagnostics


class StaticRenderer:
    """Dispatches a plot specification to the renderer class that draws its static artifact."""

    def __init__(self, *, units: dict[str, str] | None = None) -> None:
        self.units = units or {}

    def render(
        self,
        spec: PlotSpec,
        df: pl.DataFrame,
        out_path: pathlib.Path,
        run_dir: pathlib.Path | None = None,
        *,
        world_geometry: pl.DataFrame | None = None,
        diagnostics: PlotDiagnostics | None = None,
    ) -> None:
        renderer = prepare_renderer(
            spec,
            df,
            units=self.units,
            run_dir=run_dir,
            world_geometry=world_geometry,
            diagnostics=diagnostics,
        )
        renderer.render_static(df, out_path)
        if not pathlib.Path(out_path).is_file():
            detail = f": {'; '.join(diagnostics.issues)}" if diagnostics is not None and diagnostics.issues else ""
            raise PlotRenderError(f"Plot {spec.id!r}: {spec.type} renderer produced no file at {out_path}{detail}; the frame holds nothing the renderer can draw.")
