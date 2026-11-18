"""Heatmap plot renderer for pre-computed 2D matrices and correlation tables."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import seaborn as sns

from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    label_values,
    new_figure,
)


class HeatmapRenderer(BasePlotRenderer):
    """Renders 2D heatmaps from pre-computed matrix or long-form table rows."""

    PLOT_TYPE = "heatmap"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "value_col",
        "colorscale",
        "annot",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"value_col"})

    def _extract_matrix(
        self, df: pl.DataFrame
    ) -> tuple[list[str], list[str], np.ndarray] | None:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        z_col = self.resolve_channel("color")
        if not z_col:
            opt_z = self.spec.options.get("value_col")
            if isinstance(opt_z, str) and opt_z.strip():
                z_col = opt_z.strip()

        if x_col and y_col and z_col:
            self.validate_unique_cells(df, [x_col, y_col])
            x_arr = label_values(df, x_col)
            y_arr = label_values(df, y_col)
            z_arr = df[z_col].cast(pl.Float64).to_numpy()
            x_labels = sorted(set(x_arr.tolist()))
            y_labels = sorted(set(y_arr.tolist()))
            x_idx = {v: i for i, v in enumerate(x_labels)}
            y_idx = {v: i for i, v in enumerate(y_labels)}
            matrix = np.full((len(y_labels), len(x_labels)), np.nan, dtype=float)
            for xv, yv, zv in zip(x_arr, y_arr, z_arr, strict=True):
                if zv is not None and not np.isnan(zv):
                    matrix[y_idx[yv], x_idx[xv]] = float(zv)
            return x_labels, y_labels, matrix

        raise ValueError(
            f"Plot {self.spec.id!r}: heatmap requires channels 'x', 'y', and 'color' (or option 'value_col')."
        )

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly heatmap."""
        # differs from overhaul/source:plot_types/heatmap.py:65-79: binds pre-computed matrices without .corr() or .mean()
        if df.is_empty():
            return None
        extracted = self._extract_matrix(df)
        if extracted is None:
            return None
        x_labels, y_labels, matrix = extracted
        colorscale = str(self.spec.options.get("colorscale", "Viridis"))

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.tolist(),
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                hoverongaps=False,
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or "Heatmap",
            margin=dict(t=55, b=80, l=90, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn heatmap."""
        if df.is_empty():
            return
        extracted = self._extract_matrix(df)
        if extracted is None:
            return
        x_labels, y_labels, matrix = extracted
        cmap = str(self.spec.options.get("colorscale", "viridis"))
        if cmap == "Viridis":
            cmap = "viridis"

        fig, ax = new_figure(
            width_in=max(7.0, len(x_labels) * 0.9),
            height_in=max(5.0, len(y_labels) * 0.6),
        )
        annot_opt = self.spec.options.get("annot")
        annot = bool(annot_opt) if annot_opt is not None else matrix.size <= 64
        sns.heatmap(
            matrix,
            xticklabels=x_labels,
            yticklabels=y_labels,
            cmap=cmap,
            mask=np.isnan(matrix),
            annot=annot,
            fmt=".2f",
            ax=ax,
        )
        ax.tick_params(axis="x", rotation=35)
        finish_axes(
            ax,
            title=self.spec.title or "Heatmap",
            x_label="",
            y_label="",
            grid_axis="both",
        )
        ax.grid(False)
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
