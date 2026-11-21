"""Histogram plot renderer for pre-binned distribution tables."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    label_values,
    new_figure,
    palette_color,
    series_values,
)


class HistogramRenderer(BasePlotRenderer):
    """Renders pre-binned distribution_bins frames as bars without in-renderer binning."""

    PLOT_TYPE = "histogram"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "bin_low",
        "bin_high",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"bin_low", "bin_high"})

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str, str | None]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if not x_col or not y_col:
            raise ValueError(f"Plot {self.spec.id!r}: histogram requires pre-binned 'x' (e.g. bin_center) and 'y' (e.g. count or density).")
        color_col = self.resolve_channel("color")
        cell_keys = [c for c in dict.fromkeys((x_col, color_col)) if c is not None]
        self.validate_unique_cells(df, cell_keys)
        return x_col, y_col, color_col

    def _bin_columns(self) -> tuple[str, str] | None:
        low_col = self.spec.options.get("bin_low")
        high_col = self.spec.options.get("bin_high")
        if isinstance(low_col, str) and low_col.strip() and isinstance(high_col, str) and high_col.strip():
            return low_col.strip(), high_col.strip()
        return None

    def _bar_geometry(self, sub: pl.DataFrame, x_col: str) -> tuple[np.ndarray, np.ndarray | None]:
        bins = self._bin_columns()
        if bins is None:
            return series_values(sub, x_col), None
        low_edges, high_edges = (series_values(sub, column) for column in bins)
        return 0.5 * (low_edges + high_edges), high_edges - low_edges

    def _bar_groups(self, df: pl.DataFrame, color_col: str | None) -> list[tuple[str | None, pl.DataFrame]]:
        if color_col is None:
            return [(None, df)]
        labels = label_values(df, color_col)
        return [(label, df.filter(pl.Series(labels == label))) for label in sorted(set(labels.tolist()))]

    def _label(self, column: str) -> str:
        return self.format_label(column.replace("_", " ").title(), column)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly histogram from pre-binned counts."""
        # differs from overhaul/source:plot_types/histogram.py:72: renders pre-binned distribution_bins only
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        palette = get_color_palette()
        fig = go.Figure()

        for idx, (label, sub) in enumerate(self._bar_groups(df, color_col)):
            centers, widths = self._bar_geometry(sub, x_col)
            bar: dict[str, object] = {
                "x": centers.tolist(),
                "y": series_values(sub, y_col).tolist(),
                "name": label if label is not None else y_col,
                "marker_color": palette_color(palette, idx),
            }
            if widths is not None:
                bar["width"] = widths.tolist()
            if color_col is not None:
                bar["opacity"] = 0.75
            fig.add_trace(go.Bar(**bar))

        x_title = self._label(x_col)
        y_title = self._label(y_col)
        fig.update_layout(barmode="overlay")
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or x_title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib histogram from pre-binned counts."""
        if df.is_empty():
            return
        x_col, y_col, color_col = self._resolve_columns(df)
        palette = get_color_palette()
        fig, ax = new_figure()

        for idx, (label, sub) in enumerate(self._bar_groups(df, color_col)):
            centers, widths = self._bar_geometry(sub, x_col)
            bar: dict[str, object] = {
                "width": widths if widths is not None else 0.8,
                "color": palette_color(palette, idx),
            }
            if color_col is not None:
                bar.update(alpha=0.65, label=label)
            ax.bar(centers, series_values(sub, y_col), **bar)
        if color_col is not None:
            ax.legend(title=color_col)

        x_title = self._label(x_col)
        y_title = self._label(y_col)
        finish_axes(
            ax,
            title=self.spec.title or x_title,
            x_label=x_title,
            y_label=y_title,
        )
        self.save_figure(fig, out_path)
        plt.close(fig)
