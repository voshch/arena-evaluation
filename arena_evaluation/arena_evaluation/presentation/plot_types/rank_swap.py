"""Rank-swap plot renderer for pre-computed swap counts and distances."""

from __future__ import annotations

import pathlib

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


class RankSwapRenderer(BasePlotRenderer):
    """Draw one bar per x from pre-computed rank swap counts or distances."""

    PLOT_TYPE = "rank_swap"

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Require the x and y channels the bars are drawn from."""
        super().validate_frame(df)
        self._resolve_columns(df)

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str]:
        x_col, y_col = self.resolve_channel("x"), self.resolve_channel("y")
        if x_col is None or y_col is None:
            channel = "x" if x_col is None else "y"
            raise ValueError(
                f"Plot {self.spec.id!r}: rank_swap plot requires channel {channel!r}."
            )
        self.validate_unique_cells(df, [x_col])
        return x_col, y_col

    def _plot_data(self, df: pl.DataFrame) -> tuple[list[str], np.ndarray, str, str] | None:
        if df.is_empty():
            return None
        x_col, y_col = self._resolve_columns(df)
        x_title = x_col.replace("_", " ").title()
        y_title = self.format_label(y_col.replace("_", " ").title(), y_col)
        return label_values(df, x_col).tolist(), series_values(df, y_col), x_title, y_title

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the interactive rank-swap figure without serializing it."""
        data = self._plot_data(df)
        if data is None:
            return None
        labels, values, x_title, y_title = data
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values.tolist(),
                    marker_color=palette_color(get_color_palette(), 1),
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            xaxis_title=x_title,
            yaxis_title=y_title,
        )
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly rank-swap bar chart."""
        fig = self.build_plotly_figure(df)
        return None if fig is None else fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the static rank-swap figure without writing it."""
        data = self._plot_data(df)
        if data is None:
            return None
        labels, values, x_title, y_title = data
        positions = np.arange(len(labels))
        fig, ax = new_figure()
        ax.bar(positions, values, color=palette_color(get_color_palette(), 1))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=x_title,
            y_label=y_title,
        )
        return fig

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib rank-swap bar chart."""
        fig = self.build_static_figure(df)
        if fig is None:
            return
        self.save_figure(fig, out_path)
        plt.close(fig)
