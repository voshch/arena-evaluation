"""Trajectory-slice plot renderer for pre-recorded XY path samples."""

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
    new_figure,
    palette_color,
    series_values,
)


class TrajectorySliceRenderer(BasePlotRenderer):
    """Draw a pre-recorded XY grid path as one line in both renderers."""

    PLOT_TYPE = "trajectory_slice"

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Require the x and y columns the path is read from."""
        super().validate_frame(df)
        self._resolve_columns(df)

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str]:
        x_col = self.resolve_channel("x", "x") or "x"
        y_col = self.resolve_channel("y", "y") or "y"
        missing = [name for name in (x_col, y_col) if name not in df.columns]
        if missing:
            missing_text = ", ".join(repr(name) for name in missing)
            raise ValueError(
                f"Plot {self.spec.id!r}: trajectory_slice plot requires columns "
                f"{missing_text}; bind channels 'x' and 'y'."
            )
        return x_col, y_col

    def _plot_data(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_col, y_col = self._resolve_columns(df)
        return series_values(df, x_col), series_values(df, y_col)

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the interactive trajectory-slice figure without serializing it."""
        if df.is_empty():
            return None
        x_values, y_values = self._plot_data(df)
        x_col = self.resolve_channel("x", "x") or "x"
        y_col = self.resolve_channel("y", "y") or "y"
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_values.tolist(),
                    y=y_values.tolist(),
                    mode="lines",
                    line={"color": palette_color(get_color_palette(), 2)},
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
        )
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly trajectory-slice line."""
        fig = self.build_plotly_figure(df)
        if fig is None:
            return None
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the static trajectory-slice figure without writing it."""
        if df.is_empty():
            return None
        x_values, y_values = self._plot_data(df)
        x_col = self.resolve_channel("x", "x") or "x"
        y_col = self.resolve_channel("y", "y") or "y"
        fig, ax = new_figure()
        ax.plot(
            x_values,
            y_values,
            color=palette_color(get_color_palette(), 2),
        )
        ax.set_aspect("equal", adjustable="datalim")
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=x_col.replace("_", " ").title(),
            y_label=y_col.replace("_", " ").title(),
        )
        return fig

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib trajectory-slice line."""
        fig = self.build_static_figure(df)
        if fig is None:
            return
        self.save_figure(fig, out_path)
        plt.close(fig)
