"""Interval plot renderer for point estimates with pre-computed bound columns."""

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

_LOW_OPTION: str = "low"
_HIGH_OPTION: str = "high"
_CAP_HALF_WIDTH: float = 0.12


class IntervalRenderer(BasePlotRenderer):
    """Draw point estimates with the interval stored in two bound columns.

    The bound columns are named by the ``low`` and ``high`` options; a spec
    without both options is refused with a hint.
    """

    PLOT_TYPE = "interval"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        _LOW_OPTION,
        _HIGH_OPTION,
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({_LOW_OPTION, _HIGH_OPTION})

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Require the declared interval bounds before the plot is drawn."""
        super().validate_frame(df)
        self._bound_columns()

    def _bound_columns(self) -> tuple[str, str]:
        options = self.spec.options
        low = options.get(_LOW_OPTION)
        high = options.get(_HIGH_OPTION)
        low_col = low.strip() if isinstance(low, str) and low.strip() else ""
        high_col = high.strip() if isinstance(high, str) and high.strip() else ""
        if not low_col or not high_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: interval plot requires options "
                f"{_LOW_OPTION!r} and {_HIGH_OPTION!r} naming the lower and upper "
                "bound columns, e.g. options: "
                "{'low': 'ci_low', 'high': 'ci_high'}."
            )
        return low_col, high_col

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str, str, str]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if x_col is None:
            raise ValueError(
                f"Plot {self.spec.id!r}: interval plot requires channel 'x'."
            )
        if y_col is None:
            raise ValueError(
                f"Plot {self.spec.id!r}: interval plot requires channel 'y'."
            )
        low_col, high_col = self._bound_columns()
        self.validate_unique_cells(df, [x_col])
        return x_col, y_col, low_col, high_col

    def _plot_data(
        self, df: pl.DataFrame
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        x_col, y_col, low_col, high_col = self._resolve_columns(df)
        return (
            label_values(df, x_col).tolist(),
            series_values(df, y_col),
            series_values(df, low_col),
            series_values(df, high_col),
        )

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the interactive interval figure without serializing it."""
        if df.is_empty():
            return None
        labels, values, low, high = self._plot_data(df)
        x_col = self.resolve_channel("x") or ""
        y_col = self.resolve_channel("y") or ""
        color = palette_color(get_color_palette(), 0)
        whisker_x: list[str | None] = []
        whisker_y: list[float | None] = []
        for label, low_value, high_value in zip(
            labels, low.tolist(), high.tolist(), strict=True
        ):
            whisker_x.extend([label, label, None])
            whisker_y.extend([low_value, high_value, None])
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=whisker_x,
                    y=whisker_y,
                    mode="lines",
                    line={"color": color},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter(
                    x=labels,
                    y=values.tolist(),
                    mode="markers",
                    marker={"color": color},
                    showlegend=False,
                ),
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=self.format_label(y_col.replace("_", " ").title(), y_col),
        )
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly interval chart with error bars."""
        fig = self.build_plotly_figure(df)
        if fig is None:
            return None
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the static interval figure without writing it."""
        if df.is_empty():
            return None
        labels, values, low, high = self._plot_data(df)
        x_col = self.resolve_channel("x") or ""
        y_col = self.resolve_channel("y") or ""
        fig, ax = new_figure()
        positions = np.arange(len(labels))
        color = palette_color(get_color_palette(), 0)
        ax.vlines(positions, low, high, color=color, linewidth=1.5)
        for bound in (low, high):
            ax.hlines(
                bound,
                positions - _CAP_HALF_WIDTH,
                positions + _CAP_HALF_WIDTH,
                color=color,
                linewidth=1.5,
            )
        ax.plot(positions, values, "o", color=color)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=x_col.replace("_", " ").title(),
            y_label=self.format_label(y_col.replace("_", " ").title(), y_col),
        )
        return fig

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib interval plot with error bars."""
        fig = self.build_static_figure(df)
        if fig is None:
            return
        self.save_figure(fig, out_path)
        plt.close(fig)
