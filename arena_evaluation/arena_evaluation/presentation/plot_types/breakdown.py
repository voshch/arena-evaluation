"""Breakdown plot renderer for stacked attribution shares."""

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

_BreakdownStack = tuple[list[str], list[str], dict[str, np.ndarray], str, str, str]


class BreakdownRenderer(BasePlotRenderer):
    """Draw one stacked bar per x, one segment and legend entry per component."""

    PLOT_TYPE = "breakdown"

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Require the colour channel that names the stacked components."""
        super().validate_frame(df)
        self._resolve_columns(df)

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str, str]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        color_col = self.resolve_channel("color")
        if x_col is None or y_col is None or color_col is None:
            missing_name = "x" if x_col is None else ("y" if y_col is None else "color")
            hint = " naming the component column to stack, e.g. {'color': 'component'}." if missing_name == "color" else "."
            raise ValueError(f"Plot {self.spec.id!r}: breakdown plot requires channel {missing_name!r}{hint}")
        self.validate_unique_cells(df, [x_col, color_col])
        return x_col, y_col, color_col

    def _prepare(self, df: pl.DataFrame) -> _BreakdownStack | None:
        """Resolve the stack once, or ``None`` for an empty frame."""
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        x_values = label_values(df, x_col).tolist()
        component_values = label_values(df, color_col).tolist()
        x_labels = list(dict.fromkeys(x_values))
        components = list(dict.fromkeys(component_values))
        positions = {label: index for index, label in enumerate(x_labels)}
        rows = {component: np.zeros(len(x_labels)) for component in components}
        for x_label, component, value in zip(x_values, component_values, series_values(df, y_col), strict=True):
            if not np.isnan(value):
                rows[component][positions[x_label]] = float(value)
        return (
            x_labels,
            components,
            rows,
            color_col,
            x_col.replace("_", " ").title(),
            self.format_label(y_col.replace("_", " ").title(), y_col),
        )

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the stacked Plotly figure without serializing it."""
        prepared = self._prepare(df)
        if prepared is None:
            return None
        x_labels, components, rows, _, _, _ = prepared
        palette = get_color_palette()
        fig = go.Figure(
            data=[
                go.Bar(
                    name=component,
                    x=x_labels,
                    y=rows[component].tolist(),
                    marker_color=palette_color(palette, index),
                )
                for index, component in enumerate(components)
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            barmode="stack",
        )
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive stacked Plotly breakdown chart."""
        fig = self.build_plotly_figure(df)
        return None if fig is None else fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the stacked static figure without writing it."""
        prepared = self._prepare(df)
        if prepared is None:
            return None
        x_labels, components, rows, color_col, x_title, y_title = prepared
        palette = get_color_palette()
        positions = np.arange(len(x_labels))
        bottom = np.zeros(len(x_labels))
        fig, ax = new_figure()
        for index, component in enumerate(components):
            values = rows[component]
            ax.bar(
                positions,
                values,
                bottom=bottom,
                label=component,
                color=palette_color(palette, index),
            )
            bottom = bottom + values
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.legend(title=color_col)
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=x_title,
            y_label=y_title,
        )
        return fig

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static stacked Matplotlib breakdown chart."""
        fig = self.build_static_figure(df)
        if fig is None:
            return
        self.save_figure(fig, out_path)
        plt.close(fig)
