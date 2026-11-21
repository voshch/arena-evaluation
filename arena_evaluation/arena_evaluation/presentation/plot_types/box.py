"""Box plot renderer for distribution comparisons."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns

from arena_evaluation.presentation.checks import PlotRenderError
from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    label_values,
    new_figure,
    palette_color,
    require_drawable,
    series_values,
)


class BoxRenderer(BasePlotRenderer):
    """Renders distribution box plots using Plotly and Seaborn."""

    PLOT_TYPE = "box"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "show_outliers",
        "show_points",
    }

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str | None, str | None, str | None]:
        return (
            self.resolve_channel("x"),
            self.resolve_channel("y"),
            self.resolve_channel("color"),
        )

    def _title(self, column: str | None) -> str:
        return (column or "").replace("_", " ").title()

    def _label(self, column: str) -> str:
        return self.format_label(self._title(column), column)

    def _grouped(self, df: pl.DataFrame, trace_col: str | None, y_col: str) -> list[tuple[str, pl.DataFrame]]:
        if trace_col is None:
            return [(y_col, df)]
        tagged = df.with_columns(pl.Series("_box_group", label_values(df, trace_col)))
        return [(str(sub["_box_group"][0]), sub) for sub in tagged.sort("_box_group").partition_by("_box_group", maintain_order=True)]

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly box plot."""
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        if not y_col:
            return None

        trace_col = color_col or x_col
        palette = get_color_palette()
        show_points = "all" if self.spec.options.get("show_points") else "outliers"
        fig = go.Figure()

        for idx, (name, sub) in enumerate(self._grouped(df, trace_col, y_col)):
            box: dict[str, object] = {
                "y": series_values(sub, y_col).tolist(),
                "name": name,
                "boxpoints": show_points,
                "marker_color": palette_color(palette, idx),
            }
            if x_col and x_col != trace_col:
                box["x"] = label_values(sub, x_col).tolist()
            fig.add_trace(go.Box(**box))

        y_label = self._label(y_col)
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or y_label,
            yaxis_title=y_label,
            xaxis_title=self._title(x_col or trace_col),
            boxmode="group" if color_col and x_col and color_col != x_col else "overlay",
            margin={"t": 55, "b": 60, "l": 65, "r": 30},
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn box plot."""
        x_col, y_col, color_col = self._resolve_columns(df)
        if not y_col:
            raise PlotRenderError(f"Plot {self.spec.id!r}: box plot requires channel 'y'.")
        require_drawable(df, y_col, plot_id=self.spec.id, plot_type=self.PLOT_TYPE)

        axis_col = x_col or color_col
        cols = [c for c in dict.fromkeys([axis_col, color_col, y_col]) if c]
        pdf = df.select(cols).to_pandas()
        for column in dict.fromkeys([axis_col, color_col]):
            if column:
                pdf[column] = pd.Series(label_values(df, column))

        palette = get_color_palette()
        fig, ax = new_figure()
        show_outliers = bool(self.spec.options.get("show_outliers", True))
        if axis_col and color_col and color_col != axis_col:
            sns.boxplot(data=pdf, y=y_col, ax=ax, showfliers=show_outliers, x=axis_col, hue=color_col, palette=palette)
        elif axis_col:
            sns.boxplot(data=pdf, y=y_col, ax=ax, showfliers=show_outliers, x=axis_col, hue=axis_col, palette=palette, legend=False)
        else:
            sns.boxplot(data=pdf, y=y_col, ax=ax, showfliers=show_outliers, color=palette_color(palette, 0))

        if axis_col and pdf[axis_col].nunique() > 6:
            ax.tick_params(axis="x", rotation=30)
        y_label = self._label(y_col)
        finish_axes(
            ax,
            title=self.spec.title or y_label,
            x_label=self._title(axis_col),
            y_label=y_label,
        )
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
