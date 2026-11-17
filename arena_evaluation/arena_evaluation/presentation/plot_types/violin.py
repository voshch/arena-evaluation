"""Violin plot renderer for distribution density comparisons."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import seaborn as sns

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    label_values,
    new_figure,
    palette_color,
    series_values,
)


class ViolinRenderer(BasePlotRenderer):
    """Renders distribution violin plots using Plotly and Seaborn."""

    PLOT_TYPE = "violin"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS

    def _resolve_columns(
        self, df: pl.DataFrame
    ) -> tuple[str | None, str | None, str | None]:
        return (
            self.resolve_channel("x"),
            self.resolve_channel("y"),
            self.resolve_channel("color"),
        )

    def _title(self, column: str | None) -> str:
        return (column or "").replace("_", " ").title()

    def _label(self, column: str) -> str:
        return self.format_label(self._title(column), column)

    def _grouped(
        self, df: pl.DataFrame, trace_col: str | None, y_col: str
    ) -> list[tuple[str, pl.DataFrame]]:
        if trace_col is None:
            return [(y_col, df)]
        tagged = df.with_columns(
            pl.Series("_violin_group", label_values(df, trace_col))
        )
        return [
            (str(sub["_violin_group"][0]), sub)
            for sub in tagged.sort("_violin_group").partition_by(
                "_violin_group", maintain_order=True
            )
        ]

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly violin plot."""
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        if not y_col:
            return None

        trace_col = color_col or x_col
        palette = get_color_palette()
        fig = go.Figure()

        for idx, (name, sub) in enumerate(self._grouped(df, trace_col, y_col)):
            violin: dict[str, object] = {
                "y": series_values(sub, y_col).tolist(),
                "name": name,
                "box_visible": True,
                "line_color": palette_color(palette, idx),
            }
            if x_col and x_col != trace_col:
                violin["x"] = label_values(sub, x_col).tolist()
            fig.add_trace(go.Violin(**violin))

        y_label = self._label(y_col)
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or y_label,
            yaxis_title=y_label,
            xaxis_title=self._title(x_col or trace_col),
            violinmode="group"
            if color_col and x_col and color_col != x_col
            else "overlay",
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn violin plot."""
        if df.is_empty():
            return
        x_col, y_col, color_col = self._resolve_columns(df)
        if not y_col:
            return

        axis_col = x_col or color_col
        cols = [c for c in dict.fromkeys([axis_col, color_col, y_col]) if c]
        pdf = df.select(cols).to_pandas()
        for column in dict.fromkeys([axis_col, color_col]):
            if column:
                pdf[column] = pd.Series(label_values(df, column))

        palette = get_color_palette()
        fig, ax = new_figure()
        violin: dict[str, object] = {
            "data": pdf,
            "y": y_col,
            "inner": "box",
            "ax": ax,
        }
        if axis_col and color_col and color_col != axis_col:
            violin.update(x=axis_col, hue=color_col, palette=palette)
        elif axis_col:
            violin.update(x=axis_col, hue=axis_col, palette=palette, legend=False)
        else:
            violin["color"] = palette_color(palette, 0)
        sns.violinplot(**violin)

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
