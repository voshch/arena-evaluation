"""Timeseries plot renderer for temporal telemetry signals."""

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


class TimeseriesRenderer(BasePlotRenderer):
    """Renders multi-series time-domain curves using Plotly and Seaborn."""

    PLOT_TYPE = "timeseries"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS

    def _resolve_columns(
        self, df: pl.DataFrame
    ) -> tuple[str, list[str], str | None]:
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if not x_col or not y_cols:
            raise ValueError(
                f"Plot {self.spec.id!r}: timeseries plot requires channels 'x' and 'y'."
            )
        return x_col, y_cols, self.resolve_channel("color")

    def _label(self, column: str) -> str:
        return self.format_label(column.replace("_", " ").title(), column)

    def _series_groups(
        self, sorted_df: pl.DataFrame, y_cols: list[str], color_col: str | None
    ) -> list[tuple[str, pl.DataFrame, str]]:
        if len(y_cols) > 1 or color_col is None:
            return [(y_col, sorted_df, y_col) for y_col in y_cols]
        labels = sorted(
            label_values(sorted_df.select(color_col).unique(), color_col).tolist()
        )
        y_col = y_cols[0]
        return [
            (
                label,
                sorted_df.filter(pl.col(color_col).is_null())
                if label == "NULL"
                else sorted_df.filter(pl.col(color_col).cast(pl.Utf8) == label),
                y_col,
            )
            for label in labels
        ]

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly timeseries plot."""
        if df.is_empty():
            return None
        x_col, y_cols, color_col = self._resolve_columns(df)
        palette = get_color_palette()
        fig = go.Figure()

        for idx, (name, sub, y_col) in enumerate(
            self._series_groups(df.sort(x_col), y_cols, color_col)
        ):
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col].to_numpy().tolist(),
                    y=series_values(sub, y_col).tolist(),
                    mode="lines",
                    name=name,
                    line={"color": palette_color(palette, idx), "width": 2.0},
                    connectgaps=False,
                )
            )

        x_title = self._label(x_col)
        y_title = self._label(y_cols[0])
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or f"{y_title} over Time",
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn timeseries plot."""
        if df.is_empty():
            return
        x_col, y_cols, color_col = self._resolve_columns(df)
        sorted_df = df.sort(x_col)
        palette = get_color_palette()
        fig, ax = new_figure()

        if len(y_cols) > 1:
            melted = sorted_df.select([x_col, *y_cols]).unpivot(
                index=x_col, on=y_cols, variable_name="signal", value_name="value"
            )
            sns.lineplot(
                data=melted.to_pandas(),
                x=x_col,
                y="value",
                hue="signal",
                palette=palette,
                estimator=None,
                errorbar=None,
                ax=ax,
            )
            y_title = "Value"
        else:
            y_col = y_cols[0]
            cols = [c for c in dict.fromkeys([x_col, y_col, color_col]) if c]
            pdf = sorted_df.select(cols).to_pandas()
            line_kwargs: dict[str, object] = {
                "data": pdf,
                "x": x_col,
                "y": y_col,
                "estimator": None,
                "errorbar": None,
                "ax": ax,
            }
            if color_col:
                pdf[color_col] = pd.Series(label_values(sorted_df, color_col))
                line_kwargs.update(hue=color_col, palette=palette)
            else:
                line_kwargs["color"] = palette_color(palette, 0)
            sns.lineplot(**line_kwargs)
            y_title = self._label(y_col)

        x_title = self._label(x_col)
        finish_axes(
            ax,
            title=self.spec.title or f"{y_title} over Time",
            x_label=x_title,
            y_label=y_title,
            grid_axis="both",
        )
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
