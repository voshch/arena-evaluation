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
    ) -> tuple[str | None, list[str], str | None]:
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if not x_col or not y_cols:
            raise ValueError(
                f"Plot {self.spec.id!r}: timeseries plot requires channels 'x' and 'y'."
            )
        color_col = self.resolve_channel("color")
        return x_col, y_cols, color_col

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly timeseries plot."""
        if df.is_empty():
            return None
        x_col, y_cols, color_col = self._resolve_columns(df)
        if not x_col or not y_cols:
            return None

        palette = get_color_palette()
        fig = go.Figure()
        sorted_df = df.sort(x_col)

        if len(y_cols) > 1 or color_col is None:
            for idx, y_col in enumerate(y_cols):
                fig.add_trace(
                    go.Scatter(
                        x=sorted_df[x_col].to_numpy().tolist(),
                        y=series_values(sorted_df, y_col).tolist(),
                        mode="lines",
                        name=y_col,
                        line={"color": palette_color(palette, idx), "width": 2.0},
                        connectgaps=False,
                    )
                )
        else:
            y_col = y_cols[0]
            for idx, label in enumerate(
                sorted(
                    label_values(
                        sorted_df.select(color_col).unique(), color_col
                    ).tolist()
                )
            ):
                sub = (
                    sorted_df.filter(pl.col(color_col).is_null())
                    if label == "NULL"
                    else sorted_df.filter(pl.col(color_col).cast(pl.Utf8) == label)
                )
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_col].to_numpy().tolist(),
                        y=series_values(sub, y_col).tolist(),
                        mode="lines",
                        name=label,
                        line={"color": palette_color(palette, idx), "width": 2.0},
                        connectgaps=False,
                    )
                )

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_title = self.format_label(y_cols[0].replace("_", " ").title(), y_cols[0])
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
        if not x_col or not y_cols:
            return

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
            if color_col:
                pdf[color_col] = pd.Series(label_values(sorted_df, color_col))
                sns.lineplot(
                    data=pdf,
                    x=x_col,
                    y=y_col,
                    hue=color_col,
                    palette=palette,
                    estimator=None,
                    errorbar=None,
                    ax=ax,
                )
            else:
                sns.lineplot(
                    data=pdf,
                    x=x_col,
                    y=y_col,
                    color=palette_color(palette, 0),
                    estimator=None,
                    errorbar=None,
                    ax=ax,
                )
            y_title = self.format_label(y_col.replace("_", " ").title(), y_col)

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
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
