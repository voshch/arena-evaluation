"""Timeseries plot renderer for temporal telemetry signals."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
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

EPISODE_COLUMN = "episode_id"
Curve = tuple[str, pl.DataFrame, str]


class TimeseriesRenderer(BasePlotRenderer):
    """Renders time-domain curves, one per signal and per colour value and episode, in Plotly and Matplotlib alike."""

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

    def _curves(
        self, sorted_df: pl.DataFrame, y_cols: list[str], color_col: str | None
    ) -> list[Curve]:
        """One curve per signal and per (colour, episode) group: repeats of a robot never join into one line."""
        split_cols = [
            col for col in (color_col, EPISODE_COLUMN) if col and col in sorted_df.columns
        ]
        if not split_cols:
            return [(y_col, sorted_df, y_col) for y_col in y_cols]
        groups = sorted_df.partition_by(split_cols, as_dict=True, maintain_order=True)
        return [
            (self._curve_name(y_col, y_cols, split_cols, key), sub, y_col)
            for y_col in y_cols
            for key, sub in groups.items()
        ]

    @staticmethod
    def _curve_name(
        y_col: str, y_cols: list[str], split_cols: list[str], key: tuple
    ) -> str:
        parts = [] if len(y_cols) == 1 else [y_col]
        for column, value in zip(split_cols, key, strict=True):
            if column == EPISODE_COLUMN:
                parts.append(f"episode {value}")
            else:
                parts.append("NULL" if value is None else str(value))
        return " · ".join(parts)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly timeseries plot."""
        if df.is_empty():
            return None
        x_col, y_cols, color_col = self._resolve_columns(df)
        palette = get_color_palette()
        fig = go.Figure()

        for idx, (name, sub, y_col) in enumerate(
            self._curves(df.sort(x_col), y_cols, color_col)
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
        """Render a static Matplotlib timeseries plot with the same curves as the interactive one."""
        if df.is_empty():
            return
        x_col, y_cols, color_col = self._resolve_columns(df)
        curves = self._curves(df.sort(x_col), y_cols, color_col)
        palette = get_color_palette()
        fig, ax = new_figure()

        for idx, (name, sub, y_col) in enumerate(curves):
            ax.plot(
                sub[x_col].to_numpy(),
                series_values(sub, y_col),
                color=palette_color(palette, idx),
                linewidth=2.0,
                label=name,
            )
        if len(curves) > 1:
            ax.legend(frameon=False)

        y_title = "Value" if len(y_cols) > 1 else self._label(y_cols[0])
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
