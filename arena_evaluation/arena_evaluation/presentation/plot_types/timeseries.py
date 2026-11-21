"""Timeseries renderer: one mean curve per colour value, with the 10th to 90th percentile over episodes as its band."""

from __future__ import annotations

import pathlib
from typing import ClassVar, Final, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.analytics.stats import aggregate_timeseries_band
from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    new_figure,
    palette_color,
    require_drawable,
)

DEFAULT_BIN_WIDTH_S: Final[float] = 0.5
BAND_LOW_QUANTILE: Final[float] = 0.1
BAND_HIGH_QUANTILE: Final[float] = 0.9
_ALL_CURVES: Final[str] = "all"
_COLOUR: Final[str] = "colour"
_BIN: Final[str] = "x"
_VALUE: Final[str] = "value"


def aggregate_curves(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    colour_col: str | None,
    bin_width_s: float,
) -> pl.DataFrame:
    """Mean and percentile band of ``y_col`` over the episodes, per colour value and time bin."""
    return aggregate_timeseries_band(
        df,
        x_col=x_col,
        y_col=y_col,
        colour_col=colour_col,
        bin_width_s=bin_width_s,
        low_quantile=BAND_LOW_QUANTILE,
        high_quantile=BAND_HIGH_QUANTILE,
    )


class _Curve(NamedTuple):
    name: str
    x: np.ndarray
    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    on_second_axis: bool


class TimeseriesRenderer(BasePlotRenderer):
    """Renders the mean over episodes of each signal per colour value, with its 10th to 90th percentile band.

    Signals declared as ``y2`` share the time axis and are drawn dotted against a second, right-hand axis.
    The option ``bin_width_s`` sets the time resolution of the mean and the band.
    """

    PLOT_TYPE = "timeseries"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "bin_width_s",
        "show_band",
    }

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, list[str], list[str], str | None]:
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if not x_col or not y_cols:
            raise ValueError(f"Plot {self.spec.id!r}: timeseries plot requires channels 'x' and 'y'.")
        y2_cols = self.resolve_channel_list("y2")
        for y2_col in y2_cols:
            require_drawable(df, y2_col, plot_id=self.spec.id, plot_type=self.PLOT_TYPE)
        return x_col, y_cols, y2_cols, self.resolve_channel("color")

    def _label(self, column: str) -> str:
        return self.format_label(column.replace("_", " ").title(), column)

    def _curves(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_cols: list[str],
        y2_cols: list[str],
        colour_col: str | None,
    ) -> list[_Curve]:
        bin_width_s = self.option_float("bin_width_s", DEFAULT_BIN_WIDTH_S)
        signals = [*y_cols, *y2_cols]
        curves: list[_Curve] = []
        for y_col in signals:
            aggregated = aggregate_curves(
                df,
                x_col=x_col,
                y_col=y_col,
                colour_col=colour_col,
                bin_width_s=bin_width_s,
            )
            for key, sub in aggregated.partition_by(_COLOUR, as_dict=True, maintain_order=True).items():
                colour_value = key[0] if isinstance(key, tuple) else key
                parts = [] if len(signals) == 1 else [y_col]
                if colour_col:
                    parts.append(str(colour_value))
                curves.append(
                    _Curve(
                        name=" · ".join(parts) or y_col,
                        x=sub[_BIN].to_numpy(),
                        mean=sub["mean"].to_numpy(),
                        low=sub["low"].to_numpy(),
                        high=sub["high"].to_numpy(),
                        on_second_axis=y_col in y2_cols,
                    )
                )
        return curves

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly timeseries plot."""
        if df.is_empty():
            return None
        x_col, y_cols, y2_cols, color_col = self._resolve_columns(df)
        palette = get_color_palette()
        fig = go.Figure()

        show_band = bool(self.spec.options.get("show_band", True))
        for idx, curve in enumerate(self._curves(df, x_col, y_cols, y2_cols, color_col)):
            colour = palette_color(palette, idx)
            yaxis = "y2" if curve.on_second_axis else "y"
            if show_band and not np.allclose(curve.low, curve.high, equal_nan=True):
                band_x = np.concatenate([curve.x, curve.x[::-1]])
                band_y = np.concatenate([curve.high, curve.low[::-1]])
                fig.add_trace(
                    go.Scatter(
                        x=band_x.tolist(),
                        y=band_y.tolist(),
                        mode="lines",
                        line={"color": colour, "width": 0},
                        fill="toself",
                        opacity=0.2,
                        hoverinfo="skip",
                        showlegend=False,
                        yaxis=yaxis,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=curve.x.tolist(),
                    y=curve.mean.tolist(),
                    mode="lines",
                    name=curve.name,
                    line={
                        "color": colour,
                        "width": 2.0,
                        "dash": "dot" if curve.on_second_axis else "solid",
                    },
                    yaxis=yaxis,
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
            margin={"t": 55, "b": 60, "l": 65, "r": 30},
        )
        if y2_cols:
            fig.update_layout(
                yaxis2={
                    "title": self._label(y2_cols[0]),
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False,
                    "automargin": True,
                }
            )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib timeseries plot with the same curves as the interactive one."""
        if df.is_empty():
            return
        x_col, y_cols, y2_cols, color_col = self._resolve_columns(df)
        curves = self._curves(df, x_col, y_cols, y2_cols, color_col)
        palette = get_color_palette()
        fig, ax = new_figure()
        second_ax = ax.twinx() if y2_cols else ax

        show_band = bool(self.spec.options.get("show_band", True))
        for idx, curve in enumerate(curves):
            target_ax = second_ax if curve.on_second_axis else ax
            colour = palette_color(palette, idx)
            target_ax.plot(
                curve.x,
                curve.mean,
                color=colour,
                linewidth=2.0,
                linestyle=":" if curve.on_second_axis else "-",
                label=curve.name,
            )
            if show_band:
                target_ax.fill_between(curve.x, curve.low, curve.high, color=colour, alpha=0.2, linewidth=0)
        if y2_cols:
            second_ax.set_ylabel(self._label(y2_cols[0]))
        if len(curves) > 1:
            handles, labels = ax.get_legend_handles_labels()
            if y2_cols:
                second_handles, second_labels = second_ax.get_legend_handles_labels()
                handles = [*handles, *second_handles]
                labels = [*labels, *second_labels]
            ax.legend(
                handles,
                labels,
                loc="upper right",
                frameon=True,
                framealpha=0.85,
                edgecolor="none",
            )

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
