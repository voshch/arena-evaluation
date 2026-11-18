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
    require_drawable,
    series_values,
)

_LOW_OPTION: str = "low"
_HIGH_OPTION: str = "high"
_CAP_HALF_WIDTH: float = 0.12
_DODGE_WIDTH: float = 0.36


class IntervalRenderer(BasePlotRenderer):
    """Draw point estimates with the interval stored in two bound columns.

    A declared ``color`` channel groups the estimates (e.g. one interval per planner and
    crowd condition); the bounds are named by the ``low`` and ``high`` options. A spec
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

    def _channel_columns(self) -> tuple[str, str, str | None]:
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
        return x_col, y_col, self.resolve_channel("color")

    def _cell_key(self, x_col: str, color_col: str | None) -> list[str]:
        return [x_col, color_col] if color_col else [x_col]

    def _plot_data(
        self, df: pl.DataFrame
    ) -> tuple[
        list[str], np.ndarray, np.ndarray, np.ndarray, list[str] | None, str | None
    ]:
        x_col, y_col, color_col = self._channel_columns()
        low_col, high_col = self._bound_columns()
        self.validate_unique_cells(df, self._cell_key(x_col, color_col))
        return (
            label_values(df, x_col).tolist(),
            series_values(df, y_col),
            series_values(df, low_col),
            series_values(df, high_col),
            label_values(df, color_col).tolist() if color_col else None,
            color_col,
        )

    def _group_order(self, groups: list[str] | None) -> list[str]:
        return list(dict.fromkeys(groups)) if groups else [""]

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the interactive interval figure without serializing it."""
        if df.is_empty():
            return None
        labels, values, low, high, groups, color_col = self._plot_data(df)
        x_col, y_col, _ = self._channel_columns()
        palette = get_color_palette()
        fig = go.Figure()
        for group_index, group in enumerate(self._group_order(groups)):
            keep = (
                [value == group for value in groups]
                if groups is not None
                else [True] * len(labels)
            )
            xs = [label for label, drawn in zip(labels, keep, strict=True) if drawn]
            if not xs:
                continue
            ys = [
                float(value)
                for value, drawn in zip(values.tolist(), keep, strict=True)
                if drawn
            ]
            ups = [
                float(upper) - value
                for upper, value, drawn in zip(
                    high.tolist(), values.tolist(), keep, strict=True
                )
                if drawn
            ]
            downs = [
                value - float(lower)
                for lower, value, drawn in zip(
                    low.tolist(), values.tolist(), keep, strict=True
                )
                if drawn
            ]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=group or None,
                    showlegend=groups is not None,
                    marker={"color": palette_color(palette, group_index)},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": ups,
                        "arrayminus": downs,
                    },
                )
            )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=self.format_label(y_col.replace("_", " ").title(), y_col),
        )
        if color_col:
            fig.update_layout(legend_title_text=color_col.replace("_", " ").title())
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly interval chart with error bars."""
        fig = self.build_plotly_figure(df)
        if fig is None:
            return None
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the static interval figure without writing it."""
        require_drawable(
            df,
            self.resolve_channel("y") or "",
            plot_id=self.spec.id,
            plot_type=self.PLOT_TYPE,
        )
        labels, values, low, high, groups, color_col = self._plot_data(df)
        x_col, y_col, _ = self._channel_columns()
        positions_by_label = {
            label: index for index, label in enumerate(dict.fromkeys(labels))
        }
        group_order = self._group_order(groups)
        dodge = _DODGE_WIDTH if groups is not None and len(group_order) > 1 else 0.0
        palette = get_color_palette()
        fig, ax = new_figure()
        for group_index, group in enumerate(group_order):
            offset = (group_index - (len(group_order) - 1) / 2.0) * dodge
            colour = palette_color(palette, group_index)
            drew_one = False
            for label, value, lower, upper, row_group in zip(
                labels,
                values.tolist(),
                low.tolist(),
                high.tolist(),
                groups if groups is not None else [""] * len(labels),
                strict=True,
            ):
                if row_group != group:
                    continue
                drew_one = True
                position = positions_by_label[label] + offset
                ax.vlines(position, lower, upper, color=colour, linewidth=1.5)
                for bound in (lower, upper):
                    ax.hlines(
                        bound,
                        position - _CAP_HALF_WIDTH,
                        position + _CAP_HALF_WIDTH,
                        color=colour,
                        linewidth=1.5,
                    )
                ax.plot([position], [value], "o", color=colour)
            if drew_one and groups is not None:
                ax.plot([], [], "o", color=colour, label=group)
        ax.set_xticks(np.arange(len(positions_by_label)))
        ax.set_xticklabels(list(positions_by_label), rotation=20, ha="right")
        if groups is not None and len(group_order) > 1 and color_col:
            ax.legend(title=color_col.replace("_", " ").title())
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
