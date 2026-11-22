"""Interval plot renderer for point estimates with pre-computed bound columns."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _IntervalData:
    """Resolved interval cells with their bounds, groups and axis titles."""

    labels: list[str]
    values: np.ndarray
    low: np.ndarray
    high: np.ndarray
    groups: list[str] | None
    color_col: str | None
    x_title: str
    y_title: str

    def groups_with_indices(self) -> list[tuple[str, list[int]]]:
        """One ``(group, row indices)`` pair per group in first-seen order."""
        if self.groups is None:
            return [("", list(range(len(self.labels))))]
        indices: dict[str, list[int]] = {group: [] for group in self.groups}
        for index, group in enumerate(self.groups):
            indices[group].append(index)
        return list(indices.items())


class IntervalRenderer(BasePlotRenderer):
    """Draw point estimates with the interval stored in the columns named by the ``low`` and ``high`` options."""

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
        raw_low = options.get(_LOW_OPTION)
        raw_high = options.get(_HIGH_OPTION)
        low_col = raw_low.strip() if isinstance(raw_low, str) and raw_low.strip() else ""
        high_col = raw_high.strip() if isinstance(raw_high, str) and raw_high.strip() else ""
        if not low_col or not high_col:
            raise ValueError(f"Plot {self.spec.id!r}: interval plot requires options {_LOW_OPTION!r} and {_HIGH_OPTION!r} naming the lower and upper bound columns, e.g. options: {{'low': 'ci_low', 'high': 'ci_high'}}.")
        return low_col, high_col

    def _plot_data(self, df: pl.DataFrame) -> _IntervalData:
        """Resolve channels and bound columns once, with the shared axis titles."""
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if x_col is None:
            raise ValueError(f"Plot {self.spec.id!r}: interval plot requires channel 'x'.")
        if y_col is None:
            raise ValueError(f"Plot {self.spec.id!r}: interval plot requires channel 'y'.")
        color_col = self.resolve_channel("color")
        low_col, high_col = self._bound_columns()
        self.validate_unique_cells(df, [x_col, color_col] if color_col else [x_col])
        return _IntervalData(
            labels=label_values(df, x_col).tolist(),
            values=series_values(df, y_col),
            low=series_values(df, low_col),
            high=series_values(df, high_col),
            groups=label_values(df, color_col).tolist() if color_col else None,
            color_col=color_col,
            x_title=x_col.replace("_", " ").title(),
            y_title=self.format_label(y_col.replace("_", " ").title(), y_col),
        )

    def build_plotly_figure(self, df: pl.DataFrame) -> go.Figure | None:
        """Build the interactive interval figure without serializing it."""
        if df.is_empty():
            return None
        data = self._plot_data(df)
        palette = get_color_palette()
        fig = go.Figure()
        categories = list(dict.fromkeys(data.labels))
        positions = {label: idx for idx, label in enumerate(categories)}
        groups = data.groups_with_indices()
        n_groups = len(groups)
        step = _DODGE_WIDTH / max(n_groups - 1, 1) if n_groups > 1 else 0.0
        for group_index, (group, indices) in enumerate(groups):
            values = data.values[indices].astype(float)
            offset = (group_index - (n_groups - 1) / 2.0) * step if n_groups > 1 else 0.0
            fig.add_trace(
                go.Scatter(
                    x=[positions[data.labels[index]] + offset for index in indices],
                    y=values.tolist(),
                    mode="markers",
                    name=group or None,
                    showlegend=data.groups is not None,
                    marker={"color": palette_color(palette, group_index), "size": 8},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": (data.high[indices] - values).tolist(),
                        "arrayminus": (values - data.low[indices]).tolist(),
                    },
                )
            )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title,
            xaxis_title=data.x_title,
            yaxis_title=data.y_title,
            xaxis={
                "tickmode": "array",
                "tickvals": list(range(len(categories))),
                "ticktext": categories,
            },
        )
        if data.color_col:
            fig.update_layout(legend_title_text=data.color_col.replace("_", " ").title())
        return fig

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly interval chart with error bars."""
        fig = self.build_plotly_figure(df)
        return None if fig is None else fig.to_html(include_plotlyjs=False, full_html=False)

    def build_static_figure(self, df: pl.DataFrame) -> plt.Figure | None:
        """Build the static interval figure without writing it."""
        require_drawable(
            df,
            self.resolve_channel("y") or "",
            plot_id=self.spec.id,
            plot_type=self.PLOT_TYPE,
        )
        data = self._plot_data(df)
        positions_by_label = {label: index for index, label in enumerate(dict.fromkeys(data.labels))}
        groups = data.groups_with_indices()
        dodge = _DODGE_WIDTH if len(groups) > 1 else 0.0
        palette = get_color_palette()
        fig, ax = new_figure()
        for group_index, (group, indices) in enumerate(groups):
            offset = (group_index - (len(groups) - 1) / 2.0) * dodge
            colour = palette_color(palette, group_index)
            positions = [positions_by_label[data.labels[index]] + offset for index in indices]
            values = data.values[indices].astype(float)
            lowers = data.low[indices].astype(float)
            uppers = data.high[indices].astype(float)
            ax.vlines(positions, lowers, uppers, color=colour, linewidth=1.5)
            caps_left = [position - _CAP_HALF_WIDTH for position in positions]
            caps_right = [position + _CAP_HALF_WIDTH for position in positions]
            for bound_values in (lowers, uppers):
                ax.hlines(
                    bound_values,
                    caps_left,
                    caps_right,
                    color=colour,
                    linewidth=1.5,
                )
            ax.plot(positions, values, "o", color=colour)
            if data.groups is not None:
                ax.plot([], [], "o", color=colour, label=group)
        ax.set_xticks(np.arange(len(positions_by_label)))
        ax.set_xticklabels(list(positions_by_label), rotation=20, ha="right")
        if data.groups is not None and len(groups) > 1 and data.color_col:
            ax.legend(title=data.color_col.replace("_", " ").title())
        finish_axes(
            ax,
            title=self.spec.title,
            x_label=data.x_title,
            y_label=data.y_title,
        )
        return fig

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib interval plot with error bars."""
        fig = self.build_static_figure(df)
        if fig is None:
            return
        self.save_figure(fig, out_path)
        plt.close(fig)
