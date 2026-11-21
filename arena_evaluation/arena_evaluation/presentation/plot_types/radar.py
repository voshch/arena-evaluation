"""Radar (spider) chart renderer for multi-metric planner profiles."""

from __future__ import annotations

import math
import pathlib
from typing import ClassVar, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    label_values,
    palette_color,
    series_values,
)


class _RadarData(NamedTuple):
    axis_labels: list[str]
    traces: list[tuple[str, list[float | None]]]


def _optional_float(value: object) -> float | None:
    if value is None or math.isnan(value):
        return None
    return float(value)


class RadarRenderer(BasePlotRenderer):
    """Renders polar radar charts from long radar_profile rows."""

    PLOT_TYPE = "radar"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS

    def _axis_values(
        self, frame: pl.DataFrame, x_col: str, y_col: str, axes: list[str]
    ) -> list[float | None]:
        by_axis = dict(
            zip(
                label_values(frame, x_col).tolist(),
                series_values(frame, y_col).tolist(),
                strict=True,
            )
        )
        return [_optional_float(by_axis.get(axis)) for axis in axes]

    def _prepare(self, df: pl.DataFrame) -> _RadarData | None:
        """Resolve the radar series once, or ``None`` for an empty frame."""
        if df.is_empty():
            return None
        group_col = self.resolve_channel("color")
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if x_col is None or len(y_cols) != 1:
            raise ValueError(
                f"Plot {self.spec.id!r}: radar plot requires channel 'x' and a single 'y' metric value column."
            )
        self.validate_unique_cells(
            df, [c for c in dict.fromkeys((group_col, x_col)) if c is not None]
        )
        y_col = y_cols[0]
        axes = sorted(dict.fromkeys(label_values(df, x_col).tolist()))
        if len(axes) < 3:
            raise ValueError(
                f"Plot {self.spec.id!r}: radar plot requires at least 3 axes, got {len(axes)}: {axes!r}."
            )
        axis_labels = [axis.replace("_", " ").title() for axis in axes]
        if group_col is None:
            traces = [(y_col, self._axis_values(df, x_col, y_col, axes))]
        else:
            tagged = df.with_columns(
                pl.Series("_radar_group", label_values(df, group_col))
            )
            traces = [
                (
                    str(sub["_radar_group"][0]),
                    self._axis_values(sub, x_col, y_col, axes),
                )
                for sub in tagged.sort("_radar_group").partition_by(
                    "_radar_group", maintain_order=True
                )
            ]
        return _RadarData(axis_labels, traces)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly radar chart."""
        data = self._prepare(df)
        if data is None:
            return None
        palette = get_color_palette()
        fig = go.Figure(
            data=[
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=data.axis_labels + [data.axis_labels[0]],
                    fill="toself",
                    name=label,
                    line={"color": palette_color(palette, index)},
                    connectgaps=False,
                )
                for index, (label, values) in enumerate(data.traces)
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or "Multi-Metric Radar Profile",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(t=60, b=40, l=60, r=60),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib radar chart."""
        data = self._prepare(df)
        if data is None:
            return
        angles = np.linspace(
            0, 2 * np.pi, len(data.axis_labels), endpoint=False
        ).tolist()
        angles_closed = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
        palette = get_color_palette()
        for index, (label, values) in enumerate(data.traces):
            arr = np.array(
                [np.nan if value is None else value for value in values], dtype=float
            )
            arr_closed = np.append(arr, arr[0])
            color = palette_color(palette, index)
            ax.plot(angles_closed, arr_closed, linewidth=2, label=label, color=color)
            if np.all(np.isfinite(arr)):
                ax.fill(angles_closed, arr_closed, color=color, alpha=0.12)
        ax.set_xticks(angles)
        ax.set_xticklabels(data.axis_labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            self.spec.title or "Multi-Metric Radar Profile",
            pad=20,
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=False)
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
