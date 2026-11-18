"""Radar (spider) chart renderer for multi-metric planner profiles."""

from __future__ import annotations

import math
import pathlib
from typing import ClassVar

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


class RadarRenderer(BasePlotRenderer):
    """Renders polar radar charts from long radar_profile rows."""

    PLOT_TYPE = "radar"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str | None, str, str]:
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
        return group_col, x_col, y_cols[0]

    def _extract_series(
        self, df: pl.DataFrame
    ) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
        group_col, x_col, y_col = self._resolve_columns(df)
        axes = sorted(dict.fromkeys(label_values(df, x_col).tolist()))
        if len(axes) < 3:
            raise ValueError(
                f"Plot {self.spec.id!r}: radar plot requires at least 3 axes, got {len(axes)}: {axes!r}."
            )
        if not group_col:
            val_map = dict(
                zip(
                    label_values(df, x_col).tolist(),
                    series_values(df, y_col).tolist(),
                    strict=True,
                )
            )
            vals = [
                None if (v := val_map.get(ax)) is None or math.isnan(v) else float(v)
                for ax in axes
            ]
            return axes, [(y_col, vals)]

        tagged = df.with_columns(pl.Series("_radar_group", label_values(df, group_col)))
        traces: list[tuple[str, list[float | None]]] = []
        for sub in tagged.sort("_radar_group").partition_by(
            "_radar_group", maintain_order=True
        ):
            g = str(sub["_radar_group"][0])
            val_map = dict(
                zip(
                    label_values(sub, x_col).tolist(),
                    series_values(sub, y_col).tolist(),
                    strict=True,
                )
            )
            vals = [
                None if (v := val_map.get(ax)) is None or math.isnan(v) else float(v)
                for ax in axes
            ]
            traces.append((g, vals))
        return axes, traces

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly radar chart."""
        if df.is_empty():
            return None
        axes, traces = self._extract_series(df)
        if len(axes) < 3:
            return None

        theta = [m.replace("_", " ").title() for m in axes]
        theta_closed = theta + [theta[0]]
        palette = get_color_palette()
        fig = go.Figure()

        for idx, (label, vals) in enumerate(traces):
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=theta_closed,
                    fill="toself",
                    name=label,
                    line={"color": palette_color(palette, idx)},
                    connectgaps=False,
                )
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
        if df.is_empty():
            return
        axes, traces = self._extract_series(df)
        if len(axes) < 3:
            return

        num_vars = len(axes)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        theta_labels = [m.replace("_", " ").title() for m in axes]

        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True})
        palette = get_color_palette()

        for idx, (label, vals) in enumerate(traces):
            arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
            arr_closed = np.append(arr, arr[0])
            c = palette_color(palette, idx)
            ax.plot(angles_closed, arr_closed, linewidth=2, label=label, color=c)
            if np.all(np.isfinite(arr)):
                ax.fill(angles_closed, arr_closed, color=c, alpha=0.12)

        ax.set_xticks(angles)
        ax.set_xticklabels(theta_labels, fontsize=9)
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
