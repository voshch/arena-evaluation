"""Line plot renderer for bound series and pre-aggregated curves."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
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


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(99, 110, 250, {alpha})"


class LineRenderer(BasePlotRenderer):
    """Renders line charts from pre-computed or series data without in-renderer aggregation."""

    PLOT_TYPE = "line"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "ci_low",
        "ci_high",
        "error_col",
        "err_plus",
        "err_minus",
        "clamp_zero",
        "show_markers",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {"ci_low", "ci_high", "error_col", "err_plus", "err_minus"}
    )

    def _resolve_columns(
        self, df: pl.DataFrame
    ) -> tuple[str | None, str | None, str | None]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if not x_col or not y_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: line plot requires channels 'x' and 'y'."
            )
        color_col = self.resolve_channel("color")
        cell_keys = [c for c in dict.fromkeys((x_col, color_col)) if c is not None]
        self.validate_unique_cells(df, cell_keys)
        return x_col, y_col, color_col

    def _error_bounds(
        self, frame: pl.DataFrame, y_vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        opts = self.spec.options
        clamp_zero = bool(opts.get("clamp_zero", False))
        low_col = opts.get("ci_low")
        high_col = opts.get("ci_high")
        lo: np.ndarray | None = None
        hi: np.ndarray | None = None
        if isinstance(low_col, str) and isinstance(high_col, str):
            lo = series_values(frame, low_col)
            hi = series_values(frame, high_col)
        else:
            plus_opt = opts.get("err_plus")
            minus_opt = opts.get("err_minus")
            if isinstance(plus_opt, str):
                plus = series_values(frame, plus_opt)
                minus = (
                    series_values(frame, minus_opt)
                    if isinstance(minus_opt, str)
                    else plus
                )
                lo, hi = y_vals - minus, y_vals + plus
            elif isinstance(opts.get("error_col"), str):
                err = series_values(frame, str(opts["error_col"]))
                lo, hi = y_vals - err, y_vals + err
        if lo is None or hi is None:
            return None
        if clamp_zero:
            lo = np.maximum(lo, 0.0)
        return lo, hi

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly line plot from bound columns."""
        # differs from overhaul/source:plot_types/line.py:84-116: no binning, quantile filtering, or group_by mean/std; NULLs stay NaN
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        if not x_col or not y_col:
            return None

        palette = get_color_palette()
        fig = go.Figure()
        show_markers = bool(self.spec.options.get("show_markers", True))
        trace_mode = "lines+markers" if show_markers else "lines"
        groups: list[tuple[str, pl.DataFrame]]
        if color_col:
            tagged = df.with_columns(
                pl.Series("_line_group", label_values(df, color_col))
            )
            groups = [
                (str(sub["_line_group"][0]), sub.sort(x_col))
                for sub in tagged.sort("_line_group").partition_by(
                    "_line_group", maintain_order=True
                )
            ]
        else:
            groups = [(y_col, df.sort(x_col))]

        for idx, (label, sub) in enumerate(groups):
            color = palette_color(palette, idx)
            x_vals = sub[x_col].to_numpy().tolist()
            y_vals = series_values(sub, y_col)
            bounds = self._error_bounds(sub, y_vals)
            if bounds is not None:
                lo, hi = bounds
                valid = np.isfinite(y_vals) & np.isfinite(lo) & np.isfinite(hi)
                if np.any(valid):
                    vx = [x_vals[i] for i, ok in enumerate(valid) if ok]
                    fig.add_trace(
                        go.Scatter(
                            x=vx + vx[::-1],
                            y=hi[valid].tolist() + lo[valid][::-1].tolist(),
                            fill="toself",
                            fillcolor=_rgba(color, 0.18),
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{label}_band",
                        )
                    )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals.tolist(),
                    mode=trace_mode,
                    name=label,
                    line={"color": color, "width": 2.2},
                    connectgaps=False,
                )
            )

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_title = self.format_label(y_col.replace("_", " ").title(), y_col)
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or f"{y_title} vs {x_title}",
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn line plot from bound columns."""
        if df.is_empty():
            return
        x_col, y_col, color_col = self._resolve_columns(df)
        if not x_col or not y_col:
            return

        sorted_df = df.sort(x_col)
        cols = [c for c in dict.fromkeys([x_col, y_col, color_col]) if c]
        pdf = sorted_df.select(cols).to_pandas()
        if color_col:
            pdf[color_col] = pd.Series(label_values(sorted_df, color_col))

        palette = get_color_palette()
        fig, ax = new_figure()
        marker = "o" if bool(self.spec.options.get("show_markers", True)) else None
        if color_col:
            sns.lineplot(
                data=pdf,
                x=x_col,
                y=y_col,
                hue=color_col,
                palette=palette,
                estimator=None,
                errorbar=None,
                marker=marker,
                ax=ax,
            )
            tagged = sorted_df.with_columns(
                pl.Series("_line_group", label_values(sorted_df, color_col))
            )
            for idx, sub in enumerate(
                tagged.sort("_line_group").partition_by(
                    "_line_group", maintain_order=True
                )
            ):
                y_vals = series_values(sub, y_col)
                bounds = self._error_bounds(sub, y_vals)
                if bounds is not None:
                    lo, hi = bounds
                    valid = np.isfinite(y_vals) & np.isfinite(lo) & np.isfinite(hi)
                    if np.any(valid):
                        xs = series_values(sub, x_col)
                        ax.fill_between(
                            xs[valid],
                            lo[valid],
                            hi[valid],
                            color=palette_color(palette, idx),
                            alpha=0.18,
                        )
        else:
            sns.lineplot(
                data=pdf,
                x=x_col,
                y=y_col,
                color=palette_color(palette, 0),
                estimator=None,
                errorbar=None,
                marker=marker,
                ax=ax,
            )
            y_vals = series_values(sorted_df, y_col)
            bounds = self._error_bounds(sorted_df, y_vals)
            if bounds is not None:
                lo, hi = bounds
                valid = np.isfinite(y_vals) & np.isfinite(lo) & np.isfinite(hi)
                if np.any(valid):
                    xs = series_values(sorted_df, x_col)
                    ax.fill_between(
                        xs[valid],
                        lo[valid],
                        hi[valid],
                        color=palette_color(palette, 0),
                        alpha=0.18,
                    )

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_title = self.format_label(y_col.replace("_", " ").title(), y_col)
        finish_axes(
            ax,
            title=self.spec.title or f"{y_title} vs {x_title}",
            x_label=x_title,
            y_label=y_title,
            grid_axis="both",
        )
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
