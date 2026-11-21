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
from plotly.subplots import make_subplots

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
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"ci_low", "ci_high", "error_col", "err_plus", "err_minus"})

    def _resolve_columns(self, df: pl.DataFrame) -> tuple[str, str, str | None, str | None, str | None]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if not x_col or not y_col:
            raise ValueError(f"Plot {self.spec.id!r}: line plot requires channels 'x' and 'y'.")
        color_col = self.resolve_channel("color")
        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")
        cell_keys = [c for c in dict.fromkeys((x_col, color_col, facet_col, facet_row)) if c is not None]
        self.validate_unique_cells(df, cell_keys)
        return x_col, y_col, color_col, facet_col, facet_row

    def _titles(self, x_col: str, y_col: str) -> tuple[str, str]:
        return (
            self.format_label(x_col.replace("_", " ").title(), x_col),
            self.format_label(y_col.replace("_", " ").title(), y_col),
        )

    def _grouped(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None,
    ) -> list[tuple[str, pl.DataFrame]]:
        """One x-sorted frame per colour level, or one frame when uncoloured."""
        if color_col is None:
            return [(y_col, df.sort(x_col))]
        tagged = df.with_columns(pl.Series("_line_group", label_values(df, color_col)))
        return [(str(sub["_line_group"][0]), sub.sort(x_col)) for sub in tagged.sort("_line_group").partition_by("_line_group", maintain_order=True)]

    def _band(self, frame: pl.DataFrame, y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Mask and bounds of the finite error band, or ``None`` when nothing is drawable."""
        opts = self.spec.options
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
                minus = series_values(frame, minus_opt) if isinstance(minus_opt, str) else plus
                lo, hi = y_vals - minus, y_vals + plus
            elif isinstance(opts.get("error_col"), str):
                err = series_values(frame, str(opts["error_col"]))
                lo, hi = y_vals - err, y_vals + err
        if lo is None or hi is None:
            return None
        if bool(opts.get("clamp_zero", False)):
            lo = np.maximum(lo, 0.0)
        valid = np.isfinite(y_vals) & np.isfinite(lo) & np.isfinite(hi)
        if not np.any(valid):
            return None
        return valid, lo, hi

    @staticmethod
    def _facet_grid(df: pl.DataFrame, facet_col: str | None, facet_row: str | None) -> tuple[list[str | None], list[str | None]]:
        cols: list[str | None] = [str(v) for v in sorted({str(v) for v in label_values(df, facet_col).tolist()})] if facet_col else [None]
        rows: list[str | None] = [str(v) for v in sorted({str(v) for v in label_values(df, facet_row).tolist()})] if facet_row else [None]
        return rows, cols

    @staticmethod
    def _facet_slice(
        df: pl.DataFrame,
        facet_col: str | None,
        col_val: str | None,
        facet_row: str | None,
        row_val: str | None,
    ) -> pl.DataFrame:
        out = df
        if facet_col and col_val is not None:
            out = out.filter(pl.Series(label_values(out, facet_col)) == col_val)
        if facet_row and row_val is not None:
            out = out.filter(pl.Series(label_values(out, facet_row)) == row_val)
        return out

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly line plot from bound columns."""
        if df.is_empty():
            return None
        x_col, y_col, color_col, facet_col, facet_row = self._resolve_columns(df)
        palette = get_color_palette()
        row_vals, col_vals = self._facet_grid(df, facet_col, facet_row)
        nrows, ncols = len(row_vals), len(col_vals)
        is_faceted = facet_col is not None or facet_row is not None
        if is_faceted:
            titles = []
            for rv in row_vals:
                for cv in col_vals:
                    parts = [p for p in (cv, f"{facet_row}={rv}" if rv is not None else None) if p]
                    titles.append(" · ".join(parts))
            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                shared_xaxes=True,
                shared_yaxes=True,
                subplot_titles=titles,
                horizontal_spacing=0.05,
                vertical_spacing=0.09,
            )
        else:
            fig = go.Figure()
        show_markers = bool(self.spec.options.get("show_markers", True))
        trace_mode = "lines+markers" if show_markers else "lines"
        color_order = sorted({str(v) for v in label_values(df, color_col).tolist()}) if color_col else [y_col]
        seen_legend: set[str] = set()
        for r_idx, rv in enumerate(row_vals, start=1):
            for c_idx, cv in enumerate(col_vals, start=1):
                cell_df = self._facet_slice(df, facet_col, cv, facet_row, rv)
                if cell_df.is_empty():
                    continue
                for label, sub in self._grouped(cell_df, x_col, y_col, color_col):
                    color = palette_color(palette, color_order.index(label) if label in color_order else 0)
                    x_vals = sub[x_col].to_numpy().tolist()
                    y_vals = series_values(sub, y_col)
                    band = self._band(sub, y_vals)
                    if band is not None:
                        valid, lower, upper = band
                        vx = [x for x, keep in zip(x_vals, valid, strict=True) if keep]
                        band_trace = go.Scatter(
                            x=vx + vx[::-1],
                            y=upper[valid].tolist() + lower[valid][::-1].tolist(),
                            fill="toself",
                            fillcolor=_rgba(color, 0.18),
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{label}_band",
                        )
                        if is_faceted:
                            fig.add_trace(band_trace, row=r_idx, col=c_idx)
                        else:
                            fig.add_trace(band_trace)
                    show_leg = label not in seen_legend
                    seen_legend.add(label)
                    line_trace = go.Scatter(
                        x=x_vals,
                        y=y_vals.tolist(),
                        mode=trace_mode,
                        name=label,
                        legendgroup=label,
                        showlegend=show_leg,
                        line={"color": color, "width": 2.0},
                        connectgaps=False,
                    )
                    if is_faceted:
                        fig.add_trace(line_trace, row=r_idx, col=c_idx)
                    else:
                        fig.add_trace(line_trace)
        x_title, y_title = self._titles(x_col, y_col)
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or f"{y_title} vs {x_title}",
            height=max(420, 260 * nrows) if is_faceted else None,
            margin={"t": 65, "b": 60, "l": 65, "r": 30},
        )
        if is_faceted:
            fig.update_xaxes(title_text=x_title, row=nrows)
            fig.update_yaxes(title_text=y_title, col=1)
        else:
            fig.update_xaxes(title_text=x_title)
            fig.update_yaxes(title_text=y_title)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn/Matplotlib line plot from bound columns."""
        if df.is_empty():
            return
        x_col, y_col, color_col, facet_col, facet_row = self._resolve_columns(df)
        x_title, y_title = self._titles(x_col, y_col)
        palette = get_color_palette()
        marker = "o" if bool(self.spec.options.get("show_markers", True)) else None
        if facet_col is not None or facet_row is not None:
            row_vals, col_vals = self._facet_grid(df, facet_col, facet_row)
            nrows, ncols = len(row_vals), len(col_vals)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(max(9.0, 3.8 * ncols), max(4.5, 2.7 * nrows)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            color_order = sorted({str(v) for v in label_values(df, color_col).tolist()}) if color_col else [y_col]
            for r_idx, rv in enumerate(row_vals):
                for c_idx, cv in enumerate(col_vals):
                    ax = axes[r_idx][c_idx]
                    cell_df = self._facet_slice(df, facet_col, cv, facet_row, rv)
                    if not cell_df.is_empty():
                        for label, sub in self._grouped(cell_df, x_col, y_col, color_col):
                            colour = palette_color(
                                palette,
                                color_order.index(label) if label in color_order else 0,
                            )
                            xs = series_values(sub, x_col)
                            ys = series_values(sub, y_col)
                            ax.plot(
                                xs,
                                ys,
                                color=colour,
                                linewidth=1.8,
                                marker=marker,
                                label=label,
                            )
                            band = self._band(sub, ys)
                            if band is not None:
                                valid, lower, upper = band
                                ax.fill_between(
                                    xs[valid],
                                    lower[valid],
                                    upper[valid],
                                    color=colour,
                                    alpha=0.18,
                                )
                    cell_parts = [p for p in (cv, f"{facet_row}={rv}" if rv is not None else None) if p]
                    finish_axes(
                        ax,
                        title=" · ".join(cell_parts),
                        x_label=x_title if r_idx == nrows - 1 else "",
                        y_label=y_title if c_idx == 0 else "",
                        grid_axis="both",
                    )
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.96),
                    ncol=min(len(labels), 4),
                    frameon=False,
                )
            fig.suptitle(
                self.spec.title or f"{y_title} vs {x_title}",
                fontsize=12,
                fontweight="bold",
                y=0.995,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
            self.save_figure(fig, out_path)
            plt.close(fig)
            return

        sorted_df = df.sort(x_col)
        cols = [c for c in dict.fromkeys([x_col, y_col, color_col]) if c]
        pdf = sorted_df.select(cols).to_pandas()
        if color_col:
            pdf[color_col] = pd.Series(label_values(sorted_df, color_col))
        fig, ax = new_figure()
        common = {
            "data": pdf,
            "x": x_col,
            "y": y_col,
            "estimator": None,
            "errorbar": None,
            "marker": marker,
            "ax": ax,
        }
        if color_col:
            sns.lineplot(**common, hue=color_col, palette=palette)
        else:
            sns.lineplot(**common, color=palette_color(palette, 0))
        for index, (_, sub) in enumerate(self._grouped(sorted_df, x_col, y_col, color_col)):
            y_vals = series_values(sub, y_col)
            band = self._band(sub, y_vals)
            if band is None:
                continue
            valid, lower, upper = band
            xs = series_values(sub, x_col)
            ax.fill_between(
                xs[valid],
                lower[valid],
                upper[valid],
                color=palette_color(palette, index),
                alpha=0.18,
            )
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
