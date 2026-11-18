"""Bar plot renderer for pre-aggregated values and optional error bounds."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
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


def _bound_error_arrays(
    frame: pl.DataFrame,
    *,
    err_plus_col: str | None,
    err_minus_col: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read pre-computed upper/lower error arrays without filling NULLs with 0.0."""
    plus = series_values(frame, err_plus_col) if err_plus_col is not None else None
    minus = series_values(frame, err_minus_col) if err_minus_col is not None else plus
    return plus, minus


def _plotly_error_kw(
    plus: np.ndarray | None, minus: np.ndarray | None
) -> dict[str, object] | None:
    if plus is None or minus is None:
        return None
    return {
        "type": "data",
        "array": plus.tolist(),
        "arrayminus": minus.tolist(),
        "symmetric": False,
    }


class BarRenderer(BasePlotRenderer):
    """Renders pre-aggregated bar charts with optional error bars and facets."""

    PLOT_TYPE = "bar"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "barmode",
        "err_plus",
        "err_minus",
        "error_col",
        "orientation",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {"err_plus", "err_minus", "error_col"}
    )

    def _resolve_columns(
        self, df: pl.DataFrame
    ) -> tuple[str | None, list[str], str | None, str | None, str | None]:
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if not y_cols:
            raise ValueError(f"Plot {self.spec.id!r}: bar plot requires channel 'y'.")
        color_col = self.resolve_channel("color")
        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")
        group_col = x_col or color_col
        if group_col is None:
            raise ValueError(
                f"Plot {self.spec.id!r}: bar plot requires channel 'x' or 'color'."
            )
        cell_keys = [
            c
            for c in dict.fromkeys((group_col, color_col, facet_col, facet_row))
            if c is not None
        ]
        self.validate_unique_cells(df, cell_keys)
        return x_col, y_cols, color_col, facet_col, facet_row

    def _error_columns(self) -> tuple[str | None, str | None]:
        opts = self.spec.options
        plus = opts.get("err_plus") or opts.get("error_col")
        if not isinstance(plus, str) or not plus.strip():
            return None, None
        minus = opts.get("err_minus")
        m_col = str(minus).strip() if isinstance(minus, str) and minus.strip() else plus
        return plus.strip(), m_col

    def _facet_partitions(
        self, df: pl.DataFrame, facet_col: str | None, facet_row: str | None
    ) -> tuple[list[str], list[str], list[tuple[int, int, str, pl.DataFrame]]]:
        col_vals = (
            list(dict.fromkeys(label_values(df, facet_col).tolist()))
            if facet_col
            else [""]
        )
        row_vals = (
            list(dict.fromkeys(label_values(df, facet_row).tolist()))
            if facet_row
            else [""]
        )
        panels: list[tuple[int, int, str, pl.DataFrame]] = []
        tagged = df
        if facet_col:
            tagged = tagged.with_columns(
                pl.Series("_bar_fcol", label_values(df, facet_col))
            )
        if facet_row:
            tagged = tagged.with_columns(
                pl.Series("_bar_frow", label_values(df, facet_row))
            )
        for r_idx, rv in enumerate(row_vals, start=1):
            for c_idx, cv in enumerate(col_vals, start=1):
                sub = tagged
                parts: list[str] = []
                if facet_col:
                    sub = sub.filter(pl.col("_bar_fcol") == cv)
                    parts.append(f"{facet_col}={cv}")
                if facet_row:
                    sub = sub.filter(pl.col("_bar_frow") == rv)
                    parts.append(f"{facet_row}={rv}")
                panels.append((r_idx, c_idx, ", ".join(parts), sub))
        return row_vals, col_vals, panels

    def _add_plotly_panel(
        self,
        fig: go.Figure,
        sub_df: pl.DataFrame,
        *,
        x_col: str | None,
        y_cols: list[str],
        color_col: str | None,
        group_col: str,
        orientation: str,
        err_plus_col: str | None,
        err_minus_col: str | None,
        palette: list[str],
        row: int | None,
        col: int | None,
        show_legend: bool,
    ) -> None:
        if sub_df.is_empty():
            return
        pos_kw = {"row": row, "col": col} if row is not None and col is not None else {}
        if len(y_cols) > 1:
            x_labels = label_values(sub_df, group_col).tolist()
            for idx, y_col in enumerate(y_cols):
                vals = series_values(sub_df, y_col).tolist()
                fig.add_trace(
                    go.Bar(
                        name=y_col,
                        x=x_labels if orientation != "h" else vals,
                        y=vals if orientation != "h" else x_labels,
                        orientation=orientation,
                        marker_color=palette_color(palette, idx),
                        showlegend=show_legend,
                    ),
                    **pos_kw,
                )
            return
        y_col = y_cols[0]
        if color_col is not None and x_col is not None and color_col != x_col:
            color_labels = label_values(sub_df, color_col).tolist()
            tagged = sub_df.with_columns(pl.Series("_bar_color_label", color_labels))
            for idx, sub in enumerate(
                tagged.partition_by("_bar_color_label", maintain_order=True)
            ):
                group_val = str(sub["_bar_color_label"][0])
                x_labels = label_values(sub, x_col).tolist()
                vals = series_values(sub, y_col).tolist()
                plus, minus = _bound_error_arrays(
                    sub, err_plus_col=err_plus_col, err_minus_col=err_minus_col
                )
                error_kw = _plotly_error_kw(plus, minus)
                fig.add_trace(
                    go.Bar(
                        name=group_val,
                        x=x_labels if orientation != "h" else vals,
                        y=vals if orientation != "h" else x_labels,
                        orientation=orientation,
                        marker_color=palette_color(palette, idx),
                        error_y=error_kw if orientation != "h" else None,
                        error_x=error_kw if orientation == "h" else None,
                        showlegend=show_legend,
                    ),
                    **pos_kw,
                )
            return
        x_labels = label_values(sub_df, group_col).tolist()
        vals = series_values(sub_df, y_col).tolist()
        plus, minus = _bound_error_arrays(
            sub_df, err_plus_col=err_plus_col, err_minus_col=err_minus_col
        )
        error_kw = _plotly_error_kw(plus, minus)
        fig.add_trace(
            go.Bar(
                name=y_col,
                x=x_labels if orientation != "h" else vals,
                y=vals if orientation != "h" else x_labels,
                orientation=orientation,
                marker_color=[palette_color(palette, i) for i in range(len(x_labels))],
                error_y=error_kw if orientation != "h" else None,
                error_x=error_kw if orientation == "h" else None,
                showlegend=show_legend,
            ),
            **pos_kw,
        )

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly bar chart from pre-computed rows."""
        if df.is_empty():
            return None
        x_col, y_cols, color_col, facet_col, facet_row = self._resolve_columns(df)
        group_col = x_col or color_col
        if not y_cols or group_col is None:
            return None

        palette = get_color_palette()
        orientation = str(self.spec.options.get("orientation", "v"))
        barmode = str(self.spec.options.get("barmode", "group"))
        err_plus_col, err_minus_col = self._error_columns()

        if facet_col or facet_row:
            row_vals, col_vals, panels = self._facet_partitions(
                df, facet_col, facet_row
            )
            fig = make_subplots(
                rows=len(row_vals),
                cols=len(col_vals),
                subplot_titles=[p[2] for p in panels],
                shared_yaxes=True,
            )
            for idx, (r_idx, c_idx, _title, sub_df) in enumerate(panels):
                self._add_plotly_panel(
                    fig,
                    sub_df,
                    x_col=x_col,
                    y_cols=y_cols,
                    color_col=color_col,
                    group_col=group_col,
                    orientation=orientation,
                    err_plus_col=err_plus_col,
                    err_minus_col=err_minus_col,
                    palette=palette,
                    row=r_idx,
                    col=c_idx,
                    show_legend=idx == 0,
                )
        else:
            fig = go.Figure()
            self._add_plotly_panel(
                fig,
                df,
                x_col=x_col,
                y_cols=y_cols,
                color_col=color_col,
                group_col=group_col,
                orientation=orientation,
                err_plus_col=err_plus_col,
                err_minus_col=err_minus_col,
                palette=palette,
                row=None,
                col=None,
                show_legend=True,
            )

        y_title = self.format_label(y_cols[0].replace("_", " ").title(), y_cols[0])
        x_title = group_col.replace("_", " ").title()
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or y_title,
            barmode=barmode,
            xaxis_title=x_title if orientation != "h" else y_title,
            yaxis_title=y_title if orientation != "h" else x_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _draw_static_panel(
        self,
        ax: plt.Axes,
        sub_df: pl.DataFrame,
        *,
        x_col: str | None,
        y_cols: list[str],
        color_col: str | None,
        group_col: str,
        err_plus_col: str | None,
        err_minus_col: str | None,
        palette: list[str],
    ) -> str:
        if sub_df.is_empty():
            return "Value"
        if len(y_cols) > 1:
            x_labels = label_values(sub_df, group_col).tolist()
            xs = np.arange(len(x_labels))
            width = 0.8 / len(y_cols)
            for idx, y_col in enumerate(y_cols):
                offset = (idx - (len(y_cols) - 1) / 2.0) * width
                ys = series_values(sub_df, y_col)
                ax.bar(
                    xs + offset,
                    ys,
                    width=width,
                    label=y_col,
                    color=palette_color(palette, idx),
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels)
            ax.legend()
            return "Value"

        y_col = y_cols[0]
        x_labels = label_values(sub_df, group_col).tolist()
        if color_col and color_col != group_col:
            uniq_x = list(dict.fromkeys(x_labels))
            c_labels = label_values(sub_df, color_col).tolist()
            uniq_c = list(dict.fromkeys(c_labels))
            xs = np.arange(len(uniq_x))
            width = 0.8 / max(1, len(uniq_c))
            val_map = {
                (xv, cv): float(yv) if np.isfinite(yv) else np.nan
                for xv, cv, yv in zip(
                    x_labels, c_labels, series_values(sub_df, y_col), strict=True
                )
            }
            plus, minus = _bound_error_arrays(
                sub_df, err_plus_col=err_plus_col, err_minus_col=err_minus_col
            )
            err_map = (
                {
                    (xv, cv): (float(p), float(m))
                    for xv, cv, p, m in zip(
                        x_labels, c_labels, plus, minus, strict=True
                    )
                }
                if plus is not None and minus is not None
                else {}
            )
            for idx, cv in enumerate(uniq_c):
                offset = (idx - (len(uniq_c) - 1) / 2.0) * width
                ys = np.array([val_map.get((xv, cv), np.nan) for xv in uniq_x])
                ax.bar(
                    xs + offset,
                    ys,
                    width=width,
                    label=cv,
                    color=palette_color(palette, idx),
                )
                if err_map:
                    ps = np.array(
                        [err_map.get((xv, cv), (np.nan, np.nan))[0] for xv in uniq_x]
                    )
                    ms = np.array(
                        [err_map.get((xv, cv), (np.nan, np.nan))[1] for xv in uniq_x]
                    )
                    valid = np.isfinite(ys) & np.isfinite(ps) & np.isfinite(ms)
                    if valid.any():
                        ax.errorbar(
                            xs[valid] + offset,
                            ys[valid],
                            yerr=[ms[valid], ps[valid]],
                            fmt="none",
                            ecolor="black",
                            capsize=3,
                        )
            ax.set_xticks(xs)
            ax.set_xticklabels(uniq_x)
            ax.legend(title=color_col)
        else:
            xs = np.arange(len(x_labels))
            ys = series_values(sub_df, y_col)
            colors = [palette_color(palette, idx) for idx in range(len(xs))]
            ax.bar(xs, ys, color=colors)
            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels)
            plus, minus = _bound_error_arrays(
                sub_df, err_plus_col=err_plus_col, err_minus_col=err_minus_col
            )
            if plus is not None and minus is not None:
                valid_err = np.isfinite(ys) & np.isfinite(plus) & np.isfinite(minus)
                if valid_err.any():
                    ax.errorbar(
                        xs[valid_err],
                        ys[valid_err],
                        yerr=[minus[valid_err], plus[valid_err]],
                        fmt="none",
                        ecolor="black",
                        capsize=4,
                    )
        return self.format_label(y_col.replace("_", " ").title(), y_col)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib bar chart from pre-computed rows."""
        if df.is_empty():
            return
        x_col, y_cols, color_col, facet_col, facet_row = self._resolve_columns(df)
        group_col = x_col or color_col
        if not y_cols or group_col is None:
            return

        palette = get_color_palette()
        err_plus_col, err_minus_col = self._error_columns()

        if facet_col or facet_row:
            row_vals, col_vals, panels = self._facet_partitions(
                df, facet_col, facet_row
            )
            fig, axes = plt.subplots(
                len(row_vals),
                len(col_vals),
                figsize=(max(5.0, 4.5 * len(col_vals)), max(4.0, 4.0 * len(row_vals))),
                sharey=True,
                squeeze=False,
            )
            y_label = "Value"
            for r_idx, c_idx, panel_title, sub_df in panels:
                ax = axes[r_idx - 1][c_idx - 1]
                y_label = self._draw_static_panel(
                    ax,
                    sub_df,
                    x_col=x_col,
                    y_cols=y_cols,
                    color_col=color_col,
                    group_col=group_col,
                    err_plus_col=err_plus_col,
                    err_minus_col=err_minus_col,
                    palette=palette,
                )
                ax.set_title(panel_title)
                ax.set_xlabel(group_col.replace("_", " ").title())
                if c_idx == 1:
                    ax.set_ylabel(y_label)
                if len(sub_df) > 6:
                    ax.tick_params(axis="x", rotation=30)
            if self.spec.title:
                fig.suptitle(self.spec.title, fontweight="bold")
            fig.tight_layout()
            self.save_figure(fig, out_path)
            plt.close(fig)
            return

        fig, ax = new_figure()
        y_label = self._draw_static_panel(
            ax,
            df,
            x_col=x_col,
            y_cols=y_cols,
            color_col=color_col,
            group_col=group_col,
            err_plus_col=err_plus_col,
            err_minus_col=err_minus_col,
            palette=palette,
        )
        if len(df) > 6:
            ax.tick_params(axis="x", rotation=30)
        finish_axes(
            ax,
            title=self.spec.title or y_label,
            x_label=group_col.replace("_", " ").title(),
            y_label=y_label,
        )
        self.save_figure(fig, out_path)
        plt.close(fig)
