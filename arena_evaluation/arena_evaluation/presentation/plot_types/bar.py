"""Bar plot renderer for pre-aggregated values and optional error bounds."""

from __future__ import annotations

import pathlib
from typing import ClassVar, NamedTuple

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


class _BarGroup(NamedTuple):
    label: str
    x_labels: list[str]
    values: np.ndarray
    plus: np.ndarray | None
    minus: np.ndarray | None
    palette_index: int | None


class _BarContext(NamedTuple):
    x_col: str | None
    y_cols: list[str]
    color_col: str | None
    group_col: str
    facet_col: str | None
    facet_row: str | None
    err_plus_col: str | None
    err_minus_col: str | None
    palette: list[str]


def _error_arrays(frame: pl.DataFrame, *, plus_col: str | None, minus_col: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read pre-computed upper/lower error arrays without filling NULLs with 0.0."""
    if plus_col is None:
        return None, None
    plus = series_values(frame, plus_col)
    return plus, series_values(frame, minus_col) if minus_col is not None else plus


def _plotly_error(plus: np.ndarray | None, minus: np.ndarray | None) -> dict[str, object] | None:
    if plus is None or minus is None:
        return None
    return {
        "type": "data",
        "array": plus.tolist(),
        "arrayminus": minus.tolist(),
        "symmetric": False,
    }


def _draw_errorbars(
    ax: plt.Axes,
    positions: np.ndarray,
    values: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    *,
    capsize: int,
) -> None:
    valid = np.isfinite(values) & np.isfinite(plus) & np.isfinite(minus)
    if not valid.any():
        return
    ax.errorbar(
        positions[valid],
        values[valid],
        yerr=[minus[valid], plus[valid]],
        fmt="none",
        ecolor="black",
        capsize=capsize,
    )


def _bar_groups(frame: pl.DataFrame, context: _BarContext) -> tuple[list[str], list[_BarGroup]]:
    """Ordered bar series shared by the Plotly and Matplotlib panels.

    ``palette_index`` is ``None`` for a single series, which paints each bar
    with its own palette color; otherwise it selects one group color.
    """
    x_labels = label_values(frame, context.group_col).tolist()
    if len(context.y_cols) > 1:
        return x_labels, [_BarGroup(y_col, x_labels, series_values(frame, y_col), None, None, idx) for idx, y_col in enumerate(context.y_cols)]
    y_col = context.y_cols[0]
    if not (context.x_col and context.color_col and context.color_col != context.x_col):
        plus, minus = _error_arrays(frame, plus_col=context.err_plus_col, minus_col=context.err_minus_col)
        return x_labels, [_BarGroup(y_col, x_labels, series_values(frame, y_col), plus, minus, None)]
    tagged = frame.with_columns(pl.Series("_bar_color_label", label_values(frame, context.color_col)))
    groups: list[_BarGroup] = []
    for idx, sub in enumerate(tagged.partition_by("_bar_color_label", maintain_order=True)):
        plus, minus = _error_arrays(sub, plus_col=context.err_plus_col, minus_col=context.err_minus_col)
        groups.append(
            _BarGroup(
                str(sub["_bar_color_label"][0]),
                label_values(sub, context.x_col).tolist(),
                series_values(sub, y_col),
                plus,
                minus,
                idx,
            )
        )
    return x_labels, groups


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
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"err_plus", "err_minus", "error_col"})

    def _resolve_columns(self, df: pl.DataFrame) -> _BarContext:
        x_col = self.resolve_channel("x")
        y_cols = self.resolve_channel_list("y")
        if not y_cols:
            raise ValueError(f"Plot {self.spec.id!r}: bar plot requires channel 'y'.")
        color_col = self.resolve_channel("color")
        facet_col = self.resolve_channel("facet_col")
        facet_row = self.resolve_channel("facet_row")
        group_col = x_col or color_col
        if group_col is None:
            raise ValueError(f"Plot {self.spec.id!r}: bar plot requires channel 'x' or 'color'.")
        cell_keys = [c for c in dict.fromkeys((group_col, color_col, facet_col, facet_row)) if c]
        self.validate_unique_cells(df, cell_keys)

        opts = self.spec.options
        err_plus = opts.get("err_plus") or opts.get("error_col")
        err_plus_col = err_plus.strip() if isinstance(err_plus, str) and err_plus.strip() else None
        err_minus = opts.get("err_minus")
        err_minus_col = str(err_minus).strip() if isinstance(err_minus, str) and err_minus.strip() else err_plus_col
        return _BarContext(
            x_col,
            y_cols,
            color_col,
            group_col,
            facet_col,
            facet_row,
            err_plus_col,
            err_minus_col,
            get_color_palette(),
        )

    def _panels(self, df: pl.DataFrame, context: _BarContext) -> tuple[bool, list[tuple[int, int, str, pl.DataFrame]]]:
        """Facet panels in row-major order, or a single unsplit panel."""
        if not context.facet_col and not context.facet_row:
            return False, [(1, 1, "", df)]
        col_vals = list(dict.fromkeys(label_values(df, context.facet_col).tolist())) if context.facet_col else [""]
        row_vals = list(dict.fromkeys(label_values(df, context.facet_row).tolist())) if context.facet_row else [""]
        tagged = df
        for alias, column in (
            ("_bar_fcol", context.facet_col),
            ("_bar_frow", context.facet_row),
        ):
            if column:
                tagged = tagged.with_columns(pl.Series(alias, label_values(df, column)))
        panels: list[tuple[int, int, str, pl.DataFrame]] = []
        for r_idx, rv in enumerate(row_vals, start=1):
            for c_idx, cv in enumerate(col_vals, start=1):
                sub = tagged
                parts: list[str] = []
                if context.facet_col:
                    sub = sub.filter(pl.col("_bar_fcol") == cv)
                    parts.append(f"{context.facet_col}={cv}")
                if context.facet_row:
                    sub = sub.filter(pl.col("_bar_frow") == rv)
                    parts.append(f"{context.facet_row}={rv}")
                panels.append((r_idx, c_idx, ", ".join(parts), sub))
        return True, panels

    def _add_plotly_panel(
        self,
        fig: go.Figure,
        sub_df: pl.DataFrame,
        context: _BarContext,
        *,
        row: int | None,
        col: int | None,
        show_legend: bool,
    ) -> None:
        if sub_df.is_empty():
            return
        orientation = str(self.spec.options.get("orientation", "v"))
        horizontal = orientation == "h"
        _, groups = _bar_groups(sub_df, context)
        for group in groups:
            error_kw = _plotly_error(group.plus, group.minus)
            values = group.values.tolist()
            marker_color: object = palette_color(context.palette, group.palette_index) if group.palette_index is not None else [palette_color(context.palette, i) for i in range(len(group.x_labels))]
            trace = go.Bar(
                name=group.label,
                x=values if horizontal else group.x_labels,
                y=group.x_labels if horizontal else values,
                orientation=orientation,
                marker_color=marker_color,
                error_y=None if horizontal else error_kw,
                error_x=error_kw if horizontal else None,
                showlegend=show_legend,
            )
            if row is not None and col is not None:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly bar chart from pre-computed rows."""
        if df.is_empty():
            return None
        context = self._resolve_columns(df)
        faceted, panels = self._panels(df, context)
        if faceted:
            fig = make_subplots(
                rows=max(p[0] for p in panels),
                cols=max(p[1] for p in panels),
                subplot_titles=[p[2] for p in panels],
                shared_yaxes=True,
            )
        else:
            fig = go.Figure()
        for idx, (r_idx, c_idx, _title, sub_df) in enumerate(panels):
            self._add_plotly_panel(
                fig,
                sub_df,
                context,
                row=r_idx if faceted else None,
                col=c_idx if faceted else None,
                show_legend=not faceted or idx == 0,
            )

        y_title = self.format_label(context.y_cols[0].replace("_", " ").title(), context.y_cols[0])
        x_title = context.group_col.replace("_", " ").title()
        horizontal = str(self.spec.options.get("orientation", "v")) == "h"
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or y_title,
            barmode=str(self.spec.options.get("barmode", "group")),
            xaxis_title=y_title if horizontal else x_title,
            yaxis_title=x_title if horizontal else y_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def _draw_static_panel(self, ax: plt.Axes, sub_df: pl.DataFrame, context: _BarContext) -> str:
        if sub_df.is_empty():
            return "Value"
        x_labels, groups = _bar_groups(sub_df, context)
        if len(context.y_cols) > 1:
            xs = np.arange(len(x_labels))
            width = 0.8 / len(groups)
            for idx, group in enumerate(groups):
                offset = (idx - (len(groups) - 1) / 2.0) * width
                ax.bar(
                    xs + offset,
                    group.values,
                    width=width,
                    label=group.label,
                    color=palette_color(context.palette, idx),
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels)
            ax.legend()
            return "Value"

        y_col = context.y_cols[0]
        y_label = self.format_label(y_col.replace("_", " ").title(), y_col)
        if groups[0].palette_index is None:
            xs = np.arange(len(x_labels))
            group = groups[0]
            colors = [palette_color(context.palette, i) for i in range(len(xs))]
            ax.bar(xs, group.values, color=colors)
            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels)
            if group.plus is not None and group.minus is not None:
                _draw_errorbars(ax, xs, group.values, group.plus, group.minus, capsize=4)
            return y_label

        uniq_x = list(dict.fromkeys(x_labels))
        xs = np.arange(len(uniq_x))
        width = 0.8 / max(1, len(groups))
        for idx, group in enumerate(groups):
            offset = (idx - (len(groups) - 1) / 2.0) * width
            value_map = dict(zip(group.x_labels, group.values, strict=True))
            ys = np.array([value_map.get(xv, np.nan) for xv in uniq_x])
            ax.bar(
                xs + offset,
                ys,
                width=width,
                label=group.label,
                color=palette_color(context.palette, idx),
            )
            if group.plus is not None and group.minus is not None:
                plus_map = dict(zip(group.x_labels, group.plus, strict=True))
                minus_map = dict(zip(group.x_labels, group.minus, strict=True))
                _draw_errorbars(
                    ax,
                    xs + offset,
                    ys,
                    np.array([plus_map.get(xv, np.nan) for xv in uniq_x]),
                    np.array([minus_map.get(xv, np.nan) for xv in uniq_x]),
                    capsize=3,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(uniq_x)
        ax.legend(title=context.color_col)
        return y_label

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib bar chart from pre-computed rows."""
        if df.is_empty():
            return
        context = self._resolve_columns(df)
        faceted, panels = self._panels(df, context)
        if faceted:
            n_rows = max(p[0] for p in panels)
            n_cols = max(p[1] for p in panels)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(max(5.0, 4.5 * n_cols), max(4.0, 4.0 * n_rows)),
                sharey=True,
                squeeze=False,
            )
        else:
            fig, ax = new_figure()
            axes = [[ax]]

        ax = axes[0][0]
        y_label = "Value"
        for r_idx, c_idx, panel_title, sub_df in panels:
            ax = axes[r_idx - 1][c_idx - 1]
            y_label = self._draw_static_panel(ax, sub_df, context)
            if faceted:
                ax.set_title(panel_title)
                ax.set_xlabel(context.group_col.replace("_", " ").title())
                if c_idx == 1:
                    ax.set_ylabel(y_label)
            if len(sub_df) > 6:
                ax.tick_params(axis="x", rotation=30)
        if faceted:
            if self.spec.title:
                fig.suptitle(self.spec.title, fontweight="bold")
            fig.tight_layout()
        else:
            finish_axes(
                ax,
                title=self.spec.title or y_label,
                x_label=context.group_col.replace("_", " ").title(),
                y_label=y_label,
            )
        self.save_figure(fig, out_path)
        plt.close(fig)
