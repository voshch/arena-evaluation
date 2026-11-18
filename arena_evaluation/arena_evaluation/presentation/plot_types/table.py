"""Table renderer for structured scorecard and summary tables."""

from __future__ import annotations

import math
import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl

from arena_evaluation.presentation.plot_types.base import BasePlotRenderer

_STAT_VALUE_COLS = frozenset({"mean", "std", "err_plus", "err_minus"})


def _format_cell(
    value: object, *, reason: object = None, fmt: str | None = None
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return str(reason) if reason is not None else "NULL"
    if fmt:
        return fmt.format(value) if "{" in fmt else format(value, fmt)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _column_formats(raw_format: object) -> tuple[str | None, dict[str, str]]:
    if isinstance(raw_format, str):
        return raw_format, {}
    if isinstance(raw_format, dict):
        return None, {str(k): str(v) for k, v in raw_format.items()}
    return None, {}


def _label_map(raw_labels: object) -> dict[str, str]:
    if isinstance(raw_labels, dict):
        return {str(k): str(v) for k, v in raw_labels.items()}
    return {}


def _formatted_cells(
    df: pl.DataFrame,
    cols: list[str],
    raw_format: object,
    raw_labels: object = None,
) -> list[list[str]]:
    default_fmt, per_col_fmt = _column_formats(raw_format)
    labels = _label_map(raw_labels)
    metric_names: list[str] | None = None
    if "metric" in cols:
        raw_m = df["metric"].to_list()  # agent-contracts: ignore AC004 - cell export
        metric_names = [str(m) for m in raw_m]
    result: list[list[str]] = []
    for col in cols:
        col_fmt = per_col_fmt.get(col, default_fmt)
        reason_col = f"{col}_null_reason"
        values = df[col].to_list()  # agent-contracts: ignore AC004 - cell export
        if (
            reason_col in df.columns
        ):  # agent-contracts: ignore AC003 - optional companion null_reason column
            reasons = df[
                reason_col
            ].to_list()  # agent-contracts: ignore AC004 - explicit table cell serialization boundary
        else:
            reasons = [None] * len(values)
        if col == "metric" and labels:
            result.append(
                [
                    str(r)
                    if (v is None or (isinstance(v, float) and math.isnan(v)))
                    and r is not None
                    else ("NULL" if v is None else labels.get(str(v), str(v)))
                    for v, r in zip(values, reasons, strict=True)
                ]
            )
            continue
        if metric_names is not None and col in _STAT_VALUE_COLS:
            result.append(
                [
                    _format_cell(v, reason=r, fmt=per_col_fmt.get(m, col_fmt))
                    for v, r, m in zip(values, reasons, metric_names, strict=True)
                ]
            )
            continue
        result.append(
            [
                _format_cell(v, reason=r, fmt=col_fmt)
                for v, r in zip(values, reasons, strict=True)
            ]
        )
    return result


class TableRenderer(BasePlotRenderer):
    """Renders tabular data directly from bound DataFrame columns."""

    PLOT_TYPE = "table"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "format",
        "labels",
    }

    def _resolve_columns(self, df: pl.DataFrame) -> list[str]:
        y_cols = self.resolve_channel_list("y")
        x_col = self.resolve_channel("x")
        color_col = self.resolve_channel("color")
        chosen = list(dict.fromkeys([c for c in (x_col, color_col, *y_cols) if c]))
        if not chosen:
            raise ValueError(
                f"Plot {self.spec.id!r}: table plot requires declared 'x'/'y'/'color' channels."
            )
        return chosen

    def _headers(self, cols: list[str]) -> list[str]:
        labels = _label_map(self.spec.options.get("labels"))
        return [labels.get(c, c.replace("_", " ").title()) for c in cols]

    def tableau(self, df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
        """Headers and formatted cell rows of this table, the same cells the report draws."""
        cols = self._resolve_columns(df)
        if not cols:
            raise ValueError(
                f"Plot {self.spec.id!r}: table plot resolved no drawable column."
            )
        cells_by_col = _formatted_cells(
            df,
            cols,
            self.spec.options.get("format"),
            self.spec.options.get("labels"),
        )
        return self._headers(cols), [
            list(row) for row in zip(*cells_by_col, strict=True)
        ]

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly table."""
        if df.is_empty():
            return None
        headers, rows = self.tableau(df)
        cells = [list(column) for column in zip(*rows, strict=True)]

        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": headers,
                        "fill_color": "#1e293b",
                        "font": {"color": "white", "size": 12},
                        "align": "left",
                    },
                    cells={
                        "values": cells,
                        "fill_color": "#f8fafc",
                        "font": {"color": "#0f172a", "size": 11},
                        "align": "left",
                    },
                )
            ]
        )
        fig.update_layout(
            title=self.spec.title or "Summary Table",
            margin=dict(t=50, b=20, l=20, r=20),
            height=max(240, 60 + df.height * 28),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib table image."""
        if df.is_empty():
            return
        headers, rows = self.tableau(df)

        fig, ax = plt.subplots(
            figsize=(max(8.0, len(headers) * 1.6), max(2.5, 0.8 + len(rows) * 0.4))
        )
        ax.axis("off")
        tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.3)
        if self.spec.title:
            ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
