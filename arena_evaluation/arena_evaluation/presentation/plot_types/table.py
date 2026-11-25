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


def _format_cell(value: object, *, reason: object = None, fmt: str | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return str(reason) if reason is not None else "NULL"
    if fmt:
        return fmt.format(value) if "{" in fmt else format(value, fmt)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metric_label(value: object, reason: object, labels: dict[str, str]) -> str:
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    if missing and reason is not None:
        return str(reason)
    return "NULL" if value is None else labels.get(str(value), str(value))


def _string_map(raw: object) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {}


def _formatted_cells(
    df: pl.DataFrame,
    cols: list[str],
    raw_format: object,
    labels: dict[str, str],
) -> list[list[str]]:
    default_fmt = raw_format if isinstance(raw_format, str) else None
    per_col_fmt = _string_map(raw_format)
    metric_names: list[str] | None = None
    if "metric" in cols:
        # agent-contracts: ignore AC004 - cell export
        metric_names = [str(metric) for metric in df["metric"].to_list()]
    cells_by_col: list[list[str]] = []
    for col in cols:
        col_fmt = per_col_fmt.get(col, default_fmt)
        reason_col = f"{col}_null_reason"
        values = df[col].to_list()  # agent-contracts: ignore AC004 - cell export
        if reason_col in df.columns:  # agent-contracts: ignore AC003 - optional companion null_reason column
            reasons = df[reason_col].to_list()  # agent-contracts: ignore AC004 - explicit table cell serialization boundary
        else:
            reasons = [None] * len(values)
        if metric_names is not None and col in _STAT_VALUE_COLS:
            fmts = [per_col_fmt.get(metric, col_fmt) for metric in metric_names]
        else:
            fmts = [col_fmt] * len(values)
        cells_by_col.append([(_metric_label(value, reason, labels) if col == "metric" and labels else _format_cell(value, reason=reason, fmt=fmt)) for value, reason, fmt in zip(values, reasons, fmts, strict=True)])
    return cells_by_col


class TableRenderer(BasePlotRenderer):
    """Renders tabular data directly from bound DataFrame columns."""

    PLOT_TYPE = "table"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "format",
        "labels",
    }

    def _columns(self, df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
        """Resolve the rendered columns once, with their headers and formatted cells."""
        cols = list(
            dict.fromkeys(
                column
                for column in (
                    self.resolve_channel("x"),
                    self.resolve_channel("color"),
                    *self.resolve_channel_list("y"),
                )
                if column
            )
        )
        if not cols:
            raise ValueError(f"Plot {self.spec.id!r}: table plot requires declared 'x'/'y'/'color' channels.")
        labels = _string_map(self.spec.options.get("labels"))
        headers = [labels.get(col, col.replace("_", " ").title()) for col in cols]
        cells = _formatted_cells(df, cols, self.spec.options.get("format"), labels)
        return headers, cells

    def tableau(self, df: pl.DataFrame) -> tuple[list[str], list[list[str]]]:
        """Headers and formatted cell rows of this table, the same cells the report draws."""
        headers, cells = self._columns(df)
        return headers, [list(row) for row in zip(*cells, strict=True)]

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly table."""
        if df.is_empty():
            return None
        headers, cells = self._columns(df)
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

        fig, ax = plt.subplots(figsize=(max(8.0, len(headers) * 1.6), max(2.5, 0.8 + len(rows) * 0.4)))
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
