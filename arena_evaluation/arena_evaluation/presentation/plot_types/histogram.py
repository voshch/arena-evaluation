"""Histogram plot renderer for pre-binned distribution tables."""

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
    series_values,
)


class HistogramRenderer(BasePlotRenderer):
    """Renders pre-binned distribution_bins frames as bars without in-renderer binning."""

    PLOT_TYPE = "histogram"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "bin_low",
        "bin_high",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"bin_low", "bin_high"})

    def _resolve_columns(
        self, df: pl.DataFrame
    ) -> tuple[str | None, str | None, str | None]:
        x_col = self.resolve_channel("x")
        y_col = self.resolve_channel("y")
        if not x_col or not y_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: histogram requires pre-binned 'x' (e.g. bin_center) and 'y' (e.g. count or density)."
            )
        color_col = self.resolve_channel("color")
        cell_keys = [c for c in dict.fromkeys((x_col, color_col)) if c is not None]
        self.validate_unique_cells(df, cell_keys)
        return x_col, y_col, color_col

    def _bar_geometry(
        self, sub: pl.DataFrame, x_col: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        low_col = self.spec.options.get("bin_low")
        high_col = self.spec.options.get("bin_high")
        if (
            isinstance(low_col, str)
            and low_col.strip()
            and isinstance(high_col, str)
            and high_col.strip()
        ):
            lo = series_values(sub, low_col.strip())
            hi = series_values(sub, high_col.strip())
            return 0.5 * (lo + hi), hi - lo
        return series_values(sub, x_col), None

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly histogram from pre-binned counts."""
        # differs from overhaul/source:plot_types/histogram.py:72: renders pre-binned distribution_bins only
        if df.is_empty():
            return None
        x_col, y_col, color_col = self._resolve_columns(df)
        assert x_col and y_col

        palette = get_color_palette()
        fig = go.Figure()

        if color_col:
            labels = label_values(df, color_col)
            for idx, label in enumerate(sorted(set(labels.tolist()))):
                mask = labels == label
                sub = df.filter(pl.Series(mask))
                centers, widths = self._bar_geometry(sub, x_col)
                fig.add_trace(
                    go.Bar(
                        x=centers.tolist(),
                        y=series_values(sub, y_col).tolist(),
                        width=widths.tolist() if widths is not None else None,
                        name=label,
                        marker_color=palette_color(palette, idx),
                        opacity=0.75,
                    )
                )
        else:
            centers, widths = self._bar_geometry(df, x_col)
            fig.add_trace(
                go.Bar(
                    x=centers.tolist(),
                    y=series_values(df, y_col).tolist(),
                    width=widths.tolist() if widths is not None else None,
                    name=y_col,
                    marker_color=palette_color(palette, 0),
                )
            )
        fig.update_layout(barmode="overlay")

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_title = self.format_label(y_col.replace("_", " ").title(), y_col)
        fig.update_layout(
            template="plotly_white",
            title=self.spec.title or x_title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib histogram from pre-binned counts."""
        if df.is_empty():
            return
        x_col, y_col, color_col = self._resolve_columns(df)
        assert x_col and y_col

        palette = get_color_palette()
        fig, ax = new_figure()

        if color_col:
            labels = label_values(df, color_col)
            for idx, label in enumerate(sorted(set(labels.tolist()))):
                mask = labels == label
                sub = df.filter(pl.Series(mask))
                centers, widths = self._bar_geometry(sub, x_col)
                w = widths if widths is not None else 0.8
                ax.bar(
                    centers,
                    series_values(sub, y_col),
                    width=w,
                    alpha=0.65,
                    label=label,
                    color=palette_color(palette, idx),
                )
            ax.legend(title=color_col)
        else:
            centers, widths = self._bar_geometry(df, x_col)
            w = widths if widths is not None else 0.8
            ax.bar(
                centers,
                series_values(df, y_col),
                width=w,
                color=palette_color(palette, 0),
            )

        x_title = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_title = self.format_label(y_col.replace("_", " ").title(), y_col)
        finish_axes(
            ax,
            title=self.spec.title or x_title,
            x_label=x_title,
            y_label=y_title,
        )
        self.save_figure(fig, out_path)
        plt.close(fig)
