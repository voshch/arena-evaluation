"""Scatter plot renderer for bivariate and multi-dimensional relationships."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    finish_axes,
    label_values,
    new_figure,
    palette_color,
)


class ScatterRenderer(BasePlotRenderer):
    """Renders 2D scatter plots with optional color, size, symbol, and facets."""

    PLOT_TYPE = "scatter"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "alpha"
    }

    def _resolve_columns(self, df: pl.DataFrame) -> dict[str, str | None]:
        return {
            "x": self.resolve_channel("x"),
            "y": self.resolve_channel("y"),
            "color": self.resolve_channel("color"),
            "size": self.resolve_channel("size"),
            "symbol": self.resolve_channel("symbol"),
            "facet_col": self.resolve_channel("facet_col"),
            "facet_row": self.resolve_channel("facet_row"),
        }

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly scatter plot."""
        if df.is_empty():
            return None
        cols = self._resolve_columns(df)
        x_col, y_col = cols["x"], cols["y"]
        if not x_col or not y_col:
            return None

        selected = [c for c in dict.fromkeys(cols.values()) if c is not None]
        pdf = df.select(selected).to_pandas()
        if cols["color"] and df.schema[cols["color"]] in (
            pl.Utf8,
            pl.Categorical,
            pl.Boolean,
        ):
            pdf[cols["color"]] = pd.Series(label_values(df, cols["color"]))

        alpha = float(self.spec.options.get("alpha", 0.8))
        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=cols["color"],
            size=cols["size"],
            symbol=cols["symbol"],
            facet_col=cols["facet_col"],
            facet_row=cols["facet_row"],
            opacity=alpha,
            color_discrete_sequence=get_color_palette(),
            title=self.spec.title or f"{y_col} vs {x_col}",
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title=self.format_label(x_col.replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(y_col.replace("_", " ").title(), y_col),
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn scatter plot."""
        if df.is_empty():
            return
        cols = self._resolve_columns(df)
        x_col, y_col = cols["x"], cols["y"]
        if not x_col or not y_col:
            return

        selected = [c for c in dict.fromkeys(cols.values()) if c is not None]
        pdf = df.select(selected).to_pandas()
        if cols["color"] and df.schema[cols["color"]] in (
            pl.Utf8,
            pl.Categorical,
            pl.Boolean,
        ):
            pdf[cols["color"]] = pd.Series(label_values(df, cols["color"]))

        palette = get_color_palette()
        alpha = float(self.spec.options.get("alpha", 0.8))
        x_label = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_label = self.format_label(y_col.replace("_", " ").title(), y_col)
        title = self.spec.title or f"{y_col} vs {x_col}"

        if cols["facet_col"] or cols["facet_row"]:
            rel_kw: dict[str, object] = {
                "data": pdf,
                "x": x_col,
                "y": y_col,
                "kind": "scatter",
                "col": cols["facet_col"],
                "row": cols["facet_row"],
                "alpha": alpha,
                "height": 4.2,
                "aspect": 1.25,
            }
            if cols["color"]:
                rel_kw["hue"] = cols["color"]
                rel_kw["palette"] = palette
            else:
                rel_kw["color"] = palette_color(palette, 0)
            if cols["size"]:
                rel_kw["size"] = cols["size"]
            if cols["symbol"]:
                rel_kw["style"] = cols["symbol"]
            grid = sns.relplot(**rel_kw)
            grid.set_axis_labels(x_label, y_label)
            grid.figure.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
            grid.figure.tight_layout()
            self.save_figure(grid.figure, out_path)
            plt.close(grid.figure)
            return

        fig, ax = new_figure()
        scatter_kw: dict[str, object] = {
            "data": pdf,
            "x": x_col,
            "y": y_col,
            "ax": ax,
            "alpha": alpha,
        }
        if cols["color"]:
            scatter_kw["hue"] = cols["color"]
            scatter_kw["palette"] = palette
        else:
            scatter_kw["color"] = palette_color(palette, 0)
        if cols["size"]:
            scatter_kw["size"] = cols["size"]
        if cols["symbol"]:
            scatter_kw["style"] = cols["symbol"]

        sns.scatterplot(**scatter_kw)
        finish_axes(
            ax,
            title=title,
            x_label=x_label,
            y_label=y_label,
            grid_axis="both",
        )
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
