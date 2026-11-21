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
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {"alpha"}

    def _resolve_columns(self, df: pl.DataFrame) -> dict[str, str | None]:
        return {name: self.resolve_channel(name) for name in ("x", "y", "color", "size", "symbol", "facet_col", "facet_row")}

    def _prepare(self, df: pl.DataFrame) -> tuple[str, str, dict[str, str | None], pd.DataFrame] | None:
        if df.is_empty():
            return None
        cols = self._resolve_columns(df)
        x_col, y_col = cols["x"], cols["y"]
        if not x_col or not y_col:
            return None
        selected = [c for c in dict.fromkeys(cols.values()) if c is not None]
        pdf = df.select(selected).to_pandas()
        color_col = cols["color"]
        categorical = (pl.Utf8, pl.Categorical, pl.Boolean)
        if color_col and df.schema[color_col] in categorical:
            pdf[color_col] = pd.Series(label_values(df, color_col))
        return x_col, y_col, cols, pdf

    def _labels(self, x_col: str, y_col: str) -> tuple[str, str, str]:
        x_label = self.format_label(x_col.replace("_", " ").title(), x_col)
        y_label = self.format_label(y_col.replace("_", " ").title(), y_col)
        return x_label, y_label, self.spec.title or f"{y_col} vs {x_col}"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly scatter plot."""
        prepared = self._prepare(df)
        if prepared is None:
            return None
        x_col, y_col, cols, pdf = prepared
        x_label, y_label, title = self._labels(x_col, y_col)

        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=cols["color"],
            size=cols["size"],
            symbol=cols["symbol"],
            facet_col=cols["facet_col"],
            facet_row=cols["facet_row"],
            opacity=float(self.spec.options.get("alpha", 0.8)),
            color_discrete_sequence=get_color_palette(),
            title=title,
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(t=55, b=60, l=65, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn scatter plot."""
        prepared = self._prepare(df)
        if prepared is None:
            return
        x_col, y_col, cols, pdf = prepared
        x_label, y_label, title = self._labels(x_col, y_col)
        palette = get_color_palette()
        artist_kwargs: dict[str, object] = {"alpha": float(self.spec.options.get("alpha", 0.8))}
        if cols["color"]:
            artist_kwargs.update(hue=cols["color"], palette=palette)
        else:
            artist_kwargs["color"] = palette_color(palette, 0)
        if cols["size"]:
            artist_kwargs["size"] = cols["size"]
        if cols["symbol"]:
            artist_kwargs["style"] = cols["symbol"]

        if cols["facet_col"] or cols["facet_row"]:
            grid = sns.relplot(
                data=pdf,
                x=x_col,
                y=y_col,
                kind="scatter",
                col=cols["facet_col"],
                row=cols["facet_row"],
                height=4.2,
                aspect=1.25,
                **artist_kwargs,
            )
            grid.set_axis_labels(x_label, y_label)
            grid.figure.suptitle(title, y=1.02, fontsize=12, fontweight="bold")
            grid.figure.tight_layout()
            self.save_figure(grid.figure, out_path)
            plt.close(grid.figure)
            return

        fig, ax = new_figure()
        sns.scatterplot(data=pdf, x=x_col, y=y_col, ax=ax, **artist_kwargs)
        finish_axes(ax, title=title, x_label=x_label, y_label=y_label, grid_axis="both")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
