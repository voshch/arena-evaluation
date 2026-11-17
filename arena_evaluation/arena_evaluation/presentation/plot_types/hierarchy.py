"""Hierarchical breakdown renderer for treemap and sunburst charts."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px
import polars as pl

from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    label_values,
    new_figure,
    palette_color,
)

_TREEMAP_MODE: str = "treemap"


def _split_treemap(
    nodes: list[tuple[str, float, int]],
    x: float,
    y: float,
    w: float,
    h: float,
) -> list[tuple[float, float, float, float, str, float, int]]:
    """Binary area-proportional treemap partition along the longer rectangle axis."""
    if not nodes:
        return []
    if len(nodes) == 1:
        label, value, color_index = nodes[0]
        return [(x, y, w, h, label, value, color_index)]
    total = sum(value for _, value, _ in nodes)
    if total <= 0.0:
        return []
    half = total / 2.0
    accumulated = 0.0
    split_index = 1
    for index, (_, value, _) in enumerate(nodes[:-1], start=1):
        accumulated += value
        split_index = index
        if accumulated >= half:
            break
    left, right = nodes[:split_index], nodes[split_index:]
    fraction = sum(value for _, value, _ in left) / total
    if w >= h:
        left_width = w * fraction
        left_rects = _split_treemap(left, x, y, left_width, h)
        right_rects = _split_treemap(right, x + left_width, y, w - left_width, h)
    else:
        left_height = h * fraction
        left_rects = _split_treemap(left, x, y, w, left_height)
        right_rects = _split_treemap(right, x, y + left_height, w, h - left_height)
    return left_rects + right_rects


class HierarchyRenderer(BasePlotRenderer):
    """Renders hierarchical treemap and sunburst plots from bound columns."""

    PLOT_TYPE = "hierarchy"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "mode",
        "variant",
    }

    def _drawable_frame(
        self, df: pl.DataFrame
    ) -> tuple[list[str], str, pl.DataFrame] | None:
        """Resolve the hierarchy columns and keep only positive finite tiles."""
        if df.is_empty():
            return None
        path_cols = self.resolve_channel_list("hierarchy")
        if not path_cols:
            raise ValueError(
                f"Plot {self.spec.id!r}: hierarchy plot requires channel 'hierarchy'."
            )
        values_col = self.resolve_channel("y")
        if not values_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: hierarchy plot requires channel 'y' for tile values."
            )
        self.validate_unique_cells(df, path_cols)
        work = df.with_columns(
            [pl.Series(column, label_values(df, column)) for column in path_cols]
        )
        numeric = work[values_col].cast(pl.Float64)
        if (numeric < 0).any():
            raise ValueError(
                f"Plot {self.spec.id!r}: hierarchy value column {values_col!r} "
                "contains negative values."
            )
        work = work.filter(numeric.is_not_null() & numeric.is_finite() & (numeric > 0))
        if work.is_empty():
            return None
        return path_cols, values_col, work

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly treemap or sunburst."""
        prepared = self._drawable_frame(df)
        if prepared is None:
            return None
        path_cols, values_col, work = prepared
        color_col = self.resolve_channel("color")
        selected = list(
            dict.fromkeys([*path_cols, values_col, *([color_col] if color_col else [])])
        )
        mode = str(
            self.spec.options.get("mode")
            or self.spec.options.get("variant")
            or _TREEMAP_MODE
        ).lower()
        draw = {"sunburst": px.sunburst}.get(mode, px.treemap)
        fig = draw(
            work.select(selected).to_pandas(),
            path=path_cols,
            values=values_col,
            color=color_col,
            title=self.spec.title or None,
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(t=50 if self.spec.title else 30, b=30, l=30, r=30),
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib treemap breakdown for the hierarchy levels."""
        prepared = self._drawable_frame(df)
        if prepared is None:
            return
        path_cols, values_col, work = prepared
        palette = get_color_palette()
        top_col = path_cols[0]
        top_groups = sorted(
            (
                (
                    str(sub[top_col][0]),
                    float(sub[values_col].cast(pl.Float64).sum()),
                )
                for sub in work.partition_by(top_col, maintain_order=True)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        top_rects = _split_treemap(
            [
                (label, value, color_index)
                for color_index, (label, value) in enumerate(top_groups)
            ],
            0.0,
            0.0,
            1.0,
            1.0,
        )
        if len(path_cols) == 1:
            rects = top_rects
        else:
            rects = []
            for rx, ry, rw, rh, top_label, _, color_index in top_rects:
                sub = work.filter(pl.col(top_col) == top_label)
                sub_nodes = [
                    (
                        " / ".join(str(row[column]) for column in path_cols[1:]),
                        float(row[values_col]),
                        color_index,
                    )
                    for row in sub.sort(values_col, descending=True).to_dicts()
                ]
                rects.extend(
                    (lx, ly, lw, lh, f"{top_label}\n{child_label}", value, index)
                    for lx, ly, lw, lh, child_label, value, index in _split_treemap(
                        sub_nodes, rx, ry, rw, rh
                    )
                )
        fig, ax = new_figure(width_in=10.0, height_in=6.5)
        for rx, ry, rw, rh, label, value, color_index in rects:
            ax.add_patch(
                mpatches.Rectangle(
                    (rx, ry),
                    rw,
                    rh,
                    facecolor=palette_color(palette, color_index),
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.88,
                )
            )
            if rw >= 0.08 and rh >= 0.07:
                ax.text(
                    rx + rw / 2.0,
                    ry + rh / 2.0,
                    f"{label}\n{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        if self.spec.title:
            ax.set_title(self.spec.title, pad=12, fontsize=12, fontweight="bold")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
