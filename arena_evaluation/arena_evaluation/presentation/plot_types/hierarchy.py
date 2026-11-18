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
        label, val, color_idx = nodes[0]
        return [(x, y, w, h, label, val, color_idx)]

    total = sum(v for _, v, _ in nodes)
    if total <= 0.0:
        return []

    half = total / 2.0
    acc = 0.0
    split_idx = 1
    for i, (_, val, _) in enumerate(nodes[:-1], start=1):
        acc += val
        split_idx = i
        if acc >= half:
            break

    left = nodes[:split_idx]
    right = nodes[split_idx:]
    left_sum = sum(v for _, v, _ in left)
    frac = left_sum / total

    if w >= h:
        lw = w * frac
        return _split_treemap(left, x, y, lw, h) + _split_treemap(
            right, x + lw, y, w - lw, h
        )
    lh = h * frac
    return _split_treemap(left, x, y, w, lh) + _split_treemap(
        right, x, y + lh, w, h - lh
    )


class HierarchyRenderer(BasePlotRenderer):
    """Renders hierarchical treemap and sunburst plots from bound columns."""

    PLOT_TYPE = "hierarchy"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "mode",
        "variant",
    }

    def _prepare_frame(self, df: pl.DataFrame) -> tuple[list[str], str, pl.DataFrame]:
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
        work = df
        for col in path_cols:
            work = work.with_columns(pl.Series(col, label_values(work, col)))
        neg = work.filter(pl.col(values_col).cast(pl.Float64) < 0)
        if not neg.is_empty():
            raise ValueError(
                f"Plot {self.spec.id!r}: hierarchy value column {values_col!r} contains negative values."
            )
        work = work.filter(
            pl.col(values_col).cast(pl.Float64).is_not_null()
            & pl.col(values_col).cast(pl.Float64).is_finite()
            & (pl.col(values_col).cast(pl.Float64) > 0)
        )
        return path_cols, values_col, work

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        """Render an interactive Plotly treemap or sunburst."""
        # differs from overhaul/source:plot_types/hierarchy.py:29-66: dendrogram/clustermap dropped; binds channels directly
        if df.is_empty():
            return None
        path_cols, values_col, work = self._prepare_frame(df)
        if work.is_empty():
            return None
        color_col = self.resolve_channel("color")

        selected = list(
            dict.fromkeys(
                [
                    *path_cols,
                    values_col,
                    *([color_col] if color_col else []),
                ]
            )
        )
        pdf = work.select(selected).to_pandas()
        mode = str(
            self.spec.options.get("mode")
            or self.spec.options.get("variant")
            or "treemap"
        ).lower()
        if mode == "sunburst":
            fig = px.sunburst(
                pdf,
                path=path_cols,
                values=values_col,
                color=color_col,
                title=self.spec.title or None,
            )
        else:
            fig = px.treemap(
                pdf,
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
        if df.is_empty():
            return
        path_cols, values_col, work = self._prepare_frame(df)
        if work.is_empty():
            return

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
        top_nodes = [(label, val, idx) for idx, (label, val) in enumerate(top_groups)]
        top_rects = _split_treemap(top_nodes, 0.0, 0.0, 1.0, 1.0)

        leaf_rects: list[tuple[float, float, float, float, str, float, int]] = []
        if len(path_cols) == 1:
            leaf_rects = top_rects
        else:
            for rx, ry, rw, rh, top_label, _, color_idx in top_rects:
                sub = work.filter(pl.col(top_col) == top_label)
                sub_nodes = [
                    (
                        " / ".join(str(row[c]) for c in path_cols[1:]),
                        float(row[values_col]),
                        color_idx,
                    )
                    for row in sub.sort(values_col, descending=True).to_dicts()
                ]
                for lx, ly, lw, lh, child_label, val, cidx in _split_treemap(
                    sub_nodes, rx, ry, rw, rh
                ):
                    leaf_rects.append(
                        (lx, ly, lw, lh, f"{top_label}\n{child_label}", val, cidx)
                    )

        fig, ax = new_figure(width_in=10.0, height_in=6.5)
        for rx, ry, rw, rh, label, val, color_idx in leaf_rects:
            rect = mpatches.Rectangle(
                (rx, ry),
                rw,
                rh,
                facecolor=palette_color(palette, color_idx),
                edgecolor="white",
                linewidth=1.5,
                alpha=0.88,
            )
            ax.add_patch(rect)
            if rw >= 0.08 and rh >= 0.07:
                ax.text(
                    rx + rw / 2.0,
                    ry + rh / 2.0,
                    f"{label}\n{val:.2g}",
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
