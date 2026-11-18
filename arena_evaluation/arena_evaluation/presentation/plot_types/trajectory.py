"""Trajectory plot renderer backed by dim_world_geometry."""

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
    palette_color,
    series_values,
)

_GEOM_STYLES: dict[str, tuple[str, float, bool]] = {
    "zone": ("#e2e8f0", 0.35, True),
    "object": ("#cbd5e1", 0.75, True),
    "elevator": ("#06b6d4", 0.85, True),
    "door": ("#f59e0b", 1.8, False),
    "wall": ("#1e293b", 2.0, False),
}


def _extract_xy_tracks(
    df: pl.DataFrame, *, x_col: str | None, y_col: str | None, color_col: str | None
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Split trajectory samples by structural identity keys (episode_id, robot_id/ped_id, color)."""
    tracks: list[tuple[str, np.ndarray, np.ndarray]] = []
    if not x_col or not y_col:
        return tracks

    split_cols = [
        c
        for c in ("episode_id", "robot_id", "ped_id", "agent_id", color_col)
        if c and c in df.columns
    ]
    split_cols = list(dict.fromkeys(split_cols))

    if split_cols:
        for key, sub in df.partition_by(
            split_cols, as_dict=True, maintain_order=True
        ).items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            if color_col and color_col in split_cols:
                c_val = key_tuple[split_cols.index(color_col)]
                label = "NULL" if c_val is None else str(c_val)
            else:
                label = "/".join("NULL" if k is None else str(k) for k in key_tuple)
            tracks.append((label, series_values(sub, x_col), series_values(sub, y_col)))
    else:
        tracks.append(
            ("trajectory", series_values(df, x_col), series_values(df, y_col))
        )
    return tracks


class TrajectoryRenderer(BasePlotRenderer):
    """Renders robot/pedestrian trajectories overlaid on dim_world_geometry."""

    PLOT_TYPE = "trajectory"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "world",
        "x_col",
        "y_col",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"x_col", "y_col"})

    def _world_elements(
        self, world_name: str | None
    ) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        if not world_name:
            return []
        geom = self.world_geometry
        if geom is None or geom.is_empty() or geom.schema.get("world") is None:
            self.add_notice(f"dim_world_geometry has no rows for world {world_name!r}")
            return []
        sub = geom.filter(pl.col("world") == world_name)
        if sub.is_empty():
            self.add_notice(f"dim_world_geometry has no rows for world {world_name!r}")
            return []

        elements: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        ordered = sub.sort(["element_kind", "element_name", "part_idx", "vertex_idx"])
        for (_, kind, name, _), part in ordered.partition_by(
            ["world", "element_kind", "element_name", "part_idx"],
            as_dict=True,
            maintain_order=True,
        ).items():
            xs = series_values(part, "x")
            ys = series_values(part, "y")
            if len(xs) >= 2:
                elements.append((str(kind), str(name), xs, ys))
        return elements

    def _world_partitions(
        self, df: pl.DataFrame
    ) -> list[tuple[str | None, pl.DataFrame]]:
        opt_world = self.spec.options.get("world")
        has_world = "world" in df.columns  # agent-contracts: ignore AC003 - world
        if isinstance(opt_world, str) and opt_world.strip():
            w = opt_world.strip()
            if has_world:
                sub = df.filter(pl.col("world") == w)
                return [(w, sub)] if not sub.is_empty() else [(w, df)]
            return [(w, df)]
        if has_world and not df.is_empty():
            parts: list[tuple[str | None, pl.DataFrame]] = []
            for key, sub in df.partition_by(
                "world", as_dict=True, maintain_order=True
            ).items():
                w_val = key[0] if isinstance(key, tuple) else key
                parts.append((str(w_val) if w_val is not None else None, sub))
            return parts
        return [(None, df)]

    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """Render an interactive Plotly trajectory plot with dim_world_geometry."""
        # differs from overhaul/source:plot_types/trajectory.py:16-18: uses dim_world_geometry; no processing/ imports, no theta*, no raster map
        if df.is_empty():
            return None
        x_col = self.resolve_channel("x") or (
            str(self.spec.options["x_col"]) if self.spec.options.get("x_col") else None
        )
        y_col = self.resolve_channel("y") or (
            str(self.spec.options["y_col"]) if self.spec.options.get("y_col") else None
        )
        color_col = self.resolve_channel("color")
        if not x_col or not y_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: trajectory plot requires channels 'x' and 'y'."
            )

        html_parts: list[str] = []
        palette = get_color_palette()
        for world_name, w_df in self._world_partitions(df):
            tracks = _extract_xy_tracks(
                w_df, x_col=x_col, y_col=y_col, color_col=color_col
            )
            if not tracks:
                continue
            fig = go.Figure()
            for kind, name, xs, ys in self._world_elements(world_name):
                color, width, closed = _GEOM_STYLES.get(kind, ("#64748b", 1.2, False))
                px_vals = np.append(xs, xs[0]).tolist() if closed else xs.tolist()
                py_vals = np.append(ys, ys[0]).tolist() if closed else ys.tolist()
                fig.add_trace(
                    go.Scatter(
                        x=px_vals,
                        y=py_vals,
                        mode="lines",
                        line={"color": color, "width": width},
                        fill="toself" if kind in ("zone", "object") else None,
                        name=f"{kind}:{name}",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            for idx, (label, xs, ys) in enumerate(tracks):
                fig.add_trace(
                    go.Scatter(
                        x=xs.tolist(),
                        y=ys.tolist(),
                        mode="lines+markers",
                        name=label,
                        line={"color": palette_color(palette, idx), "width": 2.2},
                        marker={"size": 4},
                        connectgaps=False,
                    )
                )
            fig.update_layout(
                template="plotly_white",
                title=self.spec.title
                or (f"Trajectories ({world_name})" if world_name else "Trajectories"),
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
                margin=dict(t=55, b=60, l=65, r=30),
            )
            html_parts.append(fig.to_html(include_plotlyjs=False, full_html=False))
        if not html_parts:
            return None
        return html_parts[0] if len(html_parts) == 1 else html_parts

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib trajectory plot with dim_world_geometry."""
        if df.is_empty():
            return
        x_col = self.resolve_channel("x") or (
            str(self.spec.options["x_col"]) if self.spec.options.get("x_col") else None
        )
        y_col = self.resolve_channel("y") or (
            str(self.spec.options["y_col"]) if self.spec.options.get("y_col") else None
        )
        color_col = self.resolve_channel("color")
        if not x_col or not y_col:
            raise ValueError(
                f"Plot {self.spec.id!r}: trajectory plot requires channels 'x' and 'y'."
            )

        world_parts = self._world_partitions(df)
        n_worlds = len(world_parts)
        fig, axes = plt.subplots(
            1, n_worlds, figsize=(9.0 * n_worlds, 8.0), squeeze=False
        )
        palette = get_color_palette()
        rendered_any = False

        for w_idx, (world_name, w_df) in enumerate(world_parts):
            ax = axes[0, w_idx]
            tracks = _extract_xy_tracks(
                w_df, x_col=x_col, y_col=y_col, color_col=color_col
            )
            if not tracks:
                continue
            rendered_any = True
            for kind, _name, xs, ys in self._world_elements(world_name):
                color, width, closed = _GEOM_STYLES.get(kind, ("#64748b", 1.2, False))
                if closed:
                    ax.fill(
                        xs,
                        ys,
                        facecolor=color,
                        edgecolor="#94a3b8",
                        alpha=0.25 if kind == "zone" else 0.6,
                    )
                else:
                    ax.plot(xs, ys, color=color, linewidth=width)

            seen_labels: set[str] = set()
            for idx, (label, xs, ys) in enumerate(tracks):
                c = palette_color(palette, idx)
                lbl = label if label not in seen_labels else None
                seen_labels.add(label)
                ax.plot(xs, ys, color=c, linewidth=2.0, label=lbl)
                if len(xs) >= 1 and np.isfinite(xs[0]) and np.isfinite(ys[0]):
                    ax.scatter([xs[0]], [ys[0]], color=c, marker="o", s=30, zorder=5)
                if len(xs) >= 2 and np.isfinite(xs[-1]) and np.isfinite(ys[-1]):
                    ax.scatter([xs[-1]], [ys[-1]], color=c, marker="X", s=45, zorder=5)

            ax.set_aspect("equal", adjustable="datalim")
            if seen_labels:
                ax.legend(loc="best", fontsize=8)
            finish_axes(
                ax,
                title=(
                    f"{self.spec.title} ({world_name})"
                    if self.spec.title and world_name and n_worlds > 1
                    else self.spec.title
                    or (
                        f"Trajectories ({world_name})" if world_name else "Trajectories"
                    )
                ),
                x_label="X [m]",
                y_label="Y [m]",
                grid_axis="both",
            )

        if not rendered_any:
            plt.close(fig)
            return
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)
