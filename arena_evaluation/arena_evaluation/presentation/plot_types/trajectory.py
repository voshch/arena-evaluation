"""Trajectory plot renderer backed by dim_world_geometry."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from shapely.geometry import LineString

from arena_evaluation.presentation.checks import PlotRenderError
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
_GIF_FRAMES: int = 40
_GIF_FPS: int = 6
DISPLAY_TOLERANCE_M: float = 0.01
DISPLAY_DECIMALS: int = 3
_WorldElements = list[tuple[str, str, np.ndarray, np.ndarray]]
_WorldTracks = list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]


def _extract_tracks(df: pl.DataFrame, *, x_col: str | None, y_col: str | None, color_col: str | None) -> _WorldTracks:
    """Split trajectory samples by structural keys, with each track's time in seconds."""
    if not x_col or not y_col:
        return []
    split_cols = list(dict.fromkeys(c for c in ("episode_id", "robot_id", "ped_id", "agent_id", color_col) if c and c in df.columns))

    def one(label: str, sub: pl.DataFrame) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        xs = series_values(sub, x_col)
        ys = series_values(sub, y_col)
        times = np.arange(len(xs), dtype=np.float64)
        if sub.schema.get("t_ns") is not None:
            raw = sub["t_ns"].cast(pl.Float64).to_numpy()
            finite = raw[np.isfinite(raw)]
            if len(finite) and float(np.nanmax(finite) - np.nanmin(finite)) > 0.0:
                times = (raw - np.nanmin(finite)) / 1e9
        order = np.argsort(times, kind="stable")
        return label, xs[order], ys[order], times[order]

    if not split_cols:
        return [one("trajectory", df)]
    tracks: _WorldTracks = []
    for key, sub in df.partition_by(split_cols, as_dict=True, maintain_order=True).items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        if color_col and color_col in split_cols:
            c_val = key_tuple[split_cols.index(color_col)]
            label = "NULL" if c_val is None else str(c_val)
        else:
            label = "/".join("NULL" if k is None else str(k) for k in key_tuple)
        tracks.append(one(label, sub))
    return tracks


def _finite_runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of the consecutive ``True`` runs of ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(np.int8)))
    return [np.arange(start, stop) for start, stop in zip(edges[::2], edges[1::2], strict=True)]


def simplify_track(xs: np.ndarray, ys: np.ndarray, *, tolerance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Douglas-Peucker simplification of one track: every dropped sample lies within ``tolerance_m`` of the line.

    Corners and both ends are kept. Non-finite samples split the track into runs that stay separate
    lines, as they do when drawn.
    """
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for run in _finite_runs(np.isfinite(xs) & np.isfinite(ys)):
        run_xy = np.column_stack([xs[run], ys[run]])
        if len(run) > 2:
            run_xy = np.asarray(LineString(run_xy).simplify(tolerance_m, preserve_topology=False).coords)
        if x_parts:
            x_parts.append(np.array([np.nan]))
            y_parts.append(np.array([np.nan]))
        x_parts.append(run_xy[:, 0])
        y_parts.append(run_xy[:, 1])
    if not x_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(x_parts), np.concatenate(y_parts)


class TrajectoryRenderer(BasePlotRenderer):
    """Renders robot/pedestrian trajectories overlaid on dim_world_geometry."""

    PLOT_TYPE = "trajectory"
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = BasePlotRenderer.SUPPORTED_OPTIONS | {
        "world",
        "x_col",
        "y_col",
        "generate_gifs",
    }
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset({"x_col", "y_col"})

    def _world_elements(self, world_name: str | None) -> _WorldElements:
        if not world_name:
            return []
        geom = self.world_geometry
        usable = geom is not None and not geom.is_empty() and geom.schema.get("world") is not None
        if usable:
            sub = geom.filter(pl.col("world") == world_name)
            if not sub.is_empty():
                ordered = sub.sort(["element_kind", "element_name", "part_idx", "vertex_idx"])
                elements: _WorldElements = []
                for (_, kind, name, _), part in ordered.partition_by(
                    ["world", "element_kind", "element_name", "part_idx"],
                    as_dict=True,
                    maintain_order=True,
                ).items():
                    xs, ys = series_values(part, "x"), series_values(part, "y")
                    if len(xs) >= 2:
                        elements.append((str(kind), str(name), xs, ys))
                return elements
        self.add_notice(f"dim_world_geometry has no rows for world {world_name!r}")
        return []

    def _world_layers(self, df: pl.DataFrame) -> Iterator[tuple[str | None, _WorldElements, _WorldTracks]]:
        """World name, geometry elements and tracks for each world partition."""
        opts = self.spec.options
        x_col = self.resolve_channel("x") or (str(opts["x_col"]) if opts.get("x_col") else None)
        y_col = self.resolve_channel("y") or (str(opts["y_col"]) if opts.get("y_col") else None)
        color_col = self.resolve_channel("color")
        if not x_col or not y_col:
            raise ValueError(f"Plot {self.spec.id!r}: trajectory plot requires channels 'x' and 'y'.")
        opt_world = opts.get("world")
        has_world = "world" in df.columns  # agent-contracts: ignore AC003 - world
        if isinstance(opt_world, str) and opt_world.strip():
            w = opt_world.strip()
            if not has_world:
                partitions: list[tuple[str | None, pl.DataFrame]] = [(w, df)]
            else:
                sub = df.filter(pl.col("world") == w)
                partitions = [(w, sub if not sub.is_empty() else df)]
        elif has_world and not df.is_empty():
            partitions = []
            for key, sub in df.partition_by("world", as_dict=True, maintain_order=True).items():
                w_val = key[0] if isinstance(key, tuple) else key
                partitions.append((str(w_val) if w_val is not None else None, sub))
        else:
            partitions = [(None, df)]
        for world_name, w_df in partitions:
            yield (
                world_name,
                self._world_elements(world_name),
                _extract_tracks(w_df, x_col=x_col, y_col=y_col, color_col=color_col),
            )

    def _trajectory_title(self, world_name: str | None) -> str:
        return self.spec.title or (f"Trajectories ({world_name})" if world_name else "Trajectories")

    def _draw_geometry(self, ax: plt.Axes, elements: _WorldElements) -> None:
        for kind, _name, xs, ys in elements:
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

    @staticmethod
    def _new_world_figure(n_worlds: int) -> tuple[plt.Figure, np.ndarray]:
        return plt.subplots(1, n_worlds, figsize=(9.0 * n_worlds, 8.0), squeeze=False)

    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """Render an interactive Plotly trajectory plot with dim_world_geometry."""
        # differs from overhaul/source:plot_types/trajectory.py:16-18: uses dim_world_geometry; no processing/ imports, no theta*, no raster map
        if df.is_empty():
            return None

        html_parts: list[str] = []
        palette = get_color_palette()
        for world_name, elements, tracks in self._world_layers(df):
            if not tracks:
                continue
            fig = go.Figure()
            for kind, name, xs, ys in elements:
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
            for idx, (label, xs, ys, _times) in enumerate(tracks):
                sx, sy = simplify_track(xs, ys, tolerance_m=DISPLAY_TOLERANCE_M)
                fig.add_trace(
                    go.Scatter(
                        x=np.round(sx, DISPLAY_DECIMALS).tolist(),
                        y=np.round(sy, DISPLAY_DECIMALS).tolist(),
                        mode="lines",
                        name=label,
                        line={"color": palette_color(palette, idx), "width": 2.2},
                        connectgaps=False,
                    )
                )
            fig.update_layout(
                template="plotly_white",
                title=self._trajectory_title(world_name),
                xaxis_title="X [m]",
                yaxis_title="Y [m]",
                yaxis_scaleanchor="x",
                yaxis_scaleratio=1,
                margin=dict(t=55, b=60, l=65, r=30),
            )
            html_parts.append(fig.to_html(include_plotlyjs=False, full_html=False))
        return (html_parts[0] if len(html_parts) == 1 else html_parts) or None

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib trajectory plot with dim_world_geometry."""
        if df.is_empty():
            raise PlotRenderError(f"Plot {self.spec.id!r}: trajectory plot cannot draw an empty frame.")

        layers = list(self._world_layers(df))
        n_worlds = len(layers)
        fig, axes = self._new_world_figure(n_worlds)
        palette = get_color_palette()
        rendered_any = False

        for w_idx, (world_name, elements, tracks) in enumerate(layers):
            ax = axes[0, w_idx]
            if not tracks:
                continue
            rendered_any = True
            self._draw_geometry(ax, elements)
            seen_labels: set[str] = set()
            for idx, (label, xs, ys, _times) in enumerate(tracks):
                c = palette_color(palette, idx)
                lbl = label if label not in seen_labels else None
                seen_labels.add(label)
                ax.plot(xs, ys, color=c, linewidth=2.0, label=lbl)

            ax.set_aspect("equal", adjustable="datalim")
            if seen_labels:
                ax.legend(loc="best", fontsize=8)
            title = f"{self.spec.title} ({world_name})" if self.spec.title and world_name and n_worlds > 1 else self._trajectory_title(world_name)
            finish_axes(
                ax,
                title=title,
                x_label="X [m]",
                y_label="Y [m]",
                grid_axis="both",
            )

        if not rendered_any:
            plt.close(fig)
            raise PlotRenderError(f"Plot {self.spec.id!r}: trajectory frame holds no drawable track.")
        fig.tight_layout()
        self.save_figure(fig, out_path)
        plt.close(fig)

        if self.spec.options.get("generate_gifs"):
            self._write_gif(df, out_path.with_suffix(".gif"))

    def _write_gif(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Replay every trajectory track over its recorded time as an animated GIF."""
        import matplotlib.animation as animation

        layers = list(self._world_layers(df))
        fig, axes = self._new_world_figure(len(layers))
        palette = get_color_palette()
        lines: list[list[plt.Line2D]] = []
        heads: list[list[plt.Line2D]] = []
        end_time = 0.0
        for w_idx, (world_name, elements, tracks) in enumerate(layers):
            ax = axes[0, w_idx]
            self._draw_geometry(ax, elements)
            world_lines: list[plt.Line2D] = []
            world_heads: list[plt.Line2D] = []
            for idx, (_label, _xs, _ys, times) in enumerate(tracks):
                colour = palette_color(palette, idx)
                (line,) = ax.plot([], [], color=colour, linewidth=2.0)
                (head,) = ax.plot([], [], "o", color=colour, markersize=7)
                world_lines.append(line)
                world_heads.append(head)
                if len(times):
                    end_time = max(end_time, float(np.nanmax(times)))
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xlabel="X [m]", ylabel="Y [m]", title=self._trajectory_title(world_name))
            lines.append(world_lines)
            heads.append(world_heads)

        if end_time <= 0.0:
            end_time = float(max((len(t[1]) for _w, _e, tracks in layers for t in tracks), default=0))
        if end_time <= 0.0:
            plt.close(fig)
            return
        frames = np.linspace(0.0, end_time, _GIF_FRAMES + 1)
        clock = fig.text(0.02, 0.02, "", fontsize=11, family="monospace")

        def draw(now: float) -> list[object]:
            for w_idx, (_world, _elements, tracks) in enumerate(layers):
                for track_idx, (_label, xs, ys, times) in enumerate(tracks):
                    count = int(np.searchsorted(times, now, side="right"))
                    lines[w_idx][track_idx].set_data(xs[:count], ys[:count])
                    if count:
                        heads[w_idx][track_idx].set_data([xs[count - 1]], [ys[count - 1]])
                    else:
                        heads[w_idx][track_idx].set_data([], [])
            clock.set_text(f"t = {now:6.1f} s")
            artists = [artist for world_lines, world_heads in zip(lines, heads, strict=True) for artist in (*world_lines, *world_heads)]
            return [*artists, clock]

        animation_writer = animation.PillowWriter(fps=_GIF_FPS)
        gif = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 // _GIF_FPS)
        try:
            gif.save(out_path, writer=animation_writer)
        finally:
            plt.close(fig)
