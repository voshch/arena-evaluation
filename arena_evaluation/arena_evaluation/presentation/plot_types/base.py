"""Base class and shared helpers for all plot renderers."""

from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from arena_evaluation.presentation.checks import PlotRenderError, null_reason_of
from arena_evaluation.presentation.viz_manifest import (
    PlotSpec,  # differs from overhaul/source:plot_types/base.py:13: PlotSpec lives in viz_manifest
)

if TYPE_CHECKING:
    import matplotlib.figure

    from arena_evaluation.presentation.checks import PlotDiagnostics

__all__ = [
    "BasePlotRenderer",
    "finish_axes",
    "label_values",
    "new_figure",
    "palette_color",
    "require_drawable",
    "series_values",
]


def series_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """Float64 values of one column, NaN where the value is null, length preserved."""
    return frame[column].cast(pl.Float64).to_numpy()


def label_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    """String labels of one column, mapping NULL to its ``<column>_null_reason`` when present."""
    reason_col = f"{column}_null_reason"
    if (
        reason_col in frame.columns
    ):  # agent-contracts: ignore AC003 - optional companion null_reason column
        return (
            frame.select(
                pl.coalesce(
                    [
                        pl.col(column).cast(pl.Utf8),
                        pl.col(reason_col).cast(pl.Utf8),
                        pl.lit("NULL"),
                    ]
                )
            )
            .to_series()
            .to_numpy()
        )
    return frame[column].cast(pl.Utf8).fill_null("NULL").to_numpy()


def new_figure(
    *, width_in: float = 10.0, height_in: float = 6.0
) -> tuple[plt.Figure, plt.Axes]:
    """A fresh Matplotlib figure and its single axes."""
    return plt.subplots(figsize=(width_in, height_in))


def finish_axes(
    ax: plt.Axes, *, title: str, x_label: str, y_label: str, grid_axis: str = "y"
) -> None:
    """Apply the shared static-plot style and axis labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.5, axis=grid_axis)
    ax.set_title(title, pad=12, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def palette_color(palette: Sequence[str], index: int) -> str:
    """The palette color at ``index``, cycling through the palette."""
    return palette[index % len(palette)]


def require_drawable(
    df: pl.DataFrame, column: str, *, plot_id: str, plot_type: str
) -> None:
    """Raise ``PlotRenderError`` unless ``column`` holds at least one value to draw."""
    if df.is_empty():
        raise PlotRenderError(
            f"Plot {plot_id!r}: {plot_type} plot cannot draw an empty frame."
        )
    if column not in df.columns:
        raise PlotRenderError(
            f"Plot {plot_id!r}: {plot_type} plot requires column {column!r}."
        )
    if df[column].null_count() == df.height:
        reason = null_reason_of(df, column)
        suffix = f" (reason: {reason})" if reason else ""
        raise PlotRenderError(
            f"Plot {plot_id!r}: column {column!r} is 100% NULL, nothing to draw{suffix}."
        )


class BasePlotRenderer(ABC):
    """Base class for all plot renderers."""

    PLOT_TYPE: str = ""
    SUPPORTED_OPTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "note",
            "notes_key",
            "vector_svg",
        }
    )
    COLUMN_OPTIONS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, spec: PlotSpec, units: dict[str, str] | None = None) -> None:
        self.spec = spec
        self.units = units or {}
        self.run_dir: pathlib.Path | None = None
        self.world_geometry: pl.DataFrame | None = None
        self.diagnostics: PlotDiagnostics | None = None

    def validate_frame(self, df: pl.DataFrame) -> None:
        """Fail explicitly on unknown options or declared channels/columns missing from df."""
        for key in self.spec.options:
            if key not in self.SUPPORTED_OPTIONS:
                raise ValueError(
                    f"Plot {self.spec.id!r}: unknown option {key!r} for plot type {self.PLOT_TYPE or self.spec.type!r}."
                )
        cols = set(df.columns)
        ch = self.spec.channels
        channel_fields = (
            ("x", ch.x),
            ("y", ch.y),
            ("color", ch.color),
            ("facet_col", ch.facet_col),
            ("facet_row", ch.facet_row),
            ("size", ch.size),
            ("symbol", ch.symbol),
            ("hierarchy", ch.hierarchy),
        )
        for ch_name, raw in channel_fields:
            if raw is None:
                continue
            names = [raw] if isinstance(raw, str) else list(raw)
            for col in names:
                if isinstance(col, str) and col.strip() and col.strip() not in cols:
                    raise ValueError(
                        f"Plot {self.spec.id!r}: channel {ch_name!r} references missing column {col!r}."
                    )
        for opt_name in self.COLUMN_OPTIONS:
            val = self.spec.options.get(opt_name)
            if isinstance(val, str) and val.strip() and val.strip() not in cols:
                raise ValueError(
                    f"Plot {self.spec.id!r}: option {opt_name!r} references missing column {val!r}."
                )

    def validate_unique_cells(self, df: pl.DataFrame, key_cols: Sequence[str]) -> None:
        """Fail explicitly if df has duplicate rows for one drawn cell key."""
        if not key_cols:
            return
        missing = [k for k in key_cols if k not in df.columns]
        if missing:
            raise ValueError(
                f"Plot {self.spec.id!r}: cell key columns missing from frame: {missing!r}."
            )
        keys = list(key_cols)
        dups = df.filter(df.select(keys).is_duplicated())
        if not dups.is_empty():
            sample = {k: dups[k][0] for k in keys}
            raise ValueError(
                f"Plot {self.spec.id!r}: duplicate rows for cell key {tuple(keys)!r} "
                f"({sample!r}); bind an aggregation instead of raw rows: declare "
                "data.computation: group_summary (or correlation_matrix / radar_profile / "
                "distribution_bins) with params.metrics + params.group_cols, or source a "
                "pre-aggregated view such as v_planner_scorecard."
            )

    def add_notice(self, message: str) -> None:
        """Record a non-fatal rendering notice on attached diagnostics, if present."""
        if self.diagnostics is not None and message not in self.diagnostics.issues:
            self.diagnostics.issues.append(message)

    def format_label(self, label: str, data_key: str) -> str:
        """Format the label with the unit associated with data_key."""
        unit = self.units.get(data_key)
        if unit:
            return f"{label} [{unit}]"
        return label

    def resolve_channel(self, channel: str, fallback: str | None = None) -> str | None:
        """Resolve a channel string name (e.g. 'x', 'y', 'color', 'facet_col', 'size', 'symbol')."""
        ch = self.spec.channels
        val: str | list[str] | None = None
        match channel:
            case "x":
                val = ch.x
            case "y":
                val = ch.y
            case "color" | "hue":
                val = ch.color
            case "facet_col":
                val = ch.facet_col
            case "facet_row":
                val = ch.facet_row
            case "size":
                val = ch.size
            case "symbol":
                val = ch.symbol
            case "hierarchy":
                val = ch.hierarchy
            case _:
                val = None

        if val is not None:
            if isinstance(val, str) and val.strip():
                return val.strip()
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], str)
                and val[0].strip()
            ):
                return val[0].strip()
        return fallback

    def resolve_channel_list(
        self, channel: str, fallback: list[str] | str | None = None
    ) -> list[str]:
        """Resolve a channel that can be a list of columns (e.g. multi-metric y)."""
        ch = self.spec.channels
        val: str | list[str] | None = None
        match channel:
            case "x":
                val = ch.x
            case "y":
                val = ch.y
            case "color" | "hue":
                val = ch.color
            case "hierarchy":
                val = ch.hierarchy
            case _:
                val = None

        if isinstance(val, list):
            return [str(v).strip() for v in val if v]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        if fallback is not None:
            return (
                [fallback.strip()]
                if isinstance(fallback, str)
                else [str(f).strip() for f in fallback if f]
            )
        return []

    @staticmethod
    def save_figure(
        fig: matplotlib.figure.Figure, out_path: pathlib.Path, dpi: int = 300
    ) -> None:
        """Save matplotlib figure to vector (PDF/SVG) or high-res image (PNG)."""
        out_path = pathlib.Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    @abstractmethod
    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """Render an interactive Plotly plot as an HTML string, or a list of them."""

    @abstractmethod
    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Matplotlib/Seaborn plot to ``out_path`` as PNG, SVG or PDF."""
