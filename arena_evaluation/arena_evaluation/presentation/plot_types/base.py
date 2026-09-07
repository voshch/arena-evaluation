from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl
import pathlib

from arena_evaluation.presentation.dimension_detector import resolve_differentiate
from arena_evaluation.storage.schemas import PlotSpec


class BasePlotRenderer(ABC):
    """Base class for all plot renderers."""

    PLOT_TYPE: str = ""

    def __init__(self, spec: PlotSpec, units: dict[str, str] | None = None):
        self.spec = spec
        self.units = units or {}
        self.run_dir: pathlib.Path | None = None

    def format_label(self, label: str, data_key: str) -> str:
        """Format the label with the unit associated with data_key."""
        unit = self.units.get(data_key)
        if unit:
            return f"{label} [{unit}]"
        return label

    @abstractmethod
    def render_plotly(self, df: pl.DataFrame) -> str | list[str] | None:
        """Render an interactive Plotly plot as an HTML string, or a list of them."""
        pass

    @abstractmethod
    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        """Render a static Seaborn/Matplotlib plot, saving the PNG to out_path."""
        pass

    def resolve_diff_col(self, df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
        return resolve_differentiate(self.spec, df)

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply scalar and sequence filter predicates from PlotSpec."""
        if not self.spec.filter:
            return df

        res_df = df
        for k, v in self.spec.filter.items():
            if k not in res_df.columns:
                continue
            if res_df.schema[k] == pl.List:
                if isinstance(v, (list, tuple, set)):
                    v_list = list(v)
                    if v_list:
                        res_df = res_df.filter(
                            pl.any_horizontal([pl.col(k).list.contains(x) for x in v_list])
                        )
                    else:
                        res_df = res_df.filter(pl.lit(False))
                else:
                    res_df = res_df.filter(pl.col(k).list.contains(v))
            elif isinstance(v, (list, tuple, set)):
                res_df = res_df.filter(pl.col(k).is_in(list(v)))
            else:
                res_df = res_df.filter(pl.col(k) == v)
        return res_df

    def _apply_row_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters to scalar/exploded columns after list expansion."""
        if not self.spec.filter:
            return df

        res_df = df
        for k, v in self.spec.filter.items():
            if k in res_df.columns and res_df.schema[k] != pl.List:
                if isinstance(v, (list, tuple, set)):
                    res_df = res_df.filter(pl.col(k).is_in(list(v)))
                else:
                    res_df = res_df.filter(pl.col(k) == v)
        return res_df
