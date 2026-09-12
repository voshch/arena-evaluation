from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl

from .base import BasePlotRenderer


class ViolinRenderer(BasePlotRenderer):
    PLOT_TYPE = "violin"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        keep = [self.spec.data_key, diff_col]
        pdf = df_filtered.select(keep).to_pandas()
        if pdf.empty:
            return None

        fig = px.violin(
            pdf,
            y=self.spec.data_key,
            color=diff_col,
            box=True,
            points="all",
            labels={
                self.spec.data_key: self.format_label(self.spec.data_key.replace("_", " ").title(), self.spec.data_key),
                diff_col: diff_col.lstrip("_").replace("_", " ").title(),
            },
            template="plotly_white",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        if self.spec.data_key not in df_filtered.columns:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        keep = [self.spec.data_key, diff_col]
        pdf = df_filtered.select(keep).to_pandas()
        if pdf.empty:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=pdf,
            y=self.spec.data_key,
            hue=diff_col,
            inner="box",
        )
        plt.title(self.spec.title)
        plt.ylabel(self.format_label(self.spec.data_key.replace("_", " ").title(), self.spec.data_key))
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
