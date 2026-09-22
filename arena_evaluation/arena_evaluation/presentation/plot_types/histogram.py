from __future__ import annotations

import pathlib

import plotly.express as px
import polars as pl

from .base import BasePlotRenderer


class HistogramRenderer(BasePlotRenderer):
    PLOT_TYPE = "histogram"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        if x_col not in df_filtered.columns:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        keep = [c for c in [x_col, diff_col] if c in df_filtered.columns]
        pdf = df_filtered.select(keep).to_pandas().dropna(subset=[x_col])
        if pdf.empty:
            return None

        import numpy as np
        import pandas as pd

        num_bins = self.spec.options.get("nbins", 15)
        opacity = self.spec.options.get("opacity", 0.6)

        global_min = pdf[x_col].min()
        global_max = pdf[x_col].max()

        if global_min == global_max:
            global_min -= 1
            global_max += 1

        bins = np.linspace(global_min, global_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if diff_col in pdf.columns:
            binned_dfs = []
            for name, group in pdf.groupby(diff_col, observed=False):
                counts, _ = np.histogram(group[x_col], bins=bins)
                df_group = pd.DataFrame({"bin_center": bin_centers, "count": counts, diff_col: name})
                binned_dfs.append(df_group)
            counts_df = pd.concat(binned_dfs)
            color_arg = diff_col
        else:
            counts, _ = np.histogram(pdf[x_col], bins=bins)
            counts_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
            color_arg = None

        fig = px.area(
            counts_df,
            x="bin_center",
            y="count",
            color=color_arg,
            template="plotly_white",
        )

        fig.update_traces(opacity=opacity, line=dict(shape="spline", smoothing=0.8))

        fig.update_layout(xaxis_title=self.format_label(x_col.replace("_", " ").title(), x_col), yaxis_title="Count", legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        if x_col not in df_filtered.columns:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        keep = [c for c in [x_col, diff_col] if c in df_filtered.columns]
        pdf = df_filtered.select(keep).to_pandas().dropna(subset=[x_col])
        if pdf.empty:
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        num_bins = self.spec.options.get("nbins", 15)

        global_min = pdf[x_col].min()
        global_max = pdf[x_col].max()

        if global_min == global_max:
            global_min -= 1
            global_max += 1

        bins = np.linspace(global_min, global_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if diff_col in pdf.columns:
            binned_dfs = []
            for name, group in pdf.groupby(diff_col, observed=False):
                counts, _ = np.histogram(group[x_col], bins=bins)
                df_group = pd.DataFrame({"bin_center": bin_centers, "count": counts, diff_col: name})
                binned_dfs.append(df_group)
            counts_df = pd.concat(binned_dfs)
            hue_arg = diff_col
        else:
            counts, _ = np.histogram(pdf[x_col], bins=bins)
            counts_df = pd.DataFrame({"bin_center": bin_centers, "count": counts})
            hue_arg = None

        plt.figure(figsize=(10, 6))

        if hue_arg:
            for name, group in counts_df.groupby(hue_arg, observed=False):
                plt.plot(group["bin_center"], group["count"], label=name, marker='o', linewidth=2)
                plt.fill_between(group["bin_center"], group["count"], alpha=0.3)
            plt.legend()
        else:
            plt.plot(counts_df["bin_center"], counts_df["count"], marker='o', linewidth=2)
            plt.fill_between(counts_df["bin_center"], counts_df["count"], alpha=0.3)

        plt.title(self.spec.title)
        plt.xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
