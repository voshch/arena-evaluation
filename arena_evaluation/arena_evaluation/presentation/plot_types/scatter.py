from __future__ import annotations

import pathlib
import polars as pl
import plotly.express as px

from .base import BasePlotRenderer


class ScatterRenderer(BasePlotRenderer):
    PLOT_TYPE = "scatter"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        y_col = self.spec.options.get("y")

        if not y_col or x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            return None

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        filter_keys = [k for k in (self.spec.filter or {}) if k in df_filtered.columns]
        keep = list(dict.fromkeys([c for c in [x_col, y_col, diff_col, *filter_keys] if c in df_filtered.columns]))
        list_cols = [c for c in keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            df_filtered = df_filtered.select(keep).explode(list_cols)
        else:
            df_filtered = df_filtered.select(keep)
        df_filtered = self._apply_row_filters(df_filtered)

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        max_points = int(self.spec.options.get("max_points_per_trace", 5000))
        if max_points > 0 and len(pdf) > max_points:
            stride = max(1, len(pdf) // max_points)
            pdf = pdf.iloc[::stride]

        fig = px.scatter(
            pdf,
            x=x_col,
            y=y_col,
            color=diff_col,
            template="plotly_white",
            opacity=0.7,
        )

        fig.update_layout(
            xaxis_title=self.format_label(x_col.replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(y_col.replace("_", " ").title(), y_col),
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)

        x_col = self.spec.data_key
        y_col = self.spec.options.get("y")

        if not y_col or x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            return

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        keep = [c for c in [x_col, y_col, diff_col] if c in df_filtered.columns]
        list_cols = [c for c in keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            df_filtered = df_filtered.select(keep).explode(list_cols)
        else:
            df_filtered = df_filtered.select(keep)
        df_filtered = self._apply_row_filters(df_filtered)

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=pdf,
            x=x_col,
            y=y_col,
            hue=diff_col,
            alpha=0.7,
        )
        plt.title(self.spec.title)
        plt.xlabel(self.format_label(x_col.replace("_", " ").title(), x_col))
        plt.ylabel(self.format_label(y_col.replace("_", " ").title(), y_col))
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

