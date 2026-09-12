from __future__ import annotations

import math
import pathlib

import plotly.express as px
import polars as pl

from .base import BasePlotRenderer


class BarRenderer(BasePlotRenderer):
    PLOT_TYPE = "bar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        df_filtered = self._explode_needed(df_filtered, diff_col)

        is_stacked = self.spec.options.get("stacked", False)

        if is_stacked:
            metrics = self.spec.options.get("metrics", [])
            if not metrics:
                return None

            # Aggregate the sum of each metric per planner, divide by count (mean) or use absolute sum
            agg_exprs = [pl.col(m).mean().alias(m) for m in metrics if m in df_filtered.columns]
            if not agg_exprs:
                return None

            grouped = df_filtered.group_by(diff_col).agg(agg_exprs).to_pandas()
            if grouped.empty:
                return None

            # Normalize to 100%
            grouped[metrics] = grouped[metrics].div(grouped[metrics].sum(axis=1), axis=0) * 100

            fig = px.bar(grouped, x=diff_col, y=metrics, template="plotly_white", barmode="stack", labels={"value": "Percentage (%)", "variable": "Component", diff_col: diff_col.lstrip("_").replace("_", " ").title()})
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
            return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

        else:
            if self.spec.data_key not in df_filtered.columns:
                return None
            grouped = self._grouped_with_error(df_filtered, diff_col).to_pandas()

            if grouped.empty:
                return None

            fig = px.bar(grouped, x=diff_col, y="mean", color=diff_col, error_y="err_plus", error_y_minus="err_minus", template="plotly_white", labels={"mean": self.format_label(self.spec.data_key.replace("_", " ").title(), self.spec.data_key), diff_col: diff_col.lstrip("_").replace("_", " ").title()})
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))

            return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    @staticmethod
    def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson score interval of a proportion."""
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
        return center - half, center + half

    def _grouped_with_error(self, df_filtered: pl.DataFrame, diff_col: str) -> pl.DataFrame:
        """Per-group mean with asymmetric error columns: Wilson for boolean data, one std otherwise."""
        key = self.spec.data_key
        if df_filtered.schema[key] == pl.Boolean:
            grouped = df_filtered.group_by(diff_col).agg(
                [
                    pl.col(key).cast(pl.Float64).mean().alias("mean"),
                    pl.col(key).cast(pl.Int64).sum().alias("k"),
                    pl.col(key).count().alias("n"),
                ]
            )
            bounds = [self._wilson(int(k), int(n)) for k, n in zip(grouped["k"].to_list(), grouped["n"].to_list(), strict=True)]
            means = grouped["mean"].to_list()
            return grouped.with_columns(
                [
                    pl.Series("err_plus", [hi - m for (_, hi), m in zip(bounds, means, strict=True)]),
                    pl.Series("err_minus", [m - lo for (lo, _), m in zip(bounds, means, strict=True)]),
                ]
            ).drop(["k", "n"])
        grouped = df_filtered.group_by(diff_col).agg(
            [
                pl.col(key).mean().alias("mean"),
                pl.col(key).std().fill_null(0.0).alias("std"),
            ]
        )
        return grouped.with_columns([pl.col("std").alias("err_plus"), pl.col("std").alias("err_minus")]).drop("std")

    def _explode_needed(self, df_filtered: pl.DataFrame, diff_col: str) -> pl.DataFrame:
        """Explode per-sample list columns (timeseries metrics) before
        aggregation. Selects only the needed columns first - the full frame
        carries other list columns of differing lengths whose explosion grows
        combinatorially (polars 1.x, see the line renderer note)."""
        metrics = self.spec.options.get("metrics", [])
        keep = [c for c in [self.spec.data_key, diff_col, *metrics] if c in df_filtered.columns]
        list_cols = [c for c in keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            return df_filtered.select(keep).explode(list_cols)
        return df_filtered

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        df_filtered = self._explode_needed(df_filtered, diff_col)

        keep = [c for c in [self.spec.data_key, diff_col] if c in df_filtered.columns]
        pdf = df_filtered.select(keep).to_pandas()
        if pdf.empty:
            return

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        if df_filtered.schema[self.spec.data_key] == pl.Boolean:
            grouped = self._grouped_with_error(df_filtered, diff_col).to_pandas()
            sns.barplot(data=grouped, x=diff_col, y="mean", hue=diff_col, errorbar=None, legend=False)
            plt.errorbar(range(len(grouped)), grouped["mean"], yerr=[grouped["err_minus"], grouped["err_plus"]], fmt="none", ecolor="black", capsize=4)
        else:
            sns.barplot(
                data=pdf,
                x=diff_col,
                y=self.spec.data_key,
                hue=diff_col,
                errorbar="sd",
                legend=False,
            )
        plt.title(self.spec.title)
        plt.ylabel(self.format_label(self.spec.data_key.replace("_", " ").title(), self.spec.data_key))
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
