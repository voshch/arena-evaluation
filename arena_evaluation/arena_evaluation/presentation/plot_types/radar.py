from __future__ import annotations

import pathlib
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from arena_evaluation.presentation.color_utils import get_color_palette
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer


class RadarRenderer(BasePlotRenderer):
    PLOT_TYPE = "radar"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)

        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return None

        metrics = self.spec.options.get(
            "metrics",
            ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"],
        )

        valid_metrics = [
            m for m in metrics
            if m in df_filtered.columns and not df_filtered[m].is_null().all()
        ]

        if len(valid_metrics) < 3:
            return None

        grouped = (
            df_filtered
            .group_by(diff_col)
            .agg([pl.col(m).mean().alias(m) for m in valid_metrics])
            .to_pandas()
        )
        grouped = grouped.fillna(0.0)

        if grouped.empty:
            return None

        normalized = grouped.copy()

        use_log_scale = self.spec.options.get("use_log_scale", False)
        if use_log_scale:
            for m in valid_metrics:
                normalized[m] = np.log1p(normalized[m])

        positive_metrics = {
            "success", "success_rate", "spl", "path_efficiency",
            "relative_throughput", "passing_rule_compliance",
            "velocity_mean", "velocity_max",
        }

        for m in valid_metrics:
            max_val = normalized[m].max()
            min_val = normalized[m].min()

            if max_val == 0 and min_val == 0:
                normalized[m] = 1.0
            else:
                if m in positive_metrics:
                    normalized[m] = normalized[m] / max_val if max_val > 0 else 1.0
                else:
                    if min_val > 0:
                        normalized[m] = min_val / normalized[m]
                    else:
                        normalized[m] = 1.0 - (normalized[m] / max_val)

        fig = go.Figure()

        for _, row in normalized.iterrows():
            values = [row[m] for m in valid_metrics]
            values.append(values[0])
            formatted_metrics = [self.format_label(m.replace("_", " ").title(), m) for m in valid_metrics]
            labels = formatted_metrics + [formatted_metrics[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill=None,
                line=dict(width=3),
                name=row[diff_col],
                opacity=1.0,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:

        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)
        if diff_col not in df_filtered.columns:
            return

        metrics = self.spec.options.get(
            "metrics",
            ["path_efficiency", "time_to_goal", "collision_amount", "roughness_mean", "jerk_mean"],
        )
        valid_metrics = [
            m for m in metrics
            if m in df_filtered.columns and not df_filtered[m].is_null().all()
        ]
        if len(valid_metrics) < 3:
            return

        grouped = (
            df_filtered
            .group_by(diff_col)
            .agg([pl.col(m).mean().alias(m) for m in valid_metrics])
            .to_pandas()
        )
        grouped = grouped.fillna(0.0)
        if grouped.empty:
            return

        normalized = grouped.copy()
        use_log_scale = self.spec.options.get("use_log_scale", False)
        if use_log_scale:
            for m in valid_metrics:
                normalized[m] = np.log1p(normalized[m])

        positive_metrics = {
            "success", "success_rate", "spl", "path_efficiency",
            "relative_throughput", "passing_rule_compliance",
            "velocity_mean", "velocity_max",
        }
        for m in valid_metrics:
            max_val = normalized[m].max()
            min_val = normalized[m].min()
            if max_val == 0 and min_val == 0:
                normalized[m] = 1.0
            else:
                if m in positive_metrics:
                    normalized[m] = normalized[m] / max_val if max_val > 0 else 1.0
                else:
                    if min_val > 0:
                        normalized[m] = min_val / normalized[m]
                    else:
                        normalized[m] = 1.0 - (normalized[m] / max_val)


        num_vars = len(valid_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        formatted_labels = [self.format_label(m.replace("_", " ").title(), m) for m in valid_metrics]

        palette = get_color_palette()
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for idx, (_, row) in enumerate(normalized.iterrows()):
            values = [row[m] for m in valid_metrics]
            values += values[:1]
            color = palette[idx % len(palette)]
            ax.plot(angles, values, linewidth=2, label=row[diff_col], color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(formatted_labels, size=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=7, color="grey")

        ax.set_title(self.spec.title, size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
