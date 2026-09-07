from __future__ import annotations

import pathlib

import plotly.graph_objects as go
import polars as pl

from .base import BasePlotRenderer


class TimeseriesRenderer(BasePlotRenderer):
    PLOT_TYPE = "timeseries"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        import numpy as np
        import plotly.express as px

        df_filtered = self._apply_filters(df)
        diff_col, df_filtered = self.resolve_diff_col(df_filtered)

        metrics = self.spec.options.get("metrics", [])
        if not metrics:
            if self.spec.data_key:
                metrics = [self.spec.data_key]
            else:
                return None

        # Check for matching metric columns or aliases
        valid_metrics = []
        for m in metrics:
            if m in df_filtered.columns:
                valid_metrics.append(m)
            elif f"timeseries_{m}" in df_filtered.columns:
                valid_metrics.append(f"timeseries_{m}")
            elif m.replace("timeseries_", "") in df_filtered.columns:
                valid_metrics.append(m.replace("timeseries_", ""))

        if not valid_metrics:
            return None

        pdf = df_filtered.to_pandas()
        if pdf.empty:
            return None

        # Filter out reference runs from main planner traces
        if "is_reference" in pdf.columns and not self.spec.options.get("include_reference", False):
            pdf = pdf[~pdf["is_reference"].fillna(False)]
            if pdf.empty:
                return None

        x_col = self.spec.options.get("x", "timeseries_time_s")
        if x_col not in df_filtered.columns:
            return None

        fig = go.Figure()
        planners = pdf[diff_col].unique() if diff_col in pdf.columns else ["unknown"]
        colors = px.colors.qualitative.Plotly

        traces_added = 0
        for planner_idx, planner in enumerate(planners):
            planner_df = pdf[pdf[diff_col] == planner] if diff_col in pdf.columns else pdf

            for _, row in planner_df.iterrows():
                episode_val = row.get("episode", "unknown")
                legend_group = f"{planner} - Ep {episode_val}"
                base_color = colors[planner_idx % len(colors)]

                x_raw = row.get(x_col)
                if x_raw is None or isinstance(x_raw, (int, float, str, bool)):
                    continue
                try:
                    x_data = np.array(x_raw, dtype=float)
                except Exception:
                    continue
                if x_data.ndim != 1 or len(x_data) == 0:
                    continue

                for m_idx, metric in enumerate(valid_metrics):
                    y_raw = row.get(metric)
                    if y_raw is None:
                        continue
                    try:
                        y_data = np.array(y_raw, dtype=float)
                    except Exception:
                        continue

                    if y_data.ndim != 1 or len(y_data) == 0:
                        continue

                    # If sizes differ, interpolate or match lengths
                    if len(x_data) != len(y_data):
                        cur_x = np.linspace(x_data[0], x_data[-1], len(y_data)) if len(y_data) > 1 else x_data[: len(y_data)]
                    else:
                        cur_x = x_data

                    dash_styles = ["solid", "dash", "dot", "dashdot"]
                    dash = dash_styles[m_idx % len(dash_styles)]
                    showlegend = True if m_idx == 0 else False

                    m_label = self.format_label(metric.replace("timeseries_", "").replace("_", " ").title(), metric)
                    fig.add_trace(
                        go.Scatter(
                            x=cur_x.tolist(),
                            y=y_data.tolist(),
                            mode="lines",
                            name=f"{legend_group}" if m_idx == 0 else m_label,
                            legendgroup=legend_group,
                            line=dict(color=base_color, dash=dash, width=2),
                            hovertemplate=f"Ep {episode_val}<br>Time: %{{x:.2f}}s<br>{m_label}: %{{y:.3f}}<extra></extra>",
                            showlegend=showlegend,
                        )
                    )
                    traces_added += 1

        fig.update_layout(
            template="plotly_white",
            xaxis_title=self.format_label(x_col.replace("timeseries_", "").replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(valid_metrics[0].replace("timeseries_", "").replace("_", " ").title(), valid_metrics[0]),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=150),
        )

        return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pass
