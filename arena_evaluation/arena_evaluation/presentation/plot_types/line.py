"""Long-format line chart renderer supporting grouped traces and error bands."""

from __future__ import annotations

import pathlib
import typing

import polars as pl

from arena_evaluation.presentation.plot_types.base import BasePlotRenderer

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class LineRenderer(BasePlotRenderer):
    PLOT_TYPE = "line"

    def _prepare(self, df: pl.DataFrame) -> tuple[pd.DataFrame, str, str, list[str], str | None, dict] | None:
        """Validate and return (pdf, x_col, y_col, group_cols, error_col, opts) or None."""
        df_filtered = self._apply_filters(df)
        x_col = self.spec.data_key
        y_col = self.spec.options.get("y")
        if not y_col or x_col not in df_filtered.columns or y_col not in df_filtered.columns:
            return None

        opts = self.spec.options or {}
        group_by = self.spec.group_by
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [g for g in (group_by or []) if g in df_filtered.columns]
        if not group_cols:
            diff_col, df_filtered = self.resolve_diff_col(df_filtered)
            if diff_col in df_filtered.columns:
                group_cols = [diff_col]

        error_col = opts.get("error_y")
        if error_col and error_col not in df_filtered.columns:
            error_col = None

        filter_keys = [k for k in (self.spec.filter or {}) if k in df_filtered.columns]
        keep = list(dict.fromkeys([c for c in [x_col, y_col, *group_cols, error_col, *filter_keys] if c]))
        list_cols = [c for c in keep if df_filtered.schema[c] == pl.List]
        if list_cols:
            df_filtered = df_filtered.select(keep).explode(list_cols)
        df_filtered = self._apply_row_filters(df_filtered)

        bin_x = opts.get("bin_x")
        if bin_x is not None and float(bin_x) > 0 and x_col in df_filtered.columns:
            bx = float(bin_x)
            df_filtered = df_filtered.with_columns(((pl.col(x_col) / bx).round() * bx).round(4).alias(x_col))

        trim_p = float(opts.get("trim_percentile", 0.0) or (1.0 - float(opts.get("inliers", 1.0))))
        if 0.0 < trim_p < 0.5 and len(df_filtered) > 0 and y_col in df_filtered.columns:
            group_keys = [c for c in [x_col, *group_cols] if c in df_filtered.columns]
            p_low = trim_p / 2.0
            p_high = 1.0 - (trim_p / 2.0)
            if group_keys:
                df_filtered = df_filtered.filter((pl.col(y_col) >= pl.col(y_col).quantile(p_low).over(group_keys)) & (pl.col(y_col) <= pl.col(y_col).quantile(p_high).over(group_keys)))

        if opts.get("aggregate") and len(df_filtered) > 0:
            agg_cols = [x_col, *group_cols]
            reduce_ = opts.get("reduce", "mean")
            if reduce_ == "leq":
                df_filtered = df_filtered.group_by(agg_cols).agg((10.0 * (10.0 ** (pl.col(y_col) / 10.0)).mean().log10()).alias(y_col))
            elif reduce_ == "max":
                df_filtered = df_filtered.group_by(agg_cols).agg(pl.col(y_col).max().alias(y_col))
            else:
                error_mode = str(opts.get("error_mode", "std")).lower()
                if error_mode in ("ci95", "ci_95", "ci"):
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        (1.96 * pl.col(y_col).std() / pl.col(y_col).count().sqrt()).fill_null(0.0).alias("__std__"),
                    )
                elif error_mode == "sem":
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        (pl.col(y_col).std() / pl.col(y_col).count().sqrt()).fill_null(0.0).alias("__std__"),
                    )
                else:
                    df_filtered = df_filtered.group_by(agg_cols).agg(
                        pl.col(y_col).mean().alias(y_col),
                        pl.col(y_col).std().alias("__std__"),
                    )
                error_col = "__std__"

        pdf = df_filtered.select(pl.col(x_col), pl.col(y_col), *[pl.col(g) for g in group_cols], *([pl.col(error_col)] if error_col else [])).to_pandas()
        if pdf.empty:
            return None

        return pdf, x_col, y_col, group_cols, error_col, opts

    @staticmethod
    def _trace_data(pdf: pd.DataFrame, x_col: str, y_col: str, error_col: str | None, opts: dict, time_to_s: bool, time_relative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Downsample + transform one trace's frame -> (x, y, err) arrays."""
        import numpy as np

        max_points = int(opts.get("max_points_per_trace", 5000))
        if len(pdf) > max_points and max_points > 0:
            stride = max(1, len(pdf) // max_points)
            pdf = pdf.iloc[::stride]

        x = pdf[x_col].astype(float).to_numpy()
        y = pdf[y_col].astype(float).to_numpy()
        err = pdf[error_col].astype(float).to_numpy() if error_col else None

        if time_to_s:
            x = x / 1e9
        if time_relative and len(x) > 0:
            x = x - x.min()

        order = x.argsort()
        x, y = x[order], y[order]
        if err is not None:
            err = err[order]

        if opts.get("smooth") and len(y) >= 3:
            w = int(opts.get("smooth_window", 3))
            w = max(2, min(w, len(y)))
            kernel = np.ones(w) / w
            pad_left = w // 2
            pad_right = w - 1 - pad_left
            padded_y = np.pad(y, (pad_left, pad_right), mode="edge")
            y = np.convolve(padded_y, kernel, mode="valid")
            if err is not None:
                padded_err = np.pad(err, (pad_left, pad_right), mode="edge")
                err = np.convolve(padded_err, kernel, mode="valid")

        return x, y, err

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        import numpy as np
        import plotly.graph_objects as go

        prepared = self._prepare(df)
        if prepared is None:
            return None
        pdf, x_col, y_col, group_cols, error_col, opts = prepared

        error_style = opts.get("error_style", "band")
        mode = opts.get("mode", "lines")
        time_to_s = bool(opts.get("time_to_s", False))
        time_relative = bool(opts.get("time_relative", False))
        max_traces = int(opts.get("max_traces", 40))

        error_mode = str(opts.get("error_mode", "std")).lower()
        if error_mode in ("ci95", "ci_95", "ci"):
            band_suffix = "± 95% CI"
        elif opts.get("smooth"):
            band_suffix = "± 1σ (smoothed)"
        else:
            band_suffix = "± 1σ"

        fig = go.Figure()
        colors = _color_cycle()

        keys = pdf[group_cols].drop_duplicates().head(max_traces).to_dict("records") if group_cols else [{}]
        for t_idx, key in enumerate(keys):
            if group_cols:
                mask = pdf[group_cols].eq(pd_series(key)).all(axis=1)
                trace_pdf = pdf[mask]
            else:
                trace_pdf = pdf
            if trace_pdf.empty:
                continue

            x, y, err = self._trace_data(trace_pdf, x_col, y_col, error_col, opts, time_to_s, time_relative)
            if len(x) == 0:
                continue

            label = " / ".join(str(key.get(g, "")) for g in group_cols) if group_cols else y_col
            color = colors[t_idx % len(colors)]

            if error_col is not None and err is not None:
                err = np.nan_to_num(err, nan=0.0)
                is_non_neg = np.all(y >= 0)
                if error_style == "bars":
                    arrayminus = np.minimum(err, y) if is_non_neg else err
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode=mode,
                            name=label,
                            line=dict(color=color),
                            error_y=dict(type="data", array=err, arrayminus=arrayminus, symmetric=False, visible=True),
                        )
                    )
                else:
                    y_low = np.maximum(0.0, y - err) if is_non_neg else y - err
                    y_high = y + err
                    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=label, line=dict(color=color)))
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([x, x[::-1]]),
                            y=np.concatenate([y_high, y_low[::-1]]),
                            fill="toself",
                            fillcolor=_with_alpha(color, 0.2),
                            line=dict(width=0),
                            name=f"{label} {band_suffix}",
                            legendgroup=label,
                            showlegend=False,
                        )
                    )
            else:
                fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=label, line=dict(color=color)))

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Time [s]" if time_to_s else self.format_label(x_col.replace("_", " ").title(), x_col),
            yaxis_title=self.format_label(y_col.replace("_", " ").title(), y_col),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=30),
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        prepared = self._prepare(df)
        if prepared is None:
            return
        pdf, x_col, y_col, group_cols, error_col, opts = prepared

        error_style = opts.get("error_style", "band")
        mode = opts.get("mode", "lines")
        time_to_s = bool(opts.get("time_to_s", False))
        time_relative = bool(opts.get("time_relative", False))
        max_traces = int(opts.get("max_traces", 40))

        colors = _color_cycle()
        keys = pdf[group_cols].drop_duplicates().head(max_traces).to_dict("records") if group_cols else [{}]

        plt.figure(figsize=(10, 6))
        for t_idx, key in enumerate(keys):
            if group_cols:
                mask = pdf[group_cols].eq(pd_series(key)).all(axis=1)
                trace_pdf = pdf[mask]
            else:
                trace_pdf = pdf
            if trace_pdf.empty:
                continue

            x, y, err = self._trace_data(trace_pdf, x_col, y_col, error_col, opts, time_to_s, time_relative)
            if len(x) == 0:
                continue

            label = " / ".join(str(key.get(g, "")) for g in group_cols) if group_cols else y_col
            color = colors[t_idx % len(colors)]
            marker = "o" if "markers" in mode else None

            if error_col is not None and err is not None:
                err = np.nan_to_num(err, nan=0.0)
                is_non_neg = np.all(y >= 0)
                if error_style == "bars":
                    yerr = [np.minimum(err, y), err] if is_non_neg else err
                    plt.errorbar(x, y, yerr=yerr, label=label, color=color, marker=marker, capsize=2)
                else:
                    y_low = np.maximum(0.0, y - err) if is_non_neg else y - err
                    plt.plot(x, y, label=label, color=color, marker=marker)
                    plt.fill_between(x, y_low, y + err, color=color, alpha=0.2)
            else:
                plt.plot(x, y, label=label, color=color, marker=marker)

        plt.title(self.spec.title)
        plt.xlabel("Time [s]" if time_to_s else self.format_label(x_col.replace("_", " ").title(), x_col))
        plt.ylabel(self.format_label(y_col.replace("_", " ").title(), y_col))
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


def _color_cycle() -> list[str]:
    try:
        from arena_evaluation.presentation.color_utils import get_color_palette

        palette = get_color_palette()
        if palette:
            return list(palette)
    except Exception:
        pass
    return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _with_alpha(hex_color: str, alpha: float) -> str:
    try:
        r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(31,119,180,{alpha})"


def pd_series(d: dict) -> pd.Series:
    """Build a pandas Series for row-masking (import lazily)."""
    import pandas as pd

    return pd.Series(d)
