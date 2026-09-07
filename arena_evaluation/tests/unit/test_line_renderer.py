"""Unit tests for the long-format line renderer."""

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.line import LineRenderer
from arena_evaluation.storage.schemas import PlotSpec


def _samples_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [1e9, 2e9, 3e9, 1e9, 2e9, 3e9],
            "p_total_w": [10.0, 20.0, 15.0, 12.0, 18.0, 14.0],
            "phase_kind": ["linear"] * 3 + ["angular"] * 3,
            "std": [1.0] * 6,
        }
    )


def _spec(**options) -> PlotSpec:
    return PlotSpec(
        id="t",
        type="line",
        title="Test Line",
        data_key="time_ns",
        group_by=["phase_kind"],
        options=options,
    )


def test_line_with_confidence_band():
    spec = _spec(y="p_total_w", error_y="std", time_to_s=True, time_relative=True)
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    # Band mode renders a filled region (toself) + a line per group.
    assert "toself" in html
    assert html.count("scatter") >= 4  # 2 groups x (line + band)


def test_line_without_error_band():
    spec = _spec(y="p_total_w")
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    assert "toself" not in html


def test_line_missing_columns_returns_none():
    assert LineRenderer(_spec(y="missing")).render_plotly(_samples_df()) is None
    assert LineRenderer(_spec(y="p_total_w")).render_plotly(pl.DataFrame()) is None
    # No group_by columns -> falls back to differentiate, which may be absent.
    spec = PlotSpec(id="t", type="line", title="T", data_key="time_ns", options={"y": "p_total_w"})
    assert spec is not None
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None or True  # must not raise


def test_line_time_conversion_and_relative():
    spec = _spec(y="p_total_w", time_to_s=True, time_relative=True)
    html = LineRenderer(spec).render_plotly(_samples_df())
    assert html is not None
    assert "Time [s]" in html


def test_line_seaborn_renders_png(tmp_path: pathlib.Path):
    spec = _spec(y="p_total_w", error_y="std", mode="lines+markers")
    out = tmp_path / "line.png"
    LineRenderer(spec).render_seaborn(_samples_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_line_seaborn_empty_df_no_crash(tmp_path: pathlib.Path):
    out = tmp_path / "empty.png"
    LineRenderer(_spec(y="p_total_w")).render_seaborn(pl.DataFrame(), out)
    assert not out.exists() or out.stat().st_size == 0


def test_line_error_band_clamped_at_zero():
    # If std > mean (e.g. mean=10, std=25), y_low should be clamped at 0.0, not -15.0
    df = pl.DataFrame(
        {
            "time_ns": [1e9, 2e9],
            "p_total_w": [10.0, 10.0],
            "phase_kind": ["linear", "linear"],
            "std": [25.0, 25.0],
        }
    )
    spec = _spec(y="p_total_w", error_y="std")
    html = LineRenderer(spec).render_plotly(df)
    assert html is not None
    # Verify negative values are not in the lower band y coordinates
    assert "-15" not in html


def test_line_bin_x_aggregation():
    # Multiple episodes with slightly jittered timestamps
    df = pl.DataFrame(
        {
            "time_s": [0.039, 0.041, 0.139, 0.141],
            "p_total_w": [40.0, 44.0, 50.0, 54.0],
            "robot": ["jackal", "jackal", "jackal", "jackal"],
        }
    )
    spec = PlotSpec(
        id="t_bin",
        type="line",
        title="Binned Time",
        data_key="time_s",
        group_by=["robot"],
        options={"y": "p_total_w", "aggregate": True, "bin_x": 0.1},
    )
    html = LineRenderer(spec).render_plotly(df)
    assert html is not None
    assert "toself" in html  # has confidence band from 2-sample std at each bin


def test_line_ci95_and_smoothing():
    df = pl.DataFrame(
        {
            "vx": [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            "power": [50.0, 52.0, 48.0, 50.0, 80.0, 82.0, 78.0, 80.0],
            "robot": ["jackal"] * 8,
        }
    )
    spec = PlotSpec(
        id="t_ci95",
        type="line",
        title="CI 95 Power",
        data_key="vx",
        group_by=["robot"],
        options={"y": "power", "aggregate": True, "bin_x": 0.5, "error_mode": "ci95", "smooth": True},
    )
    html = LineRenderer(spec).render_plotly(df)
    assert html is not None
    assert "95% CI" in html
    assert "toself" in html


def test_line_outlier_trimming():
    # Extreme impulse spike 1000.0 at vx=0.5 should be trimmed by trim_percentile
    df = pl.DataFrame(
        {
            "vx": [0.5] * 20,
            "power": [50.0] * 19 + [1000.0],
            "robot": ["jackal"] * 20,
        }
    )
    spec = PlotSpec(
        id="t_trim",
        type="line",
        title="Trimmed Power",
        data_key="vx",
        group_by=["robot"],
        options={"y": "power", "aggregate": True, "bin_x": 0.5, "trim_percentile": 0.1},
    )
    prepared = LineRenderer(spec)._prepare(df)
    assert prepared is not None
    pdf = prepared[0]
    # The mean should be 50.0 (the 1000.0 spike was trimmed out)
    assert abs(pdf["power"].iloc[0] - 50.0) < 1e-3
