"""Unit tests for radar charts."""

import json

import pandas as pd
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.radar import RadarRenderer, _normalize
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_DEFAULT_METRICS = [
    "path_efficiency",
    "time_to_goal",
    "collision_amount",
    "roughness_mean",
    "jerk_mean",
]


def _radar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "success": [0.8, 1.0],
            "path_efficiency": [0.5, 0.9],
            "collision_amount": [1.0, 2.0],
            "time_to_goal": [10.0, 20.0],
            "roughness_mean": [3.0, 4.0],
        },
        schema={
            "planner": pl.Utf8,
            "success": pl.Float64,
            "path_efficiency": pl.Float64,
            "collision_amount": pl.Float64,
            "time_to_goal": pl.Float64,
            "roughness_mean": pl.Float64,
        },
    )


def _radar_spec(**options) -> PlotSpec:
    return PlotSpec(id="rad", type="radar", title="Radar Overview", data_key="*", differentiate="planner", options=options)


def test_radar_plotly_happy_path():
    html = RadarRenderer(_radar_spec()).render_plotly(_radar_df())
    assert html is not None
    assert html.startswith("<div")
    assert "scatterpolar" in html


def test_radar_plotly_fewer_than_three_metrics_returns_none():
    spec = _radar_spec(metrics=["success", "path_efficiency"])
    assert RadarRenderer(spec).render_plotly(_radar_df()) is None


def test_radar_plotly_null_metrics_are_filtered_out():
    df = _radar_df().with_columns(pl.lit(None).alias("bogus_a"), pl.lit(None).alias("bogus_b"))
    spec = _radar_spec(metrics=["success", "bogus_a", "bogus_b"])
    assert RadarRenderer(spec).render_plotly(df) is None


def test_radar_plotly_missing_diff_col_returns_none():
    df = _radar_df().drop("planner")
    assert RadarRenderer(_radar_spec()).render_plotly(df) is None


def test_radar_plotly_all_null_diff_col_renders_null_group():
    # polars group_by keeps null keys as their own group, so a fully-null
    # differentiate column still aggregates and renders (trace named None).
    df = _radar_df().with_columns(pl.lit(None).alias("planner"))
    html = RadarRenderer(_radar_spec()).render_plotly(df)
    assert html is not None and "scatterpolar" in html


@pytest.mark.parametrize(
    ("metric", "values", "expected"),
    [
        ("success", [0.0, 0.0], [1.0, 1.0]),
        ("success", [0.4, 0.8], [0.5, 1.0]),
        ("collision_amount", [1.0, 2.0], [1.0, 0.5]),
        ("collision_amount", [-1.0, 1.0], [2.0, 0.0]),
        ("collision_amount", [-2.0, 0.0], [1.0, 0.0]),
    ],
)
def test_normalize_puts_the_best_group_at_one(metric, values, expected):
    normalized = _normalize(pd.DataFrame({metric: values}), [metric], use_log_scale=False)
    assert normalized[metric].tolist() == pytest.approx(expected)


def test_normalize_scales_after_the_log():
    normalized = _normalize(pd.DataFrame({"success": [1.0, 3.0]}), ["success"], use_log_scale=True)
    assert normalized["success"].tolist() == pytest.approx([0.5, 1.0])


def test_radar_plotly_draws_the_normalized_values():
    spec = _radar_spec(metrics=["success", "path_efficiency", "collision_amount"])
    html = RadarRenderer(spec).render_plotly(_radar_df())
    start = html.index("[", html.index("Plotly.newPlot("))
    traces, _ = json.JSONDecoder().raw_decode(html, start)
    radii = {trace["name"]: dict(zip(trace["theta"], trace["r"], strict=True)) for trace in traces}
    assert radii["dwb"]["Success"] == pytest.approx(0.8)
    assert radii["teb"]["Success"] == pytest.approx(1.0)
    assert radii["dwb"]["Collision Amount"] == pytest.approx(1.0)
    assert radii["teb"]["Collision Amount"] == pytest.approx(0.5)


def test_radar_plotly_log_scale_option():
    spec = _radar_spec(use_log_scale=True)
    html = RadarRenderer(spec).render_plotly(_radar_df())
    assert html is not None and "scatterpolar" in html


def test_radar_plotly_single_planner_row():
    df = _radar_df().slice(0, 1)
    html = RadarRenderer(_radar_spec()).render_plotly(df)
    assert html is not None and "dwb" in html


def test_radar_plotly_units_in_labels():
    html = RadarRenderer(_radar_spec(), units={"collision_amount": "n"}).render_plotly(_radar_df())
    assert html is not None
    assert "Collision Amount [n]" in html


def test_radar_seaborn_happy_path(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec()).render_seaborn(_radar_df(), out)
    assert out.exists() and out.read_bytes()[:8] == _PNG_MAGIC


def test_radar_seaborn_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    df = _radar_df().drop("planner")
    RadarRenderer(_radar_spec()).render_seaborn(df, out)
    assert not out.exists()


def test_radar_seaborn_too_few_metrics_writes_nothing(tmp_path):
    out = tmp_path / "radar.png"
    spec = _radar_spec(metrics=["success", "path_efficiency"])
    RadarRenderer(spec).render_seaborn(_radar_df(), out)
    assert not out.exists()


def test_radar_seaborn_log_scale(tmp_path):
    out = tmp_path / "radar.png"
    RadarRenderer(_radar_spec(use_log_scale=True)).render_seaborn(_radar_df(), out)
    assert out.exists() and out.stat().st_size > 0


def test_radar_seaborn_metric_crossing_zero(tmp_path):
    out = tmp_path / "radar.png"
    df = _radar_df().with_columns(pl.Series("collision_amount", [-1.0, 1.0]))
    RadarRenderer(_radar_spec()).render_seaborn(df, out)
    assert out.exists() and out.stat().st_size > 0
