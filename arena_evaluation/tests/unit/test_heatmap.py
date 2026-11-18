"""Unit tests for HeatmapRenderer on pre-computed long and wide matrices."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.heatmap import HeatmapRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_heatmap_long_form_preserves_null_cells(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="hm_long",
        type="heatmap",
        data={"computation": "group_summary"},
        channels=PlotChannels(x="world", y="planner", color="mean"),
    )
    df = pl.DataFrame(
        {
            "world": ["w1", "w2", "w1", "w2"],
            "planner": ["dwb", "dwb", "teb", "teb"],
            "mean": [0.9, None, 0.8, 0.95],
        }
    )
    renderer = HeatmapRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "dwb" in html

    out = tmp_path / "hm_long.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_heatmap_wide_correlation_matrix(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="hm_corr",
        type="heatmap",
        data={"computation": "correlation_matrix"},
        channels=PlotChannels(x="metric_x", y="metric_y", color="pearson_r"),
    )
    # differs from overhaul/source:plot_types/heatmap.py:85: correlation_matrix emits long (metric_x, metric_y, pearson_r) and renders NULL as blank cell
    df = pl.DataFrame(
        {
            "metric_x": ["time_to_goal", "time_to_goal", "path_length", "path_length"],
            "metric_y": ["time_to_goal", "path_length", "time_to_goal", "path_length"],
            "pearson_r": [1.0, 0.82, None, 1.0],
            "n_pairs": [10, 10, 0, 10],
        }
    )
    renderer = HeatmapRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "null" in html.lower()
    out = tmp_path / "hm_corr.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000
