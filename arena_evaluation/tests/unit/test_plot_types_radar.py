"""Unit tests for RadarRenderer (long radar_profile shape and NULL handling)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.radar import RadarRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_radar_renderer_plotly_and_static_preserves_nulls(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="radar_1",
        type="radar",
        title="Planner Profile",
        data={"computation": "radar_profile"},
        channels=PlotChannels(x="metric", y="normalized_score", color="planner"),
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "dwb", "teb", "teb", "teb", None, None, None],
            "metric": ["success", "smoothness", "efficiency"] * 3,
            "normalized_score": [0.95, 0.80, 0.85, 0.88, None, 0.90, 0.70, 0.65, 0.75],
        }
    )
    renderer = RadarRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "dwb" in html and "NULL" in html

    out = tmp_path / "radar.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_radar_omits_null_vertex(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="radar_null",
        type="radar",
        title="Radar Null Vertex",
        data={"computation": "radar_profile"},
        channels=PlotChannels(x="metric", y="normalized_score", color="planner"),
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "dwb", "teb", "teb", "teb"],
            "metric": ["success", "speed", "safety", "success", "speed", "safety"],
            "normalized_score": [0.9, None, 0.8, 0.7, 0.6, 0.5],
        }
    )
    renderer = RadarRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "null" in html.lower()
    out = tmp_path / "radar_null.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_radar_rejects_fewer_than_three_axes_and_wide_mode() -> None:
    spec_two = PlotSpec(
        id="radar_two",
        type="radar",
        data={"computation": "radar_profile"},
        channels=PlotChannels(x="metric", y="normalized_score", color="planner"),
    )
    df_two = pl.DataFrame(
        {
            "planner": ["dwb", "dwb"],
            "metric": ["success", "spl"],
            "normalized_score": [0.9, 0.8],
        }
    )
    with pytest.raises(ValueError, match="at least 3 axes"):
        RadarRenderer(spec_two).render_plotly(df_two)

    spec_wide = PlotSpec(
        id="radar_wide",
        type="radar",
        data={"computation": "group_summary"},
        channels=PlotChannels(y=["a", "b", "c"], color="planner"),
    )
    df_wide = pl.DataFrame({"planner": ["dwb"], "a": [1.0], "b": [2.0], "c": [3.0]})
    with pytest.raises(ValueError, match="requires channel 'x'"):
        RadarRenderer(spec_wide).render_plotly(df_wide)
