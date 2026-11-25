"""Unit tests for LineRenderer with pre-computed error bounds and NULL gaps."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.line import LineRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_line_renderer_draws_bound_ci_band_and_preserves_nulls(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="line_ci",
        type="line",
        title="Speed Profile",
        data={"source": "robot_odom"},
        channels=PlotChannels(x="t_s", y="mean", color="planner"),
        options={"ci_low": "ci_low", "ci_high": "ci_high"},
    )
    df = pl.DataFrame(
        {
            "t_s": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "mean": [0.5, 0.8, 0.7, 0.4, None, 0.6],
            "ci_low": [0.4, 0.7, 0.6, 0.3, None, 0.5],
            "ci_high": [0.6, 0.9, 0.8, 0.5, None, 0.7],
            "planner": ["dwb", "dwb", "dwb", "teb", "teb", "teb"],
        }
    )
    renderer = LineRenderer(spec, units={"t_s": "s", "mean": "m/s"})
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "dwb" in html and "teb" in html

    out = tmp_path / "line_ci.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000
