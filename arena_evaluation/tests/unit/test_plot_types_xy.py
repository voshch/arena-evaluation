"""Unit tests for XY and spatial plot renderers (scatter, histogram, timeseries, trajectory)."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.histogram import HistogramRenderer
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.resolver import PlotDiagnostics
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_histogram_renderer_prebinned_and_raw(tmp_path: pathlib.Path) -> None:
    import pytest

    # differs from overhaul/source:plot_types/histogram.py:35: histogram rejects raw rows without pre-binned y
    raw_spec = PlotSpec(
        id="hist_raw",
        type="histogram",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="time_to_goal", color="planner"),
    )
    raw_df = pl.DataFrame(
        {
            "time_to_goal": [1.0, 2.0, 2.5, 3.0, 4.0],
            "planner": ["a", "a", "b", "b", "b"],
        }
    )
    with pytest.raises(ValueError, match="pre-binned"):
        HistogramRenderer(raw_spec).render_plotly(raw_df)

    binned_spec = PlotSpec(
        id="hist_binned",
        type="histogram",
        data={"computation": "distribution_bins"},
        channels=PlotChannels(x="bin_center", y="count", color="planner"),
    )
    binned_df = pl.DataFrame(
        {
            "bin_center": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "count": [5, 12, 3, 2, 8, 6],
            "planner": ["a", "a", "a", "b", "b", "b"],
        }
    )
    binned_r = HistogramRenderer(binned_spec)
    assert isinstance(binned_r.render_plotly(binned_df), str)
    out_bin = tmp_path / "hist_binned.png"
    binned_r.render_static(binned_df, out_bin)
    assert out_bin.exists() and out_bin.stat().st_size > 1000


def test_trajectory_renderer_with_world_geometry_and_missing_world_notice(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="traj_1",
        type="trajectory",
        data={"source": "robot_odom"},
        channels=PlotChannels(x="x", y="y", color="planner"),
    )
    df = pl.DataFrame(
        {
            "world": ["hospital", "hospital", "hospital"],
            "planner": ["dwb", "dwb", "dwb"],
            "x": [0.0, 1.5, 3.0],
            "y": [0.0, 0.5, 1.0],
        }
    )
    geom = pl.DataFrame(
        {
            "world": ["hospital", "hospital", "hospital", "hospital", "hospital"],
            "element_kind": ["wall", "wall", "zone", "zone", "zone"],
            "element_name": ["wall_0", "wall_0", "hall", "hall", "hall"],
            "part_idx": [0, 0, 0, 0, 0],
            "vertex_idx": [0, 1, 0, 1, 2],
            "x": [-1.0, 4.0, -1.0, 4.0, 4.0],
            "y": [-1.0, -1.0, -1.0, -1.0, 2.0],
        }
    )
    renderer = TrajectoryRenderer(spec)
    renderer.world_geometry = geom
    renderer.diagnostics = PlotDiagnostics()
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "wall:wall_0" in html
    assert renderer.diagnostics.issues == []

    out = tmp_path / "traj.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000

    # Missing world rows emit a notice and still render cleanly
    renderer_missing = TrajectoryRenderer(spec)
    renderer_missing.world_geometry = geom.filter(pl.col("world") == "other_world")
    renderer_missing.diagnostics = PlotDiagnostics()
    out_missing = tmp_path / "traj_missing.png"
    renderer_missing.render_static(df, out_missing)
    assert out_missing.exists() and out_missing.stat().st_size > 1000
    assert any(
        "dim_world_geometry has no rows" in w
        for w in renderer_missing.diagnostics.issues
    )
