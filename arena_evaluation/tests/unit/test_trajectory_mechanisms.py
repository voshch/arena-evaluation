"""Trajectory plot tests for trace partitioning, markers, and world geometry."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.checks import PlotDiagnostics
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_trajectory_multi_world_and_missing_geometry_diagnostics(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="traj_multi",
        type="trajectory",
        title="Multi-World Trajectory",
        data={"source": "robot_tracks"},
        channels=PlotChannels(x="x", y="y", color="planner"),
    )
    df = pl.DataFrame(
        {
            "world": ["w1", "w1", "w2", "w2"],
            "episode_id": ["e1", "e1", "e2", "e2"],
            "planner": ["dwb", "dwb", "mppi", "mppi"],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    diag = PlotDiagnostics()
    renderer = TrajectoryRenderer(spec)
    renderer.diagnostics = diag
    html = renderer.render_plotly(df)
    assert isinstance(html, list) and len(html) == 2
    assert "dwb" in html[0] and "mppi" in html[1]
    assert any("dim_world_geometry" in msg for msg in diag.issues)

    out = tmp_path / "traj_multi.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


