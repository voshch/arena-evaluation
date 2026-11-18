"""Trajectory plot tests for trace partitioning, markers, and world geometry."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.checks import PlotDiagnostics
from arena_evaluation.presentation.plot_types.trajectory import TrajectoryRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_trajectory_splits_by_episode_and_does_not_break_large_steps(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="traj_ep",
        type="trajectory",
        title="Episode Trajectory",
        data={"source": "robot_tracks"},
        channels=PlotChannels(x="x", y="y", color="planner"),
    )
    df = pl.DataFrame(
        {
            "episode_id": ["ep1", "ep1", "ep2", "ep2"],
            "planner": ["dwb", "dwb", "dwb", "dwb"],
            "x": [0.0, 20.0, 1.0, 2.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        }
    )
    renderer = TrajectoryRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    # Continuous single trace per episode without NaN teleport splitting
    assert "20" in html and "null" not in html.lower()
    assert "1" in html and "2" in html

    out = tmp_path / "traj_ep.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


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


def test_trajectory_renders_door_and_elevator_from_dim_world_geometry(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="traj_mech",
        type="trajectory",
        title="Mechanism Trajectory",
        data={"source": "robot_tracks"},
        channels=PlotChannels(x="x", y="y", color="planner"),
    )
    df = pl.DataFrame(
        {
            "world": ["mech_world"] * 4,
            "planner": ["dwb"] * 4,
            "x": [0.0, 2.0, 4.0, 6.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )
    geom = pl.DataFrame(
        {
            "world": ["mech_world"] * 9,
            "element_kind": [
                "door",
                "door",
                "elevator",
                "elevator",
                "elevator",
                "elevator",
                "wall",
                "wall",
                "wall",
            ],
            "element_name": [
                "door_1",
                "door_1",
                "elev_1",
                "elev_1",
                "elev_1",
                "elev_1",
                "wall_0",
                "wall_0",
                "wall_0",
            ],
            "part_idx": [0] * 9,
            "vertex_idx": [0, 1, 0, 1, 2, 3, 0, 1, 2],
            "x": [2.0, 2.0, 4.5, 5.5, 5.5, 4.5, -1.0, 7.0, 7.0],
            "y": [-1.0, 1.0, -0.5, -0.5, 0.5, 0.5, -2.0, -2.0, 2.0],
        }
    )
    renderer = TrajectoryRenderer(spec)
    renderer.world_geometry = geom
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "door:door_1" in html
    assert "elevator:elev_1" in html

    out = tmp_path / "traj_mech.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000
