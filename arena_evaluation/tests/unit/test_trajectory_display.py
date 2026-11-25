"""Trajectory display: a track is one continuous line with no sample stamps, and long tracks stay small in HTML."""

from __future__ import annotations

import pathlib

import matplotlib.figure
import numpy as np
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.trajectory import (
    TrajectoryRenderer,
    simplify_track,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec() -> PlotSpec:
    return PlotSpec(
        id="trajectory_plot",
        type="trajectory",
        data={"stored_query": "robot_tracks_non_ref"},
        channels=PlotChannels(x="x", y="y", color="robot_id"),
    )


def _track(samples: int) -> pl.DataFrame:
    steps = np.arange(samples, dtype=np.float64)
    return pl.DataFrame(
        {
            "episode_id": [1] * samples,
            "robot_id": ["jackal"] * samples,
            "t_ns": (steps * 1e9).astype(np.int64),
            "x": np.linspace(0.0, 4.0, samples),
            "y": np.linspace(0.0, 2.0, samples),
        }
    )


def test_interactive_track_is_one_continuous_line_without_sample_markers() -> None:
    html = TrajectoryRenderer(_spec()).render_plotly(_track(40))

    assert html is not None
    assert '"mode":"lines"' in html
    assert "lines+markers" not in html


def test_static_track_has_no_start_or_end_markers(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = TrajectoryRenderer(_spec())
    figures: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(renderer, "save_figure", lambda fig, _path: figures.append(fig))

    renderer.render_static(_track(40), tmp_path / "trajectory.png")

    assert len(figures[0].axes[0].collections) == 0


def test_simplify_track_keeps_the_corner_and_both_ends() -> None:
    xs = np.concatenate([np.linspace(0.0, 10.0, 1001), np.full(500, 10.0)])
    ys = np.concatenate([np.zeros(1001), np.linspace(0.0, 5.0, 500)])

    sx, sy = simplify_track(xs, ys, tolerance_m=0.01)

    assert len(sx) == 3
    assert (sx[0], sy[0]) == pytest.approx((0.0, 0.0))
    assert (sx[1], sy[1]) == pytest.approx((10.0, 0.0))
    assert (sx[2], sy[2]) == pytest.approx((10.0, 5.0))


def test_interactive_html_of_a_long_recording_stays_small() -> None:
    samples = 30_000
    steps = np.linspace(0.0, 1.0, samples)
    long_track = pl.DataFrame(
        {
            "episode_id": [1] * samples,
            "robot_id": ["jackal"] * samples,
            "x": 10.0 * steps + 0.001 * np.sin(2000.0 * steps),
            "y": 5.0 * steps**2 + 0.001 * np.cos(1500.0 * steps),
        }
    )

    html = TrajectoryRenderer(_spec()).render_plotly(long_track)

    assert html is not None
    assert len(html) < 20_000
