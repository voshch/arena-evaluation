"""Timeseries renderer: one trace per episode and robot, in the interactive and the static plot alike."""

from __future__ import annotations

import pathlib
import re

import matplotlib.figure
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.timeseries import TimeseriesRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec() -> PlotSpec:
    return PlotSpec(
        id="timeseries_velocity",
        type="timeseries",
        data={"source": "v_series_energy"},
        channels=PlotChannels(x="time_s", y="velocity_mps", color="robot_id"),
    )


def _two_episodes_of_one_robot() -> pl.DataFrame:
    """One robot in two episodes on the same time base: the repeats must not join into one curve."""
    return pl.DataFrame(
        {
            "episode_id": [1, 1, 1, 2, 2, 2],
            "robot_id": ["jackal"] * 6,
            "time_s": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "velocity_mps": [0.5, 0.6, 0.7, 1.0, 1.1, 1.2],
        }
    )


def test_each_episode_of_a_robot_is_its_own_interactive_trace() -> None:
    html = TimeseriesRenderer(_spec()).render_plotly(_two_episodes_of_one_robot())

    assert html is not None
    assert len(re.findall(r'"mode":\s*"lines"', html)) == 2


def test_each_episode_of_a_robot_is_its_own_static_curve(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = TimeseriesRenderer(_spec())
    figures: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(renderer, "save_figure", lambda fig, _path: figures.append(fig))

    renderer.render_static(_two_episodes_of_one_robot(), tmp_path / "timeseries.png")

    curves = [[float(value) for value in line.get_ydata()] for line in figures[0].axes[0].lines]
    assert [0.5, 0.6, 0.7] in curves
    assert [1.0, 1.1, 1.2] in curves
