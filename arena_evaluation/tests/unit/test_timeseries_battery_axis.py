"""The power and battery plot: power on the left axis, battery state of charge on a second axis, in the interactive and the static plot."""

from __future__ import annotations

import pathlib
import re

import matplotlib.axes
import matplotlib.figure
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.timeseries import TimeseriesRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec() -> PlotSpec:
    return PlotSpec(
        id="timeseries_power",
        type="timeseries",
        data={"source": "v_series_energy"},
        channels=PlotChannels(x="time_s", y="power_total_w", y2="battery_soc_pct", color="robot_id"),
    )


def _power_and_battery() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "episode_id": [1, 1, 1],
            "robot_id": ["jackal"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "power_total_w": [120.0, 480.0, 300.0],
            "battery_soc_pct": [99.9, 99.8, 99.7],
        }
    )


def _drawn_values(axis: matplotlib.axes.Axes) -> list[float]:
    return [float(value) for value in axis.lines[0].get_ydata()]


def test_battery_is_a_second_axis_in_the_interactive_plot() -> None:
    html = TimeseriesRenderer(_spec()).render_plotly(_power_and_battery())

    assert html is not None
    assert re.search(r'"yaxis":\s*"y2"', html)
    assert re.search(r'"yaxis2"\s*:', html)
    assert len(re.findall(r'"mode":\s*"lines"', html)) == 2


def test_battery_is_a_second_axis_in_the_static_plot(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = TimeseriesRenderer(_spec())
    figures: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(renderer, "save_figure", lambda fig, _path: figures.append(fig))

    renderer.render_static(_power_and_battery(), tmp_path / "power.png")

    power_axis, battery_axis = figures[0].axes
    assert _drawn_values(power_axis) == [120.0, 480.0, 300.0]
    assert _drawn_values(battery_axis) == [99.9, 99.8, 99.7]
