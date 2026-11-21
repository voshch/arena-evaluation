"""Timeseries renderer: one mean curve per colour value, with the 10th to 90th percentile over episodes as its band."""

from __future__ import annotations

import pathlib
import re

import matplotlib.figure
import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.timeseries import (
    TimeseriesRenderer,
    aggregate_curves,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _spec() -> PlotSpec:
    return PlotSpec(
        id="timeseries_velocity",
        type="timeseries",
        data={"source": "v_series_energy"},
        channels=PlotChannels(x="time_s", y="velocity_mps", color="planner_world"),
    )


def _three_episodes_of_one_planner() -> pl.DataFrame:
    """Three episodes of one planner on the same one-second time base."""
    return pl.DataFrame(
        {
            "episode_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "planner_world": ["dwb · concourse_headon_aware"] * 9,
            "time_s": [0.0, 1.0, 2.0] * 3,
            "velocity_mps": [0.0, 0.5, 1.0, 0.3, 0.7, 1.2, 0.6, 0.9, 1.4],
        }
    )


def _aggregate() -> pl.DataFrame:
    return aggregate_curves(
        _three_episodes_of_one_planner(),
        x_col="time_s",
        y_col="velocity_mps",
        colour_col="planner_world",
        bin_width_s=1.0,
    )


def test_each_time_bin_is_the_mean_over_the_episodes() -> None:
    assert _aggregate()["mean"].to_list() == pytest.approx([0.3, 0.7, 1.2])


def test_the_band_spans_the_tenth_to_ninetieth_percentile_over_the_episodes() -> None:
    aggregated = _aggregate()

    assert aggregated["low"].to_list() == pytest.approx([0.06, 0.54, 1.04])
    assert aggregated["high"].to_list() == pytest.approx([0.54, 0.86, 1.36])


def test_interactive_plot_draws_one_mean_line_and_band_per_colour_not_per_episode() -> (
    None
):
    html = TimeseriesRenderer(_spec()).render_plotly(_three_episodes_of_one_planner())

    assert html is not None
    assert len(re.findall(r'"mode":\s*"lines"', html)) == 2


def test_static_plot_draws_one_mean_line_per_colour(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = TimeseriesRenderer(_spec())
    figures: list[matplotlib.figure.Figure] = []
    monkeypatch.setattr(renderer, "save_figure", lambda fig, _path: figures.append(fig))

    renderer.render_static(
        _three_episodes_of_one_planner(), tmp_path / "timeseries.png"
    )

    lines = figures[0].axes[0].lines
    assert len(lines) == 1
    assert [float(value) for value in lines[0].get_ydata()] == pytest.approx(
        [0.3, 0.7, 1.2]
    )
