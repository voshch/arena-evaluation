"""Dispatch test: every reviewed plot type renders through both public renderers once."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.checks import image_is_blank
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec

_CASES: dict[str, tuple[PlotSpec, pl.DataFrame]] = {
    "interval": (
        PlotSpec(
            id="interval_dispatch",
            type="interval",
            title="Interval",
            data={"source": "v_episode_summary"},
            channels=PlotChannels(x="planner", y="estimate"),
            options={"low": "ci_low", "high": "ci_high"},
        ),
        pl.DataFrame(
            {
                "planner": ["dwb", "teb"],
                "estimate": [0.4, 0.6],
                "ci_low": [0.3, 0.5],
                "ci_high": [0.5, 0.7],
            }
        ),
    ),
    "rank_swap": (
        PlotSpec(
            id="rank_swap_dispatch",
            type="rank_swap",
            title="Rank swap",
            data={"source": "v_episode_summary"},
            channels=PlotChannels(x="planner", y="swap_rate"),
        ),
        pl.DataFrame({"planner": ["dwb", "teb"], "swap_rate": [0.1, 0.3]}),
    ),
    "breakdown": (
        PlotSpec(
            id="breakdown_dispatch",
            type="breakdown",
            title="Breakdown",
            data={"source": "v_episode_summary"},
            channels=PlotChannels(x="planner", y="share", color="component"),
        ),
        pl.DataFrame(
            {
                "planner": ["dwb", "dwb", "teb", "teb"],
                "component": ["static", "mechanical", "static", "mechanical"],
                "share": [0.4, 0.6, 0.7, 0.3],
            }
        ),
    ),
    "trajectory_slice": (
        PlotSpec(
            id="trajectory_slice_dispatch",
            type="trajectory_slice",
            title="Slice",
            data={"source": "v_episode_summary"},
            channels=PlotChannels(x="x", y="y"),
        ),
        pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.5]}),
    ),
}


@pytest.mark.parametrize("plot_type", sorted(_CASES))
def test_new_type_dispatches_through_both_renderers(
    plot_type: str, tmp_path: pathlib.Path
) -> None:
    spec, frame = _CASES[plot_type]
    png = tmp_path / f"{plot_type}.png"

    StaticRenderer().render(spec, frame, png)
    html = PlotlyRenderer().render(spec, frame, run_dir=tmp_path)

    assert png.exists()
    assert image_is_blank(png) is False
    assert isinstance(html, str) and len(html) > 0
    assert "plotly" in html.lower()
