"""Renderer contracts: undrawable frames fail loudly and the interval type dispatches fully."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.checks import PlotRenderError
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _box_spec() -> PlotSpec:
    return PlotSpec(
        id="box_all_null",
        type="box",
        title="Box",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )


def _interval_spec() -> PlotSpec:
    return PlotSpec(
        id="ci_panel",
        type="interval",
        title="Paired difference",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="planner", y="mean_diff"),
        options={"low": "ci_low", "high": "ci_high"},
    )


def test_box_renderer_refuses_an_all_null_metric(tmp_path: pathlib.Path) -> None:
    frame = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "time_to_goal": [None, None],
            "time_to_goal_null_reason": ["not_recorded", "not_recorded"],
        }
    )

    with pytest.raises(PlotRenderError, match="100% NULL"):
        StaticRenderer().render(_box_spec(), frame, tmp_path / "box.png")

    assert not (tmp_path / "box.png").exists()


def test_box_renderer_refuses_an_empty_frame(tmp_path: pathlib.Path) -> None:
    frame = pl.DataFrame(schema={"planner": pl.String, "time_to_goal": pl.Float64})

    with pytest.raises(PlotRenderError, match="empty frame"):
        StaticRenderer().render(_box_spec(), frame, tmp_path / "box.png")


def test_static_renderer_raises_when_a_renderer_writes_no_file(
    tmp_path: pathlib.Path,
) -> None:
    spec = PlotSpec(
        id="empty_histogram",
        type="histogram",
        title="Empty histogram",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="time_to_goal"),
    )
    frame = pl.DataFrame(schema={"time_to_goal": pl.Float64})

    with pytest.raises(PlotRenderError, match="produced no file"):
        StaticRenderer().render(spec, frame, tmp_path / "histogram.png")

    assert not (tmp_path / "histogram.png").exists()


def test_interval_renderer_requires_bound_columns(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="ci_unbound",
        type="interval",
        title="Unbound",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="planner", y="mean_diff"),
    )
    frame = pl.DataFrame({"planner": ["dwb"], "mean_diff": [0.1]})

    with pytest.raises(ValueError, match="requires options"):
        StaticRenderer().render(spec, frame, tmp_path / "unbound.png")
