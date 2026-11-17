"""Unit tests for BasePlotRenderer and shared helper functions."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.base import (
    BasePlotRenderer,
    label_values,
)
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


class _DummyRenderer(BasePlotRenderer):
    PLOT_TYPE = "dummy"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        return None

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        return None


def test_base_resolve_channel_and_list() -> None:
    spec = PlotSpec(
        id="p1",
        type="bar",
        data={"source": "fact_episodes"},
        channels=PlotChannels(
            x="planner", y=["success", "time_to_goal"], color="world"
        ),
    )
    renderer = _DummyRenderer(spec, units={"time_to_goal": "s"})
    assert renderer.resolve_channel("x") == "planner"
    assert renderer.resolve_channel("y") == "success"
    assert renderer.resolve_channel_list("y") == ["success", "time_to_goal"]
    assert renderer.resolve_channel("color") == "world"
    assert renderer.resolve_channel("size", fallback="default_size") == "default_size"
    assert renderer.format_label("Time To Goal", "time_to_goal") == "Time To Goal [s]"


def test_label_values_uses_null_reason_when_present() -> None:
    df = pl.DataFrame(
        {
            "planner": ["dwb", None, "teb"],
            "planner_null_reason": [None, "not_recorded", None],
        }
    )
    labels = label_values(df, "planner").tolist()
    assert labels == ["dwb", "not_recorded", "teb"]


def test_renderer_rejects_missing_channel_column_and_names_it() -> None:
    import pytest

    from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY

    df = pl.DataFrame({"planner": ["dwb", "teb"], "duration_s": [1.0, 2.0]})
    spec = PlotSpec(
        id="missing_col_plot",
        type="bar",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="planner", y="nonexistent_metric"),
    )
    for plot_type, cls in sorted(RENDERER_REGISTRY.items()):
        p_spec = spec.model_copy(update={"type": plot_type})
        renderer = cls(p_spec)
        with pytest.raises(ValueError, match="nonexistent_metric"):
            renderer.validate_frame(df)


def test_renderer_rejects_unknown_option_and_names_it() -> None:
    import pytest

    from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY

    df = pl.DataFrame({"planner": ["dwb", "teb"], "mean": [1.0, 2.0]})
    spec = PlotSpec(
        id="bad_opt_plot",
        type="bar",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="planner", y="mean"),
        options={"auto_differentiate": True},
    )
    renderer = RENDERER_REGISTRY["bar"](spec)
    with pytest.raises(ValueError, match="auto_differentiate"):
        renderer.validate_frame(df)
