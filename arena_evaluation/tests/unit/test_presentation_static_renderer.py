"""Unit tests for StaticRenderer dispatch and non-blank image outputs."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_static_renderer_dispatches_and_writes_non_blank_png(
    tmp_path: pathlib.Path,
) -> None:
    renderer = StaticRenderer(units={"time_to_goal": "s"})
    assert "acoustic_field" not in RENDERER_REGISTRY
    assert "dendrogram" not in RENDERER_REGISTRY

    spec = PlotSpec(
        id="bar_static",
        type="bar",
        title="Mean Time To Goal",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb", None],
            "time_to_goal": [12.5, 10.2, None],
        }
    )
    out = tmp_path / "bar_static.png"
    renderer.render(spec, df, out)
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_static_renderer_unknown_type_is_noop(tmp_path: pathlib.Path) -> None:
    renderer = StaticRenderer()
    spec = PlotSpec.model_construct(
        id="unknown",
        type="not_a_plot_type",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="success"),
    )
    out = tmp_path / "noop.png"
    import pytest

    with pytest.raises(ValueError, match="unknown plot type"):
        renderer.render(spec, pl.DataFrame({"planner": ["a"], "success": [1.0]}), out)
    assert not out.exists()
