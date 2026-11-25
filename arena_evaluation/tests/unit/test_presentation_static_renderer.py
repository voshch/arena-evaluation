"""Unit tests for StaticRenderer dispatch and non-blank image outputs."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


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
