"""Unit tests for PlotlyRenderer dispatch across supported plot types."""

from __future__ import annotations

import polars as pl

from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_plotly_renderer_dispatches_bar_and_excludes_dendrogram() -> None:
    renderer = PlotlyRenderer(units={"path_length": "m"})
    assert "acoustic_field" in RENDERER_REGISTRY
    assert "acoustic_field_animation" in RENDERER_REGISTRY
    assert "dendrogram" not in RENDERER_REGISTRY

    spec = PlotSpec(
        id="bar_plotly",
        type="bar",
        title="Path Length",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="path_length"),
    )
    df = pl.DataFrame(
        {"planner": ["dwb", "mppi", None], "path_length": [14.0, 11.5, None]}
    )
    html = renderer.render(spec, df)
    assert isinstance(html, str)
    assert "dwb" in html
    assert "NULL" in html
