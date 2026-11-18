"""Unit tests for HierarchyRenderer (treemap and sunburst; no dendrogram)."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.hierarchy import HierarchyRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_hierarchy_treemap_and_sunburst_plotly_and_static(
    tmp_path: pathlib.Path,
) -> None:
    df = pl.DataFrame(
        {
            "world": ["hospital", "hospital", "office"],
            "planner": ["dwb", "teb", "dwb"],
            "count": [12, 8, 15],
        }
    )
    treemap_spec = PlotSpec(
        id="hier_tree",
        type="hierarchy",
        data={"source": "fact_episodes"},
        channels=PlotChannels(hierarchy=["world", "planner"], y="count"),
        options={"mode": "treemap"},
    )
    renderer = HierarchyRenderer(treemap_spec)
    html_tree = renderer.render_plotly(df)
    assert isinstance(html_tree, str)
    assert "hospital" in html_tree

    out = tmp_path / "hier.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000

    sunburst_spec = PlotSpec(
        id="hier_sun",
        type="hierarchy",
        data={"source": "fact_episodes"},
        channels=PlotChannels(hierarchy=["world", "planner"], y="count"),
        options={"mode": "sunburst"},
    )
    html_sun = HierarchyRenderer(sunburst_spec).render_plotly(df)
    assert isinstance(html_sun, str)
    assert "office" in html_sun
