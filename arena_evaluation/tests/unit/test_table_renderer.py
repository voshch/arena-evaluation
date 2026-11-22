"""Unit tests for TableRenderer."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.presentation.plot_types.table import TableRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_table_renders_null_cell_and_null_reason(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="tbl_null",
        type="table",
        title="Null Table",
        data={"source": "v_planner_scorecard"},
        channels=PlotChannels(x="planner", y=["success_rate"]),
        options={"format": {"success_rate": "{:.1%}"}},
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb"],
            "success_rate": [0.95, None],
            "success_rate_null_reason": [None, "no_trials"],
        }
    )
    renderer = TableRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "95.0%" in html
    assert "no_trials" in html
    out = tmp_path / "tbl_null.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000
