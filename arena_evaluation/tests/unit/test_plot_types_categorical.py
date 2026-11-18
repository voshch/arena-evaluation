"""Unit tests for categorical plot renderers (bar, box, violin)."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.bar import BarRenderer
from arena_evaluation.presentation.plot_types.box import BoxRenderer
from arena_evaluation.presentation.plot_types.violin import ViolinRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


@pytest.mark.parametrize(
    ("renderer_cls", "plot_type"),
    [
        (BarRenderer, "bar"),
        (BoxRenderer, "box"),
        (ViolinRenderer, "violin"),
    ],
)
def test_categorical_renderers_handle_null_groups_and_values(
    tmp_path: pathlib.Path, renderer_cls: type, plot_type: str
) -> None:
    spec = PlotSpec(
        id=f"cat_{plot_type}",
        type=plot_type,
        title=f"Test {plot_type}",
        data={"source": "fact_episodes"},
        channels=PlotChannels(x="planner", y="time_to_goal"),
    )
    # differs from overhaul/source:plot_types/bar.py:130: bar requires unique aggregated cells, not raw duplicate rows
    if plot_type == "bar":
        df = pl.DataFrame(
            {
                "planner": ["dwb", "teb", None],
                "time_to_goal": [11.0, 9.0, None],
                "err_plus": [1.0, 0.5, None],
                "err_minus": [1.0, 0.5, None],
            }
        )
    else:
        df = pl.DataFrame(
            {
                "planner": ["dwb", "dwb", "teb", "teb", None],
                "time_to_goal": [10.0, 12.0, 8.5, 9.5, None],
                "err_plus": [1.0, 1.0, 0.5, 0.5, None],
                "err_minus": [1.0, 1.0, 0.5, 0.5, None],
            }
        )
    renderer = renderer_cls(spec, units={"time_to_goal": "s"})
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert "NULL" in html

    out = tmp_path / f"{plot_type}.png"
    renderer.render_static(df, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_bar_renderer_error_bars_and_facets(tmp_path: pathlib.Path) -> None:
    spec = PlotSpec(
        id="bar_err_facet",
        type="bar",
        title="Closing Share",
        data={"stored_query": "contact_attribution_summary"},
        channels=PlotChannels(
            x="planner", y="mean_robot_closing_share", facet_col="crowd_condition"
        ),
        options={"err_plus": "err_plus", "err_minus": "err_minus"},
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "teb", "dwb", "teb"],
            "crowd_condition": ["low", "low", "high", "high"],
            "mean_robot_closing_share": [0.4, 0.6, 0.5, 0.7],
            "err_plus": [0.05, 0.04, 0.06, 0.03],
            "err_minus": [0.05, 0.04, 0.06, 0.03],
        }
    )
    renderer = BarRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert '"error_y":{"array":[0.05,0.04]' in html.replace(" ", "")
    assert "crowd_condition=low" in html and "crowd_condition=high" in html

    out = tmp_path / "bar_facet.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000


def test_histogram_renderer_uses_declared_bin_edges(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.presentation.plot_types.histogram import HistogramRenderer

    spec = PlotSpec(
        id="hist_bins",
        type="histogram",
        title="Path Length",
        data={"computation": "distribution_bins"},
        channels=PlotChannels(x="bin_center", y="count", color="planner"),
        options={"bin_low": "bin_low", "bin_high": "bin_high"},
    )
    df = pl.DataFrame(
        {
            "planner": ["dwb", "dwb"],
            "bin_low": [0.0, 2.5],
            "bin_high": [2.5, 5.0],
            "bin_center": [1.25, 3.75],
            "count": [4, 7],
        }
    )
    renderer = HistogramRenderer(spec)
    html = renderer.render_plotly(df)
    assert isinstance(html, str)
    assert '"width":[2.5,2.5]' in html.replace(" ", "")

    out = tmp_path / "hist.png"
    renderer.render_static(df, out)
    assert out.exists() and out.stat().st_size > 1000
