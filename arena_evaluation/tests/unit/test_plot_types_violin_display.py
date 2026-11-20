"""Violin display: Tukey outliers are left out of the drawing per group and named in a note; stored values stay."""

from __future__ import annotations

import polars as pl

from arena_evaluation.presentation.plot_types.base import split_display_outliers
from arena_evaluation.presentation.plot_types.violin import ViolinRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def _roughness() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "local_planner": ["dwb"] * 6 + ["mppi"] * 6,
            "roughness_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )


def test_outliers_beyond_one_and_a_half_iqr_are_split_off_per_group() -> None:
    inliers, hidden = split_display_outliers(
        _roughness(), value="roughness_mean", groups=["local_planner"]
    )

    assert inliers.height == 11
    assert hidden.to_dicts() == [
        {"local_planner": "dwb", "hidden": 1},
        {"local_planner": "mppi", "hidden": 0},
    ]


def test_violin_names_the_points_it_does_not_draw() -> None:
    spec = PlotSpec(
        id="violin_roughness_local",
        type="violin",
        data={"source": "v_episode_summary"},
        channels=PlotChannels(x="local_planner", y="roughness_mean"),
    )

    html = ViolinRenderer(spec).render_plotly(_roughness())

    assert html is not None
    assert "Not drawn, beyond 1.5 IQR of each group: dwb 1" in html
