"""Unit tests for RadarRenderer (long radar_profile shape and NULL handling)."""

from __future__ import annotations


import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.radar import RadarRenderer
from arena_evaluation.presentation.viz_manifest import PlotChannels, PlotSpec


def test_radar_rejects_fewer_than_three_axes_and_wide_mode() -> None:
    spec_two = PlotSpec(
        id="radar_two",
        type="radar",
        data={"computation": "radar_profile"},
        channels=PlotChannels(x="metric", y="normalized_score", color="planner"),
    )
    df_two = pl.DataFrame(
        {
            "planner": ["dwb", "dwb"],
            "metric": ["success", "spl"],
            "normalized_score": [0.9, 0.8],
        }
    )
    with pytest.raises(ValueError, match="at least 3 axes"):
        RadarRenderer(spec_two).render_plotly(df_two)

    spec_wide = PlotSpec(
        id="radar_wide",
        type="radar",
        data={"computation": "group_summary"},
        channels=PlotChannels(y=["a", "b", "c"], color="planner"),
    )
    df_wide = pl.DataFrame({"planner": ["dwb"], "a": [1.0], "b": [2.0], "c": [3.0]})
    with pytest.raises(ValueError, match="requires channel 'x'"):
        RadarRenderer(spec_wide).render_plotly(df_wide)
