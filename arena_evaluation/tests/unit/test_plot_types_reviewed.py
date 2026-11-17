"""Content tests for the interval, breakdown, rank-swap and trajectory-slice types."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.breakdown import BreakdownRenderer
from arena_evaluation.presentation.plot_types.interval import IntervalRenderer
from arena_evaluation.presentation.viz_manifest import PlotSpec


def _spec(
    plot_type: str,
    channels: dict[str, str],
    options: dict[str, str] | None = None,
) -> PlotSpec:
    return PlotSpec.model_validate(
        {
            "id": plot_type,
            "title": plot_type,
            "type": plot_type,
            "data": {"source": "v_episode_summary"},
            "channels": channels,
            "options": options or {},
        }
    )


def test_interval_refuses_a_frame_without_its_bound_columns() -> None:
    frame = pl.DataFrame(
        {
            "planner": ["planner_a"],
            "mean_diff": [-4.0],
            "ci_high": [-3.0],
        }
    )
    renderer = IntervalRenderer(
        _spec(
            "interval",
            {"x": "planner", "y": "mean_diff"},
            {"low": "ci_low", "high": "ci_high"},
        ),
        units={},
    )

    with pytest.raises(ValueError, match="ci_low"):
        renderer.validate_frame(frame)


def test_breakdown_refuses_a_spec_without_a_colour_channel() -> None:
    frame = pl.DataFrame(
        {
            "planner": ["planner_a", "planner_b"],
            "share": [1.0, 1.0],
        }
    )
    renderer = BreakdownRenderer(
        _spec("breakdown", {"x": "planner", "y": "share"}), units={}
    )

    with pytest.raises(ValueError, match="requires channel 'color'"):
        renderer.validate_frame(frame)
