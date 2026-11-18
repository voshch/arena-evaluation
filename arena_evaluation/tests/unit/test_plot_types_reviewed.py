"""Content tests for the interval, breakdown, rank-swap and trajectory-slice types."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from matplotlib.patches import Rectangle

from arena_evaluation.presentation.plot_types.breakdown import BreakdownRenderer
from arena_evaluation.presentation.plot_types.interval import IntervalRenderer
from arena_evaluation.presentation.plot_types.rank_swap import RankSwapRenderer
from arena_evaluation.presentation.plot_types.trajectory_slice import (
    TrajectorySliceRenderer,
)
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


def test_interval_error_bars_span_the_bound_columns() -> None:
    frame = pl.DataFrame(
        {
            "planner": ["planner_a", "planner_b", "planner_c"],
            "mean_diff": [-4.0, 0.1, 2.0],
            "ci_low": [-5.0, -0.6, 1.0],
            "ci_high": [-3.0, 0.8, 3.0],
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
    renderer.validate_frame(frame)
    figure = renderer.build_static_figure(frame)
    assert figure is not None

    axis = figure.axes[0]
    segments = axis.collections[0].get_segments()
    extents = np.array([[segment[0][1], segment[1][1]] for segment in segments])
    expected = np.column_stack(
        [frame["ci_low"].to_numpy(), frame["ci_high"].to_numpy()]
    )
    np.testing.assert_allclose(extents, expected)
    np.testing.assert_allclose(axis.lines[0].get_ydata(), frame["mean_diff"].to_numpy())

    plotly_figure = renderer.build_plotly_figure(frame)
    assert plotly_figure is not None
    whisker_y = np.array(
        [np.nan if value is None else value for value in plotly_figure.data[0].y],
        dtype=float,
    ).reshape(-1, 3)
    np.testing.assert_allclose(whisker_y[:, :2], expected)
    np.testing.assert_allclose(
        np.asarray(plotly_figure.data[1].y, dtype=float),
        frame["mean_diff"].to_numpy(),
    )
    plt.close(figure)


def test_interval_draws_bounds_that_exclude_the_point_by_rounding() -> None:
    frame = pl.DataFrame(
        {
            "metric": ["time_to_goal"],
            "mean_diff": [1.0],
            "ci_low": [0.9999999999999993],
            "ci_high": [0.9999999999999999],
        }
    )
    renderer = IntervalRenderer(
        _spec(
            "interval",
            {"x": "metric", "y": "mean_diff"},
            {"low": "ci_low", "high": "ci_high"},
        ),
        units={},
    )
    renderer.validate_frame(frame)
    figure = renderer.build_static_figure(frame)
    assert figure is not None
    segments = figure.axes[0].collections[0].get_segments()
    np.testing.assert_allclose(
        [[segment[0][1], segment[1][1]] for segment in segments],
        [[0.9999999999999993, 0.9999999999999999]],
    )
    plt.close(figure)


def test_breakdown_stacks_components_and_lists_them_in_the_legend() -> None:
    frame = pl.DataFrame(
        {
            "planner": ["planner_a", "planner_a", "planner_b", "planner_b"],
            "component": ["robot", "target", "robot", "target"],
            "share": [0.1, 0.9, 0.6, 0.4],
        }
    )
    renderer = BreakdownRenderer(
        _spec(
            "breakdown",
            {"x": "planner", "y": "share", "color": "component"},
        ),
        units={},
    )
    renderer.validate_frame(frame)
    figure = renderer.build_static_figure(frame)
    assert figure is not None

    axis = figure.axes[0]
    patches = [item for item in axis.patches if isinstance(item, Rectangle)]
    heights_by_x: dict[float, list[float]] = {}
    for patch in patches:
        heights_by_x.setdefault(round(float(patch.get_x()), 6), []).append(
            float(patch.get_height())
        )
    assert len(heights_by_x) == 2
    np.testing.assert_allclose(
        [sum(heights) for heights in heights_by_x.values()], [1.0, 1.0]
    )
    legend = axis.get_legend()
    assert legend is not None
    assert sorted(text.get_text() for text in legend.get_texts()) == [
        "robot",
        "target",
    ]

    plotly_figure = renderer.build_plotly_figure(frame)
    assert plotly_figure is not None
    assert plotly_figure.layout.barmode == "stack"
    assert sorted(str(trace.name) for trace in plotly_figure.data) == [
        "robot",
        "target",
    ]
    plt.close(figure)


def test_rank_swap_draws_one_bar_per_x() -> None:
    frame = pl.DataFrame({"pair": ["blind-visible", "seed halves"], "swaps": [3, 1]})
    renderer = RankSwapRenderer(
        _spec("rank_swap", {"x": "pair", "y": "swaps"}), units={}
    )
    renderer.validate_frame(frame)
    figure = renderer.build_static_figure(frame)
    assert figure is not None

    axis = figure.axes[0]
    patches = [item for item in axis.patches if isinstance(item, Rectangle)]
    assert len(patches) == 2
    assert len({round(float(patch.get_x()), 6) for patch in patches}) == 2
    np.testing.assert_allclose(
        [float(patch.get_height()) for patch in patches], [3.0, 1.0]
    )

    plotly_figure = renderer.build_plotly_figure(frame)
    assert plotly_figure is not None
    assert len(plotly_figure.data) == 1
    assert len(plotly_figure.data[0].x) == 2
    plt.close(figure)


def test_trajectory_slice_draws_one_line_with_every_point() -> None:
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.5, 0.4, 1.0]})
    renderer = TrajectorySliceRenderer(
        _spec("trajectory_slice", {"x": "x", "y": "y"}), units={}
    )
    renderer.validate_frame(frame)
    figure = renderer.build_static_figure(frame)
    assert figure is not None

    axis = figure.axes[0]
    assert len(axis.lines) == 1
    line = axis.lines[0]
    np.testing.assert_allclose(line.get_xdata(), frame["x"].to_numpy())
    np.testing.assert_allclose(line.get_ydata(), frame["y"].to_numpy())

    plotly_figure = renderer.build_plotly_figure(frame)
    assert plotly_figure is not None
    assert len(plotly_figure.data) == 1
    assert len(plotly_figure.data[0].x) == frame.height
    plt.close(figure)


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
