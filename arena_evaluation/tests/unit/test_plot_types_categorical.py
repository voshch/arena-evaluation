"""Unit tests for the categorical plot types: bar, box, histogram, violin."""

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.plot_types.bar import BarRenderer
from arena_evaluation.presentation.plot_types.box import BoxRenderer
from arena_evaluation.presentation.plot_types.histogram import HistogramRenderer
from arena_evaluation.presentation.plot_types.violin import ViolinRenderer
from arena_evaluation.storage.schemas import PlotSpec

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "episode": [1, 2, 1, 2],
            "success": [1.0, 0.5, 0.75, 0.25],
            "path_efficiency": [0.8, 0.6, 0.7, 0.9],
        },
        schema={
            "planner": pl.Utf8,
            "episode": pl.Int64,
            "success": pl.Float64,
            "path_efficiency": pl.Float64,
        },
    )


def _spec(ptype: str, data_key: str = "success", **overrides) -> PlotSpec:
    base = dict(id=f"p_{ptype}", type=ptype, title=f"Title {ptype}", data_key=data_key, differentiate="planner")
    base.update(overrides)
    return PlotSpec(**base)


def _assert_png(path: pathlib.Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == _PNG_MAGIC


# ═══════════════════════════════════════════════════════════════════════════
# BarRenderer
# ═══════════════════════════════════════════════════════════════════════════


def test_bar_plotly_happy_path_with_units():
    html = BarRenderer(_spec("bar"), units={"success": "%"}).render_plotly(_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Success [%]" in html


def test_bar_plotly_missing_diff_col_returns_none():
    df = _df().drop("planner")
    assert BarRenderer(_spec("bar")).render_plotly(df) is None


def test_bar_plotly_missing_data_key_returns_none():
    assert BarRenderer(_spec("bar", data_key="bogus")).render_plotly(_df()) is None


def test_bar_plotly_stacked_happy_path():
    spec = _spec("bar", options={"stacked": True, "metrics": ["success", "path_efficiency"]})
    html = BarRenderer(spec).render_plotly(_df())
    assert html is not None
    assert "Percentage (%)" in html
    assert "Component" in html


def test_bar_plotly_stacked_no_metrics_returns_none():
    spec = _spec("bar", options={"stacked": True})
    assert BarRenderer(spec).render_plotly(_df()) is None


def test_bar_plotly_stacked_metrics_all_missing_returns_none():
    spec = _spec("bar", options={"stacked": True, "metrics": ["m1", "m2"]})
    assert BarRenderer(spec).render_plotly(_df()) is None


def test_bar_plotly_filtered_empty_returns_none():
    spec = _spec("bar", filter={"planner": ["nobody"]})
    assert BarRenderer(spec).render_plotly(_df()) is None


def test_bar_plotly_empty_df_returns_none():
    assert BarRenderer(_spec("bar")).render_plotly(pl.DataFrame()) is None


def test_bar_seaborn_happy_path(tmp_path):
    out = tmp_path / "bar.png"
    BarRenderer(_spec("bar")).render_seaborn(_df(), out)
    _assert_png(out)


def test_bar_seaborn_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "bar.png"
    df = _df().drop("planner")
    BarRenderer(_spec("bar")).render_seaborn(df, out)
    assert not out.exists()


def test_bar_seaborn_empty_df_writes_nothing(tmp_path):
    out = tmp_path / "bar.png"
    BarRenderer(_spec("bar")).render_seaborn(pl.DataFrame(), out)
    assert not out.exists()


def test_bar_seaborn_filtered_empty_writes_nothing(tmp_path):
    out = tmp_path / "bar.png"
    spec = _spec("bar", filter={"planner": ["nobody"]})
    BarRenderer(spec).render_seaborn(_df(), out)
    assert not out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# BoxRenderer
# ═══════════════════════════════════════════════════════════════════════════


def test_box_plotly_happy_path_with_units():
    html = BoxRenderer(_spec("box"), units={"success": "%"}).render_plotly(_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Success [%]" in html


def test_box_plotly_missing_data_key_returns_none():
    assert BoxRenderer(_spec("box", data_key="bogus")).render_plotly(_df()) is None


def test_box_plotly_missing_diff_col_returns_none():
    df = _df().drop("planner")
    assert BoxRenderer(_spec("box")).render_plotly(df) is None


def test_box_plotly_empty_df_returns_none():
    assert BoxRenderer(_spec("box")).render_plotly(pl.DataFrame()) is None


def test_box_seaborn_happy_path(tmp_path):
    out = tmp_path / "box.png"
    BoxRenderer(_spec("box")).render_seaborn(_df(), out)
    _assert_png(out)


def test_box_seaborn_missing_data_key_writes_nothing(tmp_path):
    out = tmp_path / "box.png"
    BoxRenderer(_spec("box", data_key="bogus")).render_seaborn(_df(), out)
    assert not out.exists()


def test_box_seaborn_all_nan_data_writes_nothing(tmp_path):
    out = tmp_path / "box.png"
    df = pl.DataFrame(
        {"planner": ["dwb", "teb"], "success": [None, None]},
        schema={"planner": pl.Utf8, "success": pl.Float64},
    )
    BoxRenderer(_spec("box")).render_seaborn(df, out)
    assert not out.exists()


def test_box_seaborn_empty_df_writes_nothing(tmp_path):
    out = tmp_path / "box.png"
    BoxRenderer(_spec("box")).render_seaborn(pl.DataFrame(), out)
    assert not out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# HistogramRenderer
# ═══════════════════════════════════════════════════════════════════════════


def _hist_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb"] * 6 + ["teb"] * 4,
            "time_to_goal": [0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3, 0.7, 0.9, 0.1],
        },
        schema={"planner": pl.Utf8, "time_to_goal": pl.Float64},
    )


def test_histogram_plotly_happy_path():
    html = HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_plotly(_hist_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Count" in html


def test_histogram_plotly_nbins_and_opacity_options():
    spec = _spec("histogram", data_key="time_to_goal", options={"nbins": 4, "opacity": 0.9})
    html = HistogramRenderer(spec).render_plotly(_hist_df())
    assert html is not None
    assert "0.5" in html  # first bin centre of the 4-bin grid


def test_histogram_plotly_single_value_column_adjusts_range():
    df = pl.DataFrame(
        {"planner": ["dwb", "teb"], "time_to_goal": [1.0, 1.0]},
        schema={"planner": pl.Utf8, "time_to_goal": pl.Float64},
    )
    html = HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_plotly(df)
    assert html is not None


def test_histogram_plotly_missing_x_col_returns_none():
    assert HistogramRenderer(_spec("histogram", data_key="bogus")).render_plotly(_hist_df()) is None


def test_histogram_plotly_missing_diff_col_returns_none():
    df = _hist_df().drop("planner")
    assert HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_plotly(df) is None


def test_histogram_plotly_all_nan_returns_none():
    df = pl.DataFrame(
        {"planner": ["dwb"], "time_to_goal": [None]},
        schema={"planner": pl.Utf8, "time_to_goal": pl.Float64},
    )
    assert HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_plotly(df) is None


def test_histogram_seaborn_happy_path(tmp_path):
    out = tmp_path / "hist.png"
    HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_seaborn(_hist_df(), out)
    _assert_png(out)


def test_histogram_seaborn_single_value_adjusts_range(tmp_path):
    out = tmp_path / "hist.png"
    df = pl.DataFrame(
        {"planner": ["dwb", "teb"], "time_to_goal": [1.0, 1.0]},
        schema={"planner": pl.Utf8, "time_to_goal": pl.Float64},
    )
    HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_seaborn(df, out)
    _assert_png(out)


def test_histogram_seaborn_missing_x_writes_nothing(tmp_path):
    out = tmp_path / "hist.png"
    HistogramRenderer(_spec("histogram", data_key="bogus")).render_seaborn(_hist_df(), out)
    assert not out.exists()


def test_histogram_seaborn_all_nan_writes_nothing(tmp_path):
    out = tmp_path / "hist.png"
    df = pl.DataFrame(
        {"planner": ["dwb"], "time_to_goal": [None]},
        schema={"planner": pl.Utf8, "time_to_goal": pl.Float64},
    )
    HistogramRenderer(_spec("histogram", data_key="time_to_goal")).render_seaborn(df, out)
    assert not out.exists()


# ═══════════════════════════════════════════════════════════════════════════
# ViolinRenderer
# ═══════════════════════════════════════════════════════════════════════════


def test_violin_plotly_happy_path_with_units():
    html = ViolinRenderer(_spec("violin"), units={"success": "%"}).render_plotly(_df())
    assert html is not None
    assert html.startswith("<div")
    assert "Success [%]" in html


def test_violin_plotly_missing_data_key_returns_none():
    assert ViolinRenderer(_spec("violin", data_key="bogus")).render_plotly(_df()) is None


def test_violin_plotly_missing_diff_col_returns_none():
    df = _df().drop("planner")
    assert ViolinRenderer(_spec("violin")).render_plotly(df) is None


def test_violin_plotly_empty_df_returns_none():
    assert ViolinRenderer(_spec("violin")).render_plotly(pl.DataFrame()) is None


def test_violin_seaborn_happy_path(tmp_path):
    out = tmp_path / "violin.png"
    ViolinRenderer(_spec("violin")).render_seaborn(_df(), out)
    _assert_png(out)


def test_violin_seaborn_missing_diff_writes_nothing(tmp_path):
    out = tmp_path / "violin.png"
    df = _df().drop("planner")
    ViolinRenderer(_spec("violin")).render_seaborn(df, out)
    assert not out.exists()


def test_violin_seaborn_empty_df_writes_nothing(tmp_path):
    out = tmp_path / "violin.png"
    ViolinRenderer(_spec("violin")).render_seaborn(pl.DataFrame(), out)
    assert not out.exists()
