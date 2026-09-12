"""Unit tests for the abstract plot renderer base class.

Covers ``format_label`` (unit suffixes), list-valued manifest filters
(membership vs scalar equality), and the ``resolve_diff_col`` delegation to
the dimension detector.
"""

import pathlib

import polars as pl
import pytest

from arena_evaluation.presentation.dimension_detector import COMPOUND_LABEL_COL
from arena_evaluation.presentation.plot_types.base import BasePlotRenderer
from arena_evaluation.storage.schemas import PlotSpec


class _ConcreteRenderer(BasePlotRenderer):
    PLOT_TYPE = "concrete"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        return "<div>concrete</div>"

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        out_path.write_bytes(b"png")


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "planner": ["dwb", "teb", "dwb", "mppi"],
            "stage": ["a", "b", "b", "a"],
            "success": [1.0, 0.5, 0.25, 0.1],
        },
        schema={"planner": pl.Utf8, "stage": pl.Utf8, "success": pl.Float64},
    )


def _spec(**overrides) -> PlotSpec:
    base = dict(id="t", type="concrete", title="T", data_key="success", differentiate="planner")
    base.update(overrides)
    return PlotSpec(**base)


# ── abstractness / construction ────────────────────────────────────────────


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BasePlotRenderer(_spec())


def test_plot_type_default_empty():
    assert _ConcreteRenderer.PLOT_TYPE == "concrete"


def test_init_stores_spec_and_defaults():
    renderer = _ConcreteRenderer(_spec())
    assert renderer.spec.title == "T"
    assert renderer.units == {}
    assert renderer.run_dir is None


def test_init_units_none_becomes_empty_dict():
    assert _ConcreteRenderer(_spec(), units=None).units == {}


def test_init_units_preserved():
    renderer = _ConcreteRenderer(_spec(), units={"success": "%"})
    assert renderer.units == {"success": "%"}


# ── format_label ───────────────────────────────────────────────────────────


def test_format_label_with_unit_appends_bracket():
    renderer = _ConcreteRenderer(_spec(), units={"time_to_goal": "s"})
    assert renderer.format_label("Time To Goal", "time_to_goal") == "Time To Goal [s]"


def test_format_label_without_unit_unchanged():
    renderer = _ConcreteRenderer(_spec(), units={"other": "m"})
    assert renderer.format_label("Success", "success") == "Success"


def test_format_label_unknown_key_unchanged():
    renderer = _ConcreteRenderer(_spec())
    assert renderer.format_label("Success", "success") == "Success"


def test_format_label_empty_unit_unchanged():
    renderer = _ConcreteRenderer(_spec(), units={"success": ""})
    assert renderer.format_label("Success", "success") == "Success"


# ── resolve_diff_col delegation ────────────────────────────────────────────


def test_resolve_diff_col_compound_when_two_dims_vary():
    df = pl.DataFrame(
        {
            "local_planner": ["dwb", "teb", "dwb", "teb"],
            "stage": ["a", "a", "b", "b"],
            "success": [1.0, 0.5, 0.25, 0.1],
        },
        schema={"local_planner": pl.Utf8, "stage": pl.Utf8, "success": pl.Float64},
    )
    col, out = _ConcreteRenderer(_spec()).resolve_diff_col(df)
    assert col == COMPOUND_LABEL_COL
    assert COMPOUND_LABEL_COL in out.columns


def test_resolve_diff_col_single_varying_dim():
    df = pl.DataFrame(
        {"local_planner": ["dwb", "teb"], "success": [1.0, 0.5]},
        schema={"local_planner": pl.Utf8, "success": pl.Float64},
    )
    col, out = _ConcreteRenderer(_spec()).resolve_diff_col(df)
    assert col == "local_planner"
    assert out is df


# ── _apply_filters ─────────────────────────────────────────────────────────


def test_apply_filters_no_filter_returns_same_frame():
    df = _df()
    renderer = _ConcreteRenderer(_spec(filter=None))
    out = renderer._apply_filters(df)
    assert out is df


def test_apply_filters_empty_filter_dict_returns_same_frame():
    df = _df()
    renderer = _ConcreteRenderer(_spec(filter={}))
    out = renderer._apply_filters(df)
    assert out is df


def test_apply_filters_scalar_matches_equality():
    renderer = _ConcreteRenderer(_spec(filter={"planner": "dwb"}))
    out = renderer._apply_filters(_df())
    assert out["planner"].to_list() == ["dwb", "dwb"]
    assert out["success"].to_list() == [1.0, 0.25]


def test_apply_filters_list_matches_membership():
    renderer = _ConcreteRenderer(_spec(filter={"planner": ["dwb", "teb"]}))
    out = renderer._apply_filters(_df())
    assert out["planner"].to_list() == ["dwb", "teb", "dwb"]


def test_apply_filters_tuple_and_set_match_membership():
    renderer = _ConcreteRenderer(_spec(filter={"planner": ("dwb", "teb")}))
    out = renderer._apply_filters(_df())
    assert out.height == 3
    renderer = _ConcreteRenderer(_spec(filter={"planner": {"dwb"}}))
    out = renderer._apply_filters(_df())
    assert out["planner"].to_list() == ["dwb", "dwb"]


def test_apply_filters_unknown_key_is_ignored():
    renderer = _ConcreteRenderer(_spec(filter={"bogus_col": 1}))
    out = renderer._apply_filters(_df())
    assert out.height == 4


def test_apply_filters_multiple_keys_combine_with_and():
    renderer = _ConcreteRenderer(_spec(filter={"planner": ["dwb", "teb"], "stage": "b"}))
    out = renderer._apply_filters(_df())
    assert out.to_dict(as_series=False) == {
        "planner": ["teb", "dwb"],
        "stage": ["b", "b"],
        "success": [0.5, 0.25],
    }


def test_apply_filters_empty_list_matches_nothing():
    renderer = _ConcreteRenderer(_spec(filter={"planner": []}))
    out = renderer._apply_filters(_df())
    assert out.height == 0


def test_apply_filters_scalar_and_list_mixed():
    renderer = _ConcreteRenderer(_spec(filter={"stage": "a", "planner": ["dwb", "mppi"]}))
    out = renderer._apply_filters(_df())
    assert out["planner"].to_list() == ["dwb", "mppi"]
