"""Unit tests for auto-detection of varying identity dimensions.

``detect_varying_dims`` drives compound axis labels: when more than one
identity column varies in the metrics frame, the renderers differentiate on a
compound ``__label__`` column instead of a single planner column.
"""

import types

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from arena_evaluation.presentation.dimension_detector import (
    COMPOUND_LABEL_COL,
    IDENTITY_COLS,
    build_label_column,
    detect_varying_dims,
    resolve_differentiate,
)
from arena_evaluation.storage.schemas import PlotSpec


# ── Helpers ────────────────────────────────────────────────────────────────


def _identity_frame(n_varying: int) -> pl.DataFrame:
    """Frame over all identity columns; the first ``n_varying`` columns vary."""
    data = {}
    for i, col in enumerate(IDENTITY_COLS):
        data[col] = ["a", "b"] if i < n_varying else ["const", "const"]
    data["success"] = [1.0, 0.5]
    return pl.DataFrame(data)


def _spec(differentiate="planner", auto_differentiate=True) -> PlotSpec:
    return PlotSpec(
        id="t",
        type="bar",
        title="T",
        data_key="success",
        differentiate=differentiate,
        auto_differentiate=auto_differentiate,
    )


# ── detect_varying_dims ────────────────────────────────────────────────────


def test_detect_all_identity_columns_varying():
    df = _identity_frame(n_varying=len(IDENTITY_COLS))
    assert detect_varying_dims(df) == IDENTITY_COLS


def test_detect_none_varying():
    df = _identity_frame(n_varying=0)
    assert detect_varying_dims(df) == []


def test_detect_subset_varying():
    df = _identity_frame(n_varying=2)
    assert detect_varying_dims(df) == IDENTITY_COLS[:2]


def test_detect_missing_columns_skipped():
    df = pl.DataFrame({"local_planner": ["a", "b"], "success": [1.0, 0.5]})
    assert detect_varying_dims(df) == ["local_planner"]


def test_detect_ignores_constant_null_column():
    df = pl.DataFrame(
        {"robot": [None, None, "r1"], "success": [1.0, 0.5, 0.25]},
        schema={"robot": pl.Utf8, "success": pl.Float64},
    )
    # After drop_nulls there is a single value -> constant.
    assert detect_varying_dims(df) == []


def test_detect_null_and_value_counts_as_varying():
    df = pl.DataFrame(
        {"robot": [None, "r1", "r2"], "success": [1.0, 0.5, 0.25]},
        schema={"robot": pl.Utf8, "success": pl.Float64},
    )
    assert detect_varying_dims(df) == ["robot"]


def test_detect_empty_frame():
    assert detect_varying_dims(pl.DataFrame()) == []


@given(
    varying=st.lists(st.sampled_from(IDENTITY_COLS), unique=True, max_size=6),
    constant=st.lists(st.sampled_from(IDENTITY_COLS), unique=True, max_size=6),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_detect_varying_dims_property(varying, constant):
    """Property: exactly the present columns with >1 non-null value vary."""
    include = list(dict.fromkeys(varying + constant))
    data = {c: (["a", "b"] if c in varying else ["x", "x"]) for c in include}
    df = pl.DataFrame(data)
    assert set(detect_varying_dims(df)) == set(varying)


# ── build_label_column ─────────────────────────────────────────────────────


def test_build_label_no_dims_returns_same_frame():
    df = _identity_frame(n_varying=0)
    out = build_label_column(df, [])
    assert out is df
    assert COMPOUND_LABEL_COL not in out.columns


def test_build_label_single_dim_aliases_column():
    df = _identity_frame(n_varying=1)
    out = build_label_column(df, ["local_planner"])
    assert out[COMPOUND_LABEL_COL].to_list() == df["local_planner"].to_list()


def test_build_label_compound_joins_with_separator():
    df = _identity_frame(n_varying=2)
    out = build_label_column(df, IDENTITY_COLS[:2])
    assert out[COMPOUND_LABEL_COL].to_list() == ["a / a", "b / b"]


def test_build_label_compound_three_dims():
    df = _identity_frame(n_varying=3)
    out = build_label_column(df, IDENTITY_COLS[:3])
    assert out[COMPOUND_LABEL_COL].to_list() == ["a / a / a", "b / b / b"]


def test_build_label_mixed_types_cast_to_string():
    df = pl.DataFrame(
        {"local_planner": ["dwb", "teb"], "stage": [1, 2]},
        schema={"local_planner": pl.Utf8, "stage": pl.Int64},
    )
    out = build_label_column(df, ["local_planner", "stage"])
    assert out[COMPOUND_LABEL_COL].to_list() == ["dwb / 1", "teb / 2"]


def test_build_label_overwrites_existing_label_column():
    df = pl.DataFrame(
        {COMPOUND_LABEL_COL: ["old", "old"], "local_planner": ["dwb", "teb"]},
        schema={COMPOUND_LABEL_COL: pl.Utf8, "local_planner": pl.Utf8},
    )
    out = build_label_column(df, ["local_planner"])
    assert out[COMPOUND_LABEL_COL].to_list() == ["dwb", "teb"]


# ── resolve_differentiate: auto=False ──────────────────────────────────────


def test_resolve_auto_false_uses_requested_in_columns():
    df = _identity_frame(n_varying=3)
    col, out = resolve_differentiate(_spec(differentiate="robot", auto_differentiate=False), df)
    assert col == "robot"
    assert out is df


def test_resolve_auto_false_requested_missing_falls_back_to_planner():
    df = _identity_frame(n_varying=3)
    col, _ = resolve_differentiate(_spec(differentiate="bogus", auto_differentiate=False), df)
    assert col == "planner"


def test_resolve_auto_false_requested_none_falls_back_to_planner():
    df = _identity_frame(n_varying=3)
    col, _ = resolve_differentiate(_spec(differentiate=None, auto_differentiate=False), df)
    assert col == "planner"


def test_resolve_auto_false_empty_requested_falls_back():
    df = _identity_frame(n_varying=3)
    col, _ = resolve_differentiate(_spec(differentiate="", auto_differentiate=False), df)
    assert col == "planner"


# ── resolve_differentiate: auto=True ───────────────────────────────────────


def test_resolve_auto_no_varying_requested_present():
    df = _identity_frame(n_varying=0)
    col, _ = resolve_differentiate(_spec(differentiate="robot"), df)
    assert col == "robot"


def test_resolve_auto_no_varying_requested_missing_falls_back_to_planner():
    df = _identity_frame(n_varying=0)
    col, _ = resolve_differentiate(_spec(differentiate="bogus"), df)
    assert col == "planner"


def test_resolve_auto_no_varying_no_requested_falls_back_to_planner():
    df = pl.DataFrame({"success": [1.0, 0.5]})
    col, _ = resolve_differentiate(_spec(differentiate=None), df)
    assert col == "planner"


def test_resolve_auto_single_varying_prefers_requested():
    df = _identity_frame(n_varying=1)
    col, _ = resolve_differentiate(_spec(differentiate="inter_planner"), df)
    assert col == "inter_planner"


def test_resolve_auto_single_varying_uses_varying_when_requested_missing():
    df = _identity_frame(n_varying=1)
    col, _ = resolve_differentiate(_spec(differentiate="bogus"), df)
    assert col == "local_planner"


def test_resolve_auto_two_varying_builds_compound_label():
    df = _identity_frame(n_varying=2)
    col, out = resolve_differentiate(_spec(), df)
    assert col == COMPOUND_LABEL_COL
    assert COMPOUND_LABEL_COL in out.columns
    assert out[COMPOUND_LABEL_COL].to_list() == ["a / a", "b / b"]


def test_resolve_auto_three_varying_compound():
    df = _identity_frame(n_varying=3)
    col, out = resolve_differentiate(_spec(), df)
    assert col == COMPOUND_LABEL_COL
    assert out[COMPOUND_LABEL_COL].to_list() == ["a / a / a", "b / b / b"]


def test_resolve_compound_overrides_requested_column():
    df = _identity_frame(n_varying=2)
    col, _ = resolve_differentiate(_spec(differentiate="robot"), df)
    assert col == COMPOUND_LABEL_COL


def test_resolve_default_auto_differentiate_true():
    """PlotSpec defaults auto_differentiate to True."""
    df = _identity_frame(n_varying=2)
    spec = _spec(differentiate="planner")
    col, out = resolve_differentiate(spec, df)
    assert col == COMPOUND_LABEL_COL
    assert COMPOUND_LABEL_COL in out.columns


@given(
    n_varying=st.integers(0, len(IDENTITY_COLS)),
    requested=st.one_of(st.none(), st.sampled_from(IDENTITY_COLS + ["planner", "bogus"])),
    auto=st.booleans(),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_resolve_differentiate_property(n_varying, requested, auto):
    """Property: result column is the manifest-preferred col unless >1 dims
    vary (then compound), and must always be present in the output frame."""
    df = _identity_frame(n_varying=n_varying)
    col, out = resolve_differentiate(_spec(differentiate=requested, auto_differentiate=auto), df)
    varying = detect_varying_dims(df)
    if auto and len(varying) > 1:
        assert col == COMPOUND_LABEL_COL
        assert COMPOUND_LABEL_COL in out.columns
    elif auto and len(varying) <= 1:
        expected = requested if (requested and requested in df.columns) else (varying[0] if varying else "planner")
        assert col == expected
    else:
        expected = requested if (requested and requested in df.columns) else "planner"
        assert col == expected
    # The "planner" fallback may not physically exist in a frame without it.
    assert col in out.columns or col == "planner"
