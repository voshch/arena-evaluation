from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING

from arena_evaluation.storage.planner_names import split_planner_name

if TYPE_CHECKING:
    from arena_evaluation.storage.schemas import PlotSpec

# Ordered priority list of identity columns
IDENTITY_COLS: list[str] = ["local_planner", "inter_planner", "robot", "stage", "map", "benchmark_id"]

COMPOUND_LABEL_COL = "__label__"



def detect_varying_dims(df: pl.DataFrame) -> list[str]:
    """Return identity columns that have more than one unique value in df."""
    varying: list[str] = []
    for col in IDENTITY_COLS:
        if col not in df.columns:
            continue
        series = df[col].drop_nulls()
        if series.n_unique() > 1:
            varying.append(col)
    return varying


def build_label_column(df: pl.DataFrame, dims: list[str]) -> pl.DataFrame:
    """Add (or replace) a __label__ column holding a readable compound of dims."""
    if not dims:
        return df

    if len(dims) == 1:
        return df.with_columns(pl.col(dims[0]).alias(COMPOUND_LABEL_COL))

    parts = [pl.col(d).cast(pl.Utf8) for d in dims]
    label_expr = pl.concat_str(parts, separator=" / ")
    return df.with_columns(label_expr.alias(COMPOUND_LABEL_COL))


def resolve_differentiate(spec: "PlotSpec", df: pl.DataFrame) -> tuple[str, pl.DataFrame]:
    """Determine the effective differentiation column for spec given df."""
    auto = spec.auto_differentiate
    requested = spec.differentiate

    if not auto:
        fallback = requested if (requested and requested in df.columns) else "planner"
        return fallback, df

    varying = detect_varying_dims(df)

    if len(varying) <= 1:
        col = requested if (requested and requested in df.columns) else (varying[0] if varying else "planner")
        return col, df

    df = build_label_column(df, varying)
    return COMPOUND_LABEL_COL, df
