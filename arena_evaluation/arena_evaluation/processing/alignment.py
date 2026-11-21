"""Time alignment of recorded streams: the one owner of how a sample is matched to a query time."""

from __future__ import annotations

import typing

import numpy as np
import polars as pl

TRACK_ALIGN_TOLERANCE_NS: typing.Final[int] = 100_000_000

_FrameT = typing.TypeVar("_FrameT", pl.DataFrame, pl.LazyFrame)


def asof_index(
    source_t_ns: np.ndarray, query_t_ns: np.ndarray, tolerance_ns: int | None = None
) -> np.ndarray:
    """Per query, the index of the last source sample at or before it; -1 where there is none, or where it is more than tolerance_ns older (no bound when None)."""
    source = np.asarray(source_t_ns, dtype=np.int64)
    query = np.asarray(query_t_ns, dtype=np.int64)
    if source.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    index = np.searchsorted(source, query, side="right").astype(np.int64) - 1
    if tolerance_ns is None:
        return index
    held = source[np.maximum(index, 0)]
    stale = (index < 0) | (query - held > tolerance_ns)
    return np.where(stale, -1, index)


def asof_position(
    source_t_ns: np.ndarray, query_t_ns: int, tolerance_ns: int | None = None
) -> int:
    """The asof_index of one query time."""
    return int(
        asof_index(source_t_ns, np.asarray([query_t_ns], dtype=np.int64), tolerance_ns)[
            0
        ]
    )


def asof_values(
    source_t_ns: np.ndarray,
    source_values: np.ndarray,
    query_t_ns: np.ndarray,
    tolerance_ns: int | None = None,
) -> np.ndarray:
    """The source value held at each query time, NaN where asof_index finds no sample."""
    index = asof_index(source_t_ns, query_t_ns, tolerance_ns)
    values = np.asarray(source_values, dtype=np.float64)
    if values.size == 0:
        return np.full(index.shape, np.nan)
    return np.where(index >= 0, values[np.maximum(index, 0)], np.nan)


def interpolate_values(
    source_t_ns: np.ndarray, source_values: np.ndarray, query_t_ns: np.ndarray
) -> np.ndarray:
    """Linear interpolation of the source at each query time; beyond the source span the nearest end value is held."""
    return np.interp(query_t_ns, source_t_ns, source_values)


def asof_join(
    query: _FrameT,
    source: _FrameT,
    *,
    tolerance_ns: int | None,
    on: str = "t_ns",
    left_on: str | None = None,
    right_on: str | None = None,
) -> _FrameT:
    """Each query row joined to the last source row at or before its time, null where none lies within tolerance_ns (no bound when None)."""
    keys = (
        {"on": on}
        if left_on is None
        else {"left_on": left_on, "right_on": right_on or on}
    )
    return query.join_asof(source, strategy="backward", tolerance=tolerance_ns, **keys)
