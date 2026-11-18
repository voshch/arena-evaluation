from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.metrics.common.alignment import (
    asof_join,
    asof_position,
    asof_values,
    interpolate_values,
)

SOURCE = np.array([10, 20, 30], dtype=np.int64)


@pytest.mark.parametrize(
    ("query", "tolerance", "expected"),
    [
        (5, None, -1),
        (10, None, 0),
        (25, None, 1),
        (45, None, 2),
        (45, 14, -1),
        (45, 15, 2),
    ],
)
def test_asof_holds_the_last_sample_at_or_before_within_the_tolerance(
    query: int, tolerance: int | None, expected: int
) -> None:
    assert asof_position(SOURCE, query, tolerance) == expected


def test_asof_values_are_nan_where_no_sample_is_held() -> None:
    result = asof_values(
        SOURCE,
        np.array([1.0, 2.0, 3.0]),
        np.array([5, 25, 45], dtype=np.int64),
        tolerance_ns=14,
    )

    np.testing.assert_array_equal(np.isnan(result), [True, False, True])
    assert result[1] == 2.0


def test_interpolation_is_linear_and_holds_the_end_values() -> None:
    result = interpolate_values(
        SOURCE, np.array([0.0, 10.0, 0.0]), np.array([0, 15, 25, 99], dtype=np.int64)
    )

    np.testing.assert_allclose(result, [0.0, 5.0, 5.0, 0.0])


def test_asof_join_pairs_each_row_with_the_last_source_row_within_the_tolerance() -> (
    None
):
    query = pl.DataFrame({"t_ns": [5, 25, 45]})
    source = pl.DataFrame({"t_ns": [10, 20, 30], "value": [1.0, 2.0, 3.0]})

    joined = asof_join(query, source, tolerance_ns=14)

    assert joined.get_column("value").to_list() == [None, 2.0, None]
