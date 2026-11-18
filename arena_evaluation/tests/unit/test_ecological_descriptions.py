"""Every ecological calculator describes what it measures, in what unit, and when its value is NULL."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.dictionary import (
    metric_dictionary,
    shipped_calculators,
)

PLACEHOLDER = "ecological metric of the"
ECOLOGICAL_CALCULATORS = tuple(
    sorted(
        (
            calculator
            for calculator in shipped_calculators()
            if calculator.CATEGORY == "ecological"
        ),
        key=lambda calculator: calculator.NAME,
    )
)


@pytest.mark.parametrize(
    "calculator",
    ECOLOGICAL_CALCULATORS,
    ids=[calculator.NAME for calculator in ECOLOGICAL_CALCULATORS],
)
def test_an_ecological_metric_description_is_not_a_placeholder(calculator):
    metrics = metric_dictionary().filter(pl.col("producer") == calculator.NAME)
    offenders = (
        metrics.filter(pl.col("description").str.contains(PLACEHOLDER, literal=True))[
            "metric"
        ]
        .unique()
        .to_list()
    )

    assert offenders == [], (
        f"{calculator.NAME} still has placeholder descriptions for {offenders}"
    )
