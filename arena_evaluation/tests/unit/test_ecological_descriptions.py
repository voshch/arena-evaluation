"""Every ecological calculator describes what it measures, in what unit, and when its value is NULL."""

from __future__ import annotations


from arena_evaluation.processing.metrics.dictionary import (
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
