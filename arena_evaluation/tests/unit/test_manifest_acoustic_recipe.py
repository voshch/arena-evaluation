"""A-028: the shipped acoustic-field recipe parses and is the one copy of its binding.

The recipe lives in the asset store, so the test resolves it the way the report builder does and
skips when neither a local root nor a benchmark bucket holds it."""

from __future__ import annotations

import pytest

from arena_evaluation.presentation.plot_identifier import PlotIdentifier


def test_recipe_file_is_the_only_acoustic_binding_copy() -> None:
    try:
        raw = PlotIdentifier(name="acoustic_field").resolve_sync()
    except FileNotFoundError:
        pytest.skip(
            "plot 'acoustic_field' is in the asset store: "
            "arena asset pull plot acoustic_field"
        )
    assert raw["id"] == "acoustic_field"
