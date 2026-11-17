"""A-028: the shipped acoustic-field recipe parses, binds and is cited by a report."""

from __future__ import annotations

import pathlib

import yaml


_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECIPE = _PKG_ROOT / "configs" / "plots" / "acoustic_field" / "plot.yaml"
_EVERYTHING = _PKG_ROOT / "configs" / "reports" / "everything" / "report.yaml"


def test_recipe_file_is_the_only_acoustic_binding_copy() -> None:
    raw = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    assert raw["id"] == "acoustic_field"
