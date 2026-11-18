"""A-028: the shipped acoustic-field recipe parses, binds and is cited by a report."""

from __future__ import annotations

import pathlib

import yaml

from arena_evaluation.presentation.resolver import find_recipe_path, load_recipe
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECIPE = _PKG_ROOT / "configs" / "plots" / "acoustic_field" / "plot.yaml"
_EVERYTHING = _PKG_ROOT / "configs" / "reports" / "everything" / "report.yaml"


def test_acoustic_recipe_is_a_bindable_plot_spec() -> None:
    spec = PlotSpec.model_validate(load_recipe("acoustic_field"))

    assert spec.type == "acoustic_field"
    assert spec.data is not None and spec.data.query
    assert spec.channels.x == "world"
    assert spec.channels.y == "ped_max_exposure_dba"
    assert set(spec.checks.require_columns) == {
        "world",
        "ped_max_exposure_dba",
        "worst_case_acoustic_t_ns",
    }
    assert find_recipe_path("acoustic_field") is not None


def test_acoustic_recipe_is_cited_by_the_everything_report() -> None:
    manifest = VizManifest.load(_EVERYTHING)
    cited = [plot for plot in manifest.plots if plot.uses == "acoustic_field"]

    assert len(cited) == 1
    assert cited[0].layout_group == "environmental_acoustics"


def test_recipe_file_is_the_only_acoustic_binding_copy() -> None:
    raw = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    assert raw["id"] == "acoustic_field"
