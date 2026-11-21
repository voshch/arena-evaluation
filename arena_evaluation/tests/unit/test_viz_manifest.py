"""Unit tests for the single VizManifest and PlotSpec declaration grammar."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from arena_evaluation.presentation.data_sources import (
    declared_data_sources,
    declared_view_names,
    validate_data_source,
)
from arena_evaluation.presentation.viz_manifest import (
    PlotDataSpec,
    PlotSpec,
    VizManifest,
)


@pytest.mark.parametrize(
    "legacy_source",
    ["metrics", "episode_summary", "v_unknown_view", "no_such_table", ""],
)
def test_manifest_rejects_legacy_and_unknown_data_sources(legacy_source: str) -> None:
    with pytest.raises((ValidationError, ValueError), match="data_source"):
        VizManifest.model_validate({"data_source": legacy_source, "plots": []})
    with pytest.raises(ValueError, match="data_source"):
        validate_data_source(legacy_source)


def test_manifest_rejects_legacy_sections_field() -> None:
    with pytest.raises(ValidationError, match="sections"):
        VizManifest.model_validate({"sections": [{"id": "s1", "plots": []}]})


def test_plotspec_rejects_bare_string_data() -> None:
    with pytest.raises(ValidationError, match="data"):
        PlotSpec.model_validate({"id": "p", "type": "bar", "data": "v_episode_summary"})


@pytest.mark.parametrize("filename", ["viz_manifest.yaml", "manifest.yaml"])
def test_manifest_load_rejects_legacy_filenames(
    tmp_path: pathlib.Path, filename: str
) -> None:
    legacy_path = tmp_path / filename
    legacy_path.write_text("plots: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match=filename):
        VizManifest.load(legacy_path)


def test_plot_data_spec_rejects_multiple_bindings_and_orphan_params() -> None:
    with pytest.raises(ValidationError, match="at most one data binding"):
        PlotDataSpec(source="v_episode_summary", query="SELECT 1")
    with pytest.raises(ValidationError, match="params"):
        PlotDataSpec(source="v_episode_summary", params={"k": 1})


def test_declared_data_sources_include_views_and_tables() -> None:
    assert "v_episode_summary" in declared_view_names()
    assert "v_series_characterization" in declared_view_names()
    assert "fact_robots" in declared_data_sources()
    assert "dim_world_geometry" in declared_data_sources()


def test_plot_spec_requires_type_or_uses() -> None:
    with pytest.raises(ValidationError, match="type.*uses|uses.*type"):
        PlotSpec(id="missing_type")
    spec = PlotSpec(id="from_recipe", uses="scorecard_table")
    assert spec.uses == "scorecard_table"
