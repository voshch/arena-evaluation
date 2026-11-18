"""Unit tests for the single VizManifest and PlotSpec declaration grammar."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from arena_evaluation.presentation.data_sources import (
    EPISODE_SUMMARY_VIEW,
    declared_data_sources,
    declared_view_names,
    validate_data_source,
)
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotDataSpec,
    PlotSpec,
    VizManifest,
)


def test_minimal_manifest_defaults_to_episode_summary_view() -> None:
    manifest = VizManifest.model_validate({"plots": []})
    assert manifest.data_source == EPISODE_SUMMARY_VIEW
    assert manifest.groups == []
    assert manifest.summary == []
    assert manifest.units == {}


def test_manifest_roundtrip_with_single_grammar() -> None:
    raw = {
        "manifest_version": "1.0",
        "name": "custom",
        "title": "Custom Report",
        "data_source": "v_episode_summary",
        "groups": [{"id": "perf", "title": "Performance"}],
        "summary": [{"metric": "time_to_goal", "label": "Time", "format": "{:.2f}"}],
        "summary_group_by": ["planner"],
        "units": {"time_to_goal": "s"},
        "plots": [
            {
                "id": "p_time",
                "type": "bar",
                "title": "Time to Goal",
                "layout_group": "perf",
                "data": {
                    "computation": "group_summary",
                    "params": {"metrics": ["time_to_goal"], "group_cols": ["planner"]},
                },
                "channels": {"x": "planner", "y": "mean", "color": "planner"},
                "checks": {"require_columns": ["planner", "mean"]},
            }
        ],
    }
    manifest = VizManifest.model_validate(raw)
    dumped = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    again = VizManifest.model_validate(yaml.safe_load(dumped))
    assert again == manifest


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


@pytest.mark.parametrize(
    ("legacy_field", "payload"),
    [
        ("data_key", {"id": "p", "type": "bar", "data_key": "time_to_goal"}),
        ("group_by", {"id": "p", "type": "bar", "group_by": ["planner"]}),
        ("differentiate", {"id": "p", "type": "bar", "differentiate": "planner"}),
        ("auto_differentiate", {"id": "p", "type": "bar", "auto_differentiate": False}),
        ("filter", {"id": "p", "type": "bar", "filter": {"planner": "dwb"}}),
        ("encodings", {"id": "p", "type": "bar", "encodings": {"x": "planner"}}),
        ("source", {"id": "p", "type": "bar", "source": "v_episode_summary"}),
        (
            "stored_query",
            {"id": "p", "type": "bar", "stored_query": "planner_scorecard_overview"},
        ),
        ("params", {"id": "p", "type": "bar", "params": {"metric": "time_to_goal"}}),
        ("data_source", {"id": "p", "type": "bar", "data_source": "v_episode_summary"}),
    ],
)
def test_plotspec_rejects_every_legacy_field(
    legacy_field: str, payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match=legacy_field):
        PlotSpec.model_validate(payload)


def test_plotspec_rejects_bare_string_data() -> None:
    with pytest.raises(ValidationError, match="data"):
        PlotSpec.model_validate({"id": "p", "type": "bar", "data": "v_episode_summary"})


@pytest.mark.parametrize("legacy_channel", ["hue", "facet"])
def test_plot_channels_rejects_legacy_aliases(legacy_channel: str) -> None:
    with pytest.raises(ValidationError, match=legacy_channel):
        PlotChannels.model_validate({legacy_channel: "planner"})


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


def test_all_bundled_manifests_load_cleanly() -> None:
    manifests_dir = pathlib.Path(__file__).resolve().parents[2] / "configs" / "reports"
    files = sorted(manifests_dir.glob("*/report.yaml"))
    assert len(files) == 7
    for path in files:
        manifest = VizManifest.load(path)
        assert len(manifest.plots) > 0
