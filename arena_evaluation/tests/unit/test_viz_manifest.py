"""Unit tests for VizManifest YAML parsing/round-trip with the declarative fields."""

import pathlib
import pytest
import yaml

from arena_evaluation.presentation.manifest_registry import find_manifest_file, available_manifests
from arena_evaluation.presentation.viz_manifest import VizManifest


@pytest.mark.parametrize("manifest_name", ["standard", "safety", "social", "ecological", "characterization", "everything", "failure_modes"])
def test_all_bundled_manifests_load_and_roundtrip(manifest_name: str):
    """Test that all bundled manifests exist, validate schema, have unique IDs, and roundtrip cleanly."""
    p = find_manifest_file(manifest_name)
    assert p is not None, f"Manifest {manifest_name}.yaml must be bundled"

    manifest = VizManifest.load(p)
    assert manifest.name == manifest_name
    assert manifest.data_source in ("metrics", "characterization_samples")
    assert manifest.units, f"Manifest {manifest_name} must declare units"
    assert manifest.groups, f"Manifest {manifest_name} must declare layout groups"
    assert len(manifest.plots) > 0, f"Manifest {manifest_name} must have plots"

    # All plot specs must have unique IDs
    ids = [spec.id for spec in manifest.plots]
    assert len(ids) == len(set(ids)), f"Duplicate plot IDs found in {manifest_name}: {[x for x in ids if ids.count(x) > 1]}"

    # YAML round-trip: dump -> re-validate
    dumped = yaml.safe_dump(manifest.model_dump(exclude_none=True), sort_keys=False)
    again = VizManifest.model_validate(yaml.safe_load(dumped))
    assert again.name == manifest.name
    assert len(again.plots) == len(manifest.plots)


def test_available_manifests_contains_all_manifests():
    """Verify registry discovers all bundled manifests including everything and failure_modes."""
    manifests = available_manifests()
    for expected in ["standard", "safety", "social", "ecological", "characterization", "everything", "failure_modes"]:
        assert expected in manifests, f"Expected manifest '{expected}' in available_manifests()"



def test_old_style_manifest_still_validates():
    manifest = VizManifest.model_validate({"plots": []})
    assert manifest.data_source == "metrics"
    assert manifest.groups == []
    assert manifest.summary == []
    assert manifest.units == {}


def test_characterization_manifest_loads():
    p = find_manifest_file("characterization")
    assert p is not None

    manifest = VizManifest.load(p)
    assert manifest.name == "characterization"
    assert manifest.data_source == "metrics"
    assert manifest.summary_group_by == ["timeseries_char_phase_kind", "timeseries_char_speed_target"]
    assert manifest.summary  # declarative summary table
    line_specs = [s for s in manifest.plots if s.type == "line"]
    assert line_specs, "characterization manifest must use line charts"
    # Curves aggregate per working point from the long per-sample frame.
    assert any(
        s.options.get("aggregate") for s in manifest.plots
    )
    assert manifest.units.get("timeseries_char_power_total_w") == "W"
    assert manifest.units.get("timeseries_char_dba") == "dBA"

