"""Unit tests for declarative report-manifest resolution."""

import pathlib

import pytest

from arena_evaluation.benchmark.tree import ConfigPathResolver, ManifestIdentifier
from arena_evaluation.presentation import manifest_registry as registry
from arena_evaluation.presentation.viz_manifest import VizManifest


@pytest.fixture
def manifests_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A real configs/reports bundle tree, registered as ManifestIdentifier's only resolver for the test."""
    cfg_dir = tmp_path / "configs"
    d = cfg_dir / "reports"
    (d / "foo").mkdir(parents=True)
    (d / "foo" / "report.yaml").write_text(
        "manifest_version: '1.0'\nname: foo\ntitle: Foo Report\ndata_source: v_episode_summary\nplots: []\n"
    )
    (d / "standard").mkdir(parents=True)
    (d / "standard" / "report.yaml").write_text(
        "manifest_version: '1.0'\nname: standard\nplots: []\n"
    )
    registered = ManifestIdentifier._resolvers
    ManifestIdentifier._resolvers = [ConfigPathResolver(ManifestIdentifier, cfg_dir)]
    try:
        yield d
    finally:
        ManifestIdentifier._resolvers = registered


def test_resolve_by_name(manifests_dir):
    m = registry.resolve_manifest("foo")
    assert isinstance(m, VizManifest)


def test_resolve_by_name_with_yaml_suffix(manifests_dir):
    m = registry.resolve_manifest("foo.yaml")
    assert isinstance(m, VizManifest)


def test_resolve_inline_dict():
    m = registry.resolve_manifest("{name: inline, plots: []}")
    assert m.name == "inline"
    assert m.plots == []


def test_resolve_inline_list_rejected():
    with pytest.raises(ValueError):
        registry.resolve_manifest("[a, b]")


def test_resolve_explicit_path(manifests_dir):
    p = manifests_dir / "foo" / "report.yaml"
    m = registry.resolve_manifest(str(p))
    assert m.name == "foo"


def test_resolve_missing_name_raises(manifests_dir):
    with pytest.raises(registry.ManifestNotFoundError) as exc:
        registry.resolve_manifest("does_not_exist")
    assert "does_not_exist" in str(exc.value)
    assert "foo" in str(exc.value)  # available list included


def test_flat_report_yaml_is_not_found(manifests_dir):
    (manifests_dir / "flat_only.yaml").write_text(
        "manifest_version: '1.0'\nname: flat_only\nplots: []\n"
    )
    with pytest.raises(registry.ManifestNotFoundError):
        registry.resolve_manifest("flat_only")


def test_run_local_report_overrides_package(tmp_path: pathlib.Path, manifests_dir):
    local = tmp_path / "analysis" / "reports" / "foo" / "report.yaml"
    local.parent.mkdir(parents=True)
    local.write_text(
        "manifest_version: '1.0'\nname: foo\ntitle: Run Override\ndata_source: v_episode_summary\nplots: []\n"
    )
    m = registry.resolve_manifest("foo", benchmark_dir=tmp_path)
    assert m.title == "Run Override"


def test_legacy_viz_manifest_and_note_are_ignored(
    tmp_path: pathlib.Path, manifests_dir
):
    (tmp_path / "viz_manifest.yaml").write_text(
        "manifest_version: '1.0'\nname: legacy\ndata_source: v_episode_summary\nplots: []\n"
    )
    (tmp_path / "report_manifest.yaml").write_text("name: foo\n")
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "standard"


def test_default_when_nothing_else(tmp_path: pathlib.Path, manifests_dir):
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "standard"


def test_available_manifests(manifests_dir):
    assert registry.available_manifests() == ["foo", "standard"]
