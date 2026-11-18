"""Unit tests for declarative report-manifest resolution."""

import pathlib

import pytest

from arena_evaluation.benchmark.tree import ConfigPathResolver
from arena_evaluation.presentation.manifest_identifier import (
    ManifestIdentifier,
)
from arena_evaluation.presentation import manifest_registry as registry


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


def test_resolve_inline_list_rejected():
    with pytest.raises(ValueError):
        registry.resolve_manifest("[a, b]")


def test_flat_report_yaml_is_not_found(manifests_dir):
    (manifests_dir / "flat_only.yaml").write_text(
        "manifest_version: '1.0'\nname: flat_only\nplots: []\n"
    )
    with pytest.raises(registry.ManifestNotFoundError):
        registry.resolve_manifest("flat_only")


def test_default_when_nothing_else(tmp_path: pathlib.Path, manifests_dir):
    m = registry.resolve_manifest(None, benchmark_dir=tmp_path)
    assert m.name == "standard"


