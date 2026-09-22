"""Manifest validation via the shipped _validate_manifest from tools.py."""

import pytest

pytest.importorskip("yaml")
pytest.importorskip("mcp")
pytest.importorskip("polars")
pytest.importorskip("arena_evaluation_mcp")


def _validate(content: str) -> dict:
    from arena_evaluation_mcp.tools import _validate_manifest

    return _validate_manifest(content)


class TestManifestValidationFailures:
    """Failure paths that do not need the VizManifest schema installed."""

    def test_empty_content_fails(self):
        result = _validate("")
        assert result["valid"] is False
        assert "Empty" in result["error"]

    def test_not_a_dict_fails(self):
        result = _validate("- item1\n- item2\n")
        assert result["valid"] is False
        assert "Expected a mapping" in result["error"]

    def test_invalid_yaml_fails(self):
        result = _validate(": invalid: yaml: :")
        assert result["valid"] is False


class TestManifestValidationSchema:
    """Paths that exercise the real VizManifest schema."""

    def test_bundled_manifests_validate(self):
        pytest.importorskip("arena_evaluation.presentation.viz_manifest")
        from arena_evaluation.presentation.manifest_registry import (
            available_manifests,
            find_manifest_file,
        )

        stems = available_manifests()
        if not stems:
            pytest.skip("no bundled manifests found")
        for stem in stems:
            path = find_manifest_file(stem)
            assert path is not None
            result = _validate(path.read_text())
            assert result["valid"], f"bundled manifest '{stem}' should validate: {result}"
            assert result["n_plots"] >= 1

    def test_missing_required_plot_field_fails(self):
        pytest.importorskip("arena_evaluation.presentation.viz_manifest")
        result = _validate("plots:\n  - id: bad\n    type: violin\n    title: No data_key\n")
        assert result["valid"] is False

    def test_reserialized_bundled_manifest_still_validates(self):
        pytest.importorskip("arena_evaluation.presentation.viz_manifest")
        import yaml

        from arena_evaluation.presentation.manifest_registry import find_manifest_file

        path = find_manifest_file("standard")
        if path is None:
            pytest.skip("bundled standard manifest not found")
        data = yaml.safe_load(path.read_text())
        redumped = yaml.safe_dump(data, sort_keys=False)
        result = _validate(redumped)
        assert result["valid"], f"re-serialized standard manifest should validate: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
