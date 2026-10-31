from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier
from arena_evaluation.mcp import catalog


def test_run_manifests_lists_only_the_valid_report_files(tmp_path: pathlib.Path):
    (tmp_path / "valid.yaml").write_text("name: valid\nplots: []\n")
    (tmp_path / "no_plots.yaml").write_text("name: other\n")
    (tmp_path / "a_list.yaml").write_text("- plots\n")
    (tmp_path / "broken.yaml").write_text("plots: [\n")
    (tmp_path / "bad_schema.yaml").write_text("plots: 3\n")
    (tmp_path / "notes.txt").write_text("plots: []\n")
    assert catalog.run_manifests(tmp_path) == ["valid"]


@pytest.mark.parametrize(("kind", "identifier"), [("suite", SuiteIdentifier), ("contest", ContestIdentifier), ("manifest", ManifestIdentifier)])
def test_config_identifier_maps_each_kind(kind, identifier):
    assert catalog.config_identifier(kind) is identifier


def test_readme_section_keeps_only_the_sections_of_that_word():
    section = catalog._readme_section("suite")
    assert section.startswith("## Suite Files")
    assert "## Contest files" not in section


def test_readme_section_of_an_unknown_word_is_empty():
    assert catalog._readme_section("nothing_named_this") == ""


def test_config_file_rejects_a_path_as_the_name():
    with pytest.raises(ToolError, match="one path component"):
        catalog.config_file("suite", "../secrets")
