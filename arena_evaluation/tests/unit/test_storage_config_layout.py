"""Unit tests for probing the package config root of the storage layout."""

from __future__ import annotations

import logging
import pathlib

import pytest

from arena_evaluation.storage import config_layout
from arena_evaluation.storage.config_layout import (
    find_config_asset,
    get_config_asset,
    package_config_roots,
    share_dir,
    source_tree_dir,
)


def _empty_ament_prefix(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ament at an empty prefix so arena_evaluation resolves as not installed."""
    prefix = tmp_path / "ament_prefix"
    prefix.mkdir()
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(prefix))


def test_a_package_that_is_not_installed_resolves_to_the_source_tree(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The not-installed state falls back to the source tree and logs one warning."""
    _empty_ament_prefix(tmp_path, monkeypatch)

    assert share_dir() is None
    with caplog.at_level(logging.WARNING, logger="arena_evaluation.storage.config_layout"):
        roots = package_config_roots()

    source = source_tree_dir()
    assert source is not None
    assert roots == [(source / "configs").resolve()]
    warnings = [record for record in caplog.records if "source tree" in record.getMessage()]
    assert len(warnings) == 1


def test_an_unset_ament_prefix_is_the_source_tree_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source checkout without ROS has no ament prefix and resolves to the source tree."""
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)

    assert share_dir() is None
    assert source_tree_dir() is not None


def test_a_missing_config_asset_warns_instead_of_defaulting_silently(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An absent palette names every searched root in a warning, not a silent default."""
    monkeypatch.setattr(config_layout, "all_config_roots", lambda *args: [tmp_path])
    with caplog.at_level(logging.WARNING, logger="arena_evaluation.storage.config_layout"):
        assert find_config_asset("palette") is None
    assert any("palette" in record.getMessage() and str(tmp_path) in record.getMessage() for record in caplog.records)


def test_an_unknown_config_asset_kind_is_refused() -> None:
    """Only the declared flat assets resolve; an unknown kind names the valid ones."""
    with pytest.raises(KeyError, match="palette"):
        get_config_asset("colour")
