"""Suites, contests, report manifests, plot and query declarations as Identifiers."""

from __future__ import annotations

import typing
from pathlib import Path

import yaml
from arena_simulation_setup.tree import AssetIdentifier

from arena_evaluation.analytics.query_identifier import QueryIdentifier
from arena_evaluation.presentation.manifest_identifier import ManifestIdentifier
from arena_evaluation.presentation.plot_identifier import PlotIdentifier
from arena_evaluation.storage.config_identifiers import (
    ConfigPathResolver,
    register_config_identifier,
)
from arena_evaluation.storage.config_layout import bundle_config_file

if typing.TYPE_CHECKING:
    from .config import Contest, Suite

__all__ = [
    "ConfigPathResolver",
    "ContestIdentifier",
    "ManifestIdentifier",
    "PlotIdentifier",
    "QueryIdentifier",
    "SuiteIdentifier",
]


class SuiteIdentifier(AssetIdentifier["Suite"]):
    _asset_type = "suites"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SuiteIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> Suite:
        del kwargs
        from .config import Suite

        return Suite.parse(
            self.name, yaml.safe_load(bundle_config_file(path, "suite").read_text())
        )


class ContestIdentifier(AssetIdentifier["Contest"]):
    _asset_type = "contests"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContestIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> Contest:
        del kwargs
        from .config import Contest

        return Contest.parse(
            self.name, yaml.safe_load(bundle_config_file(path, "contest").read_text())
        )


register_config_identifier(SuiteIdentifier)
register_config_identifier(ContestIdentifier)
register_config_identifier(ManifestIdentifier)
