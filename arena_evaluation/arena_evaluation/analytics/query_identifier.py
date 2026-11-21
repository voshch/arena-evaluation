"""Stored SQL queries as resolvable assets: ``queries/<name>/query.yaml`` in the config chain."""

from __future__ import annotations

import pathlib

import yaml
from arena_simulation_setup.tree import AssetIdentifier

from arena_evaluation.storage.config_identifiers import register_config_identifier
from arena_evaluation.storage.config_layout import bundle_config_file

__all__ = ["QueryIdentifier"]


class QueryIdentifier(AssetIdentifier[dict[str, object]]):
    """A stored query declaration, resolved local first and then from the benchmark buckets.

    It loads the parsed YAML; :mod:`arena_evaluation.analytics.queries` validates it into a
    ``QueryDeclaration``, so this module imports nothing from the catalog that imports it."""

    _asset_type = "queries"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QueryIdentifier) and self.name == other.name

    def load(self, path: pathlib.Path, /, **kwargs: object) -> dict[str, object]:
        del kwargs
        data = yaml.safe_load(bundle_config_file(path, "query").read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Query declaration '{self.name}' must be a YAML mapping, got {type(data).__name__}.")
        return data


register_config_identifier(QueryIdentifier)
