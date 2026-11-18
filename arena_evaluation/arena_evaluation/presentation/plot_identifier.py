"""Plot declarations as resolvable assets: ``plots/<name>/plot.yaml`` in the config chain."""

from __future__ import annotations

import pathlib

import yaml
from arena_simulation_setup.tree import AssetIdentifier

from arena_evaluation.storage.config_identifiers import register_config_identifier
from arena_evaluation.storage.config_layout import bundle_config_file

__all__ = ["PlotIdentifier"]


class PlotIdentifier(AssetIdentifier[dict[str, object]]):
    """A reusable plot declaration, resolved local first and then from the benchmark buckets."""

    _asset_type = "plots"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PlotIdentifier) and self.name == other.name

    def load(self, path: pathlib.Path, /, **kwargs: object) -> dict[str, object]:
        del kwargs
        data = yaml.safe_load(
            bundle_config_file(path, "plot").read_text(encoding="utf-8")
        )
        if not isinstance(data, dict):
            raise ValueError(
                f"Plot declaration '{self.name}' must be a YAML mapping, got {type(data).__name__}."
            )
        return data


register_config_identifier(PlotIdentifier)
