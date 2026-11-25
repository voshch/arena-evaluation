"""Report manifests as a resolvable asset identifier in the presentation layer."""

from __future__ import annotations

import pathlib
import typing

from arena_simulation_setup.tree import AssetIdentifier

from arena_evaluation.storage.config_layout import bundle_config_file, get_config_kind

if typing.TYPE_CHECKING:
    from .viz_manifest import VizManifest

__all__ = ["ManifestIdentifier"]


class ManifestIdentifier(AssetIdentifier["VizManifest"]):
    """A report manifest bundle resolved through the config asset chain."""

    _asset_type = get_config_kind("report").folder

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ManifestIdentifier) and self.name == other.name

    def load(self, path: pathlib.Path, /, **kwargs: object) -> VizManifest:
        del kwargs
        from .viz_manifest import VizManifest

        return VizManifest.load(bundle_config_file(path, "report"))
