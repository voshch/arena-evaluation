from __future__ import annotations

import pathlib
import typing

import yaml

from .exceptions import ManifestGenerationError, UnmigratedMetadataError
from .schemas import RunMetadata

_LEGACY_METADATA_PREFIX: typing.Final[str] = "tm_"
_MIGRATION_HINT: typing.Final[str] = "run `evaluation migrate <benchmark>` on the benchmark; the migration fills modes and params from the tm_* fields"


def _refuse_unmigrated_metadata(data: dict[str, typing.Any], source: pathlib.Path) -> None:
    """Refuse metadata that carries tm_* fields or lacks modes and params, pointing at the migration."""
    legacy = sorted(str(key) for key in data if str(key).startswith(_LEGACY_METADATA_PREFIX))
    if legacy:
        raise UnmigratedMetadataError(f"{source}: metadata still carries the unmigrated field(s) {', '.join(legacy)}; {_MIGRATION_HINT}")
    missing = [name for name in ("modes", "params") if name not in data]
    if missing and "benchmark_id" in data:
        raise UnmigratedMetadataError(f"{source}: metadata has no {' or '.join(missing)}; {_MIGRATION_HINT}")


class MetadataWriter:
    """Helper for reading and writing metadata.yaml files."""

    @staticmethod
    def write(metadata: RunMetadata, dest: pathlib.Path) -> None:
        """Write RunMetadata to a YAML file."""
        try:
            data = metadata.model_dump(exclude_none=True)
            with open(dest, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            try:
                dest.chmod(0o666)
            except Exception:
                pass
        except Exception as e:
            raise ManifestGenerationError(f"Failed to write metadata to {dest}: {e}") from e

    @staticmethod
    def read(source: pathlib.Path) -> RunMetadata:
        """Read RunMetadata from a YAML file, refusing metadata that still needs the migration."""
        if not source.exists():
            raise ManifestGenerationError(f"Metadata file not found: {source}")

        try:
            with open(source) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ManifestGenerationError(f"Failed to read metadata from {source}: {e}") from e

        if not isinstance(data, dict):
            raise ManifestGenerationError(f"Failed to read metadata from {source}: the metadata is not a mapping")
        _refuse_unmigrated_metadata(data, source)

        try:
            return RunMetadata.model_validate(data)
        except Exception as e:
            raise ManifestGenerationError(f"Failed to read metadata from {source}: {e}") from e
