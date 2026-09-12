from __future__ import annotations

import pathlib

import yaml

from .exceptions import ManifestGenerationError
from .schemas import RunMetadata


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
        """Read RunMetadata from a YAML file."""
        if not source.exists():
            raise ManifestGenerationError(f"Metadata file not found: {source}")

        try:
            with open(source) as f:
                data = yaml.safe_load(f)
            return RunMetadata.model_validate(data)
        except Exception as e:
            raise ManifestGenerationError(f"Failed to read metadata from {source}: {e}") from e

    @staticmethod
    def update(source: pathlib.Path, **kwargs: object) -> RunMetadata:
        """Update existing metadata with new fields and save."""
        metadata = MetadataWriter.read(source)

        for key, value in kwargs.items():
            if key not in RunMetadata.model_fields:
                raise ManifestGenerationError(f"Invalid metadata field: {key}")
            setattr(metadata, key, value)

        MetadataWriter.write(metadata, source)
        return metadata
