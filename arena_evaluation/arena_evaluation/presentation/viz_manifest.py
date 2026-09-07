from __future__ import annotations

import pathlib
import typing
import yaml
from pydantic import BaseModel, Field

from arena_evaluation.storage.schemas import PlotSpec


class ManifestGroup(BaseModel):
    """A report section (layout_group -> rendered heading)."""
    id: str
    title: str


class SummarySpec(BaseModel):
    """One column of the report's summary table."""
    metric: str
    label: str
    format: str = "{:.2f}"


class VizManifest(BaseModel):
    """Declarative specification of plots and tables for a benchmark report."""
    manifest_version: str = "1.0"
    name: str | None = None
    title: str | None = None
    description: str | None = None
    # metrics | characterization_samples | characterization_summary | <parquet filename>
    data_source: str = "metrics"
    groups: list[ManifestGroup] = Field(default_factory=list)
    summary: list[SummarySpec] = Field(default_factory=list)
    summary_group_by: list[str] | str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    plots: list[PlotSpec] = Field(default_factory=list)

    @classmethod
    def load(cls, path: pathlib.Path | None) -> "VizManifest":
        """Load manifest from a YAML file path (missing path -> default)."""
        if path is None or not pathlib.Path(path).exists():
            return cls.load_default()

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest {path} must be a YAML mapping, got {type(data).__name__}")
        return cls.model_validate(data)

    @classmethod
    def load_default(cls) -> "VizManifest":
        """Load the default ('standard') named manifest."""
        from .manifest_registry import ManifestNotFoundError, find_manifest_file

        p = find_manifest_file("standard")
        if p is None:
            raise ManifestNotFoundError(
                "standard",
                "Manifest 'standard' not found. Install arena_evaluation or check "
                "configs/benchmark/manifests/standard.yaml.",
            )
        return cls.load(p)

    @classmethod
    def _default_manifest(cls) -> "VizManifest":
        """Backward-compatible alias for :meth:`load_default`."""
        return cls.load_default()
