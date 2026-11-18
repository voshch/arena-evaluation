"""Declarative report manifest and plot specification models for the single reporting grammar."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data_sources import EPISODE_SUMMARY_VIEW, validate_data_source

__all__ = [
    "ManifestGroup",
    "PlotChannels",
    "PlotChecks",
    "PlotDataSpec",
    "PlotSliceSpec",
    "PlotSpec",
    "SummarySpec",
    "VizManifest",
]

_REJECTED_PLOT_FIELDS: Final[tuple[str, ...]] = (
    "data_key",
    "group_by",
    "differentiate",
    "auto_differentiate",
    "filter",
    "encodings",
    "source",
    "stored_query",
    "params",
    "data_source",
)
_REJECTED_CHANNEL_FIELDS: Final[tuple[str, ...]] = ("hue", "facet")
_REJECTED_MANIFEST_FILENAMES: Final[frozenset[str]] = frozenset(
    {"viz_manifest.yaml", "manifest.yaml"}
)


class PlotDataSpec(BaseModel):
    """Data binding for a plot: catalog source, SQL query, stored query, computation, or evidence."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    query: str | None = None
    stored_query: str | None = None
    computation: str | None = None
    params: dict[str, object] | None = None
    evidence_id: str | None = None
    where: str | None = None

    @model_validator(mode="after")
    def _validate_binding(self) -> PlotDataSpec:
        primaries = [
            name
            for name, val in (
                ("source", self.source),
                ("query", self.query),
                ("stored_query", self.stored_query),
                ("computation", self.computation),
                ("evidence_id", self.evidence_id),
            )
            if val is not None
        ]
        if len(primaries) > 1:
            raise ValueError(
                f"PlotDataSpec accepts at most one data binding, got {primaries}."
            )
        if (
            self.params is not None
            and self.stored_query is None
            and self.computation is None
        ):
            raise ValueError(
                "PlotDataSpec 'params' is only valid with 'stored_query' or 'computation'."
            )
        if self.source is not None:
            validate_data_source(self.source)
        return self


class PlotChannels(BaseModel):
    """Visual channel mappings for grammar-of-graphics plot declarations."""

    model_config = ConfigDict(extra="forbid")

    x: str | list[str] | None = None
    y: str | list[str] | None = None
    color: str | None = None
    facet_col: str | None = None
    facet_row: str | None = None
    size: str | None = None
    symbol: str | None = None
    hierarchy: str | list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_channels(cls, data: object) -> object:
        if isinstance(data, Mapping):
            for field_name in _REJECTED_CHANNEL_FIELDS:
                if field_name in data:
                    raise ValueError(
                        f"legacy channel field {field_name!r} is rejected; use 'color' / 'facet_col'."
                    )
        return data


class PlotSliceSpec(BaseModel):
    """Entity or window slice for targeted episode, encounter, or contact plots."""

    model_config = ConfigDict(extra="forbid")

    episode_id: int | None = None
    encounter_id: str | None = None
    contact_id: str | None = None


class PlotChecks(BaseModel):
    """Declared per-plot build checks; an unknown key is rejected instead of ignored."""

    model_config = ConfigDict(extra="forbid")

    require_columns: list[str] = Field(default_factory=list)
    min_rows: int | None = None
    # Columns whose 100% NULL value is expected for this plot (e.g. an optional telemetry
    # channel); every other declared numeric channel must carry at least one value.
    allow_all_null: list[str] = Field(default_factory=list)


class PlotSpec(BaseModel):
    """Single plot declaration in a report manifest or recipe."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    type: str = ""
    title: str = ""
    uses: str | None = None
    data: PlotDataSpec | None = None
    channels: PlotChannels = Field(default_factory=PlotChannels)
    slice: PlotSliceSpec | None = None
    options: dict[str, object] = Field(default_factory=dict)
    checks: PlotChecks = Field(default_factory=PlotChecks)
    layout_group: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_plot_fields(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        for field_name in _REJECTED_PLOT_FIELDS:
            if field_name in data:
                raise ValueError(
                    f"legacy PlotSpec field {field_name!r} is rejected; use 'data' and 'channels'."
                )
        if "data" in data and isinstance(data["data"], str):
            raise ValueError(
                "bare-string 'data' is rejected; pass a mapping with source, query, stored_query, computation, or evidence_id."
            )
        updated = dict(data)
        if "uses" in updated and not updated.get("id"):
            updated["id"] = updated["uses"]
        return updated

    @model_validator(mode="after")
    def _require_type_or_uses(self) -> PlotSpec:
        if not self.type.strip() and not (self.uses and self.uses.strip()):
            raise ValueError(f"PlotSpec {self.id!r} must declare 'type' or 'uses'.")
        return self


class ManifestGroup(BaseModel):
    """Report section grouping plots under a rendered heading."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str


class SummarySpec(BaseModel):
    """One metric column of the report summary table."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    label: str
    format: str = "{:.2f}"


class VizManifest(BaseModel):
    """Declarative specification of plots and summary tables for a benchmark report."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: str = "1.0"
    name: str | None = None
    title: str | None = None
    description: str | None = None
    data_source: str = EPISODE_SUMMARY_VIEW
    groups: list[ManifestGroup] = Field(default_factory=list)
    summary: list[SummarySpec] = Field(default_factory=list)
    summary_group_by: list[str] | str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    plots: list[PlotSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_manifest_fields(cls, data: object) -> object:
        if isinstance(data, Mapping) and "sections" in data:
            raise ValueError(
                "legacy manifest field 'sections' is rejected; use 'groups' and 'plots'."
            )
        return data

    @model_validator(mode="after")
    def _validate_data_source(self) -> VizManifest:
        validate_data_source(self.data_source)
        return self

    @classmethod
    def load(cls, path: pathlib.Path | None) -> VizManifest:
        """Load a manifest from a YAML file path, or load the default manifest when path is None."""
        if path is None:
            return cls.load_default()
        resolved = pathlib.Path(path)
        if resolved.name in _REJECTED_MANIFEST_FILENAMES:
            raise ValueError(f"legacy manifest filename {resolved.name!r} is rejected.")
        if not resolved.exists():
            raise FileNotFoundError(
                f"Report manifest {resolved} does not exist; pass None for the default manifest."
            )

        with resolved.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(
                f"Manifest {resolved} must be a YAML mapping, got {type(data).__name__}."
            )
        return cls.model_validate(data)

    @classmethod
    def load_default(cls) -> VizManifest:
        """Load the default 'standard' manifest."""
        from .manifest_identifier import ManifestIdentifier

        try:
            path = ManifestIdentifier(name="standard").resolve_source_sync().path
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Manifest 'standard' not found. Install arena_evaluation or check "
                "configs/reports/standard/report.yaml."
            ) from error
        return cls.load(path)
