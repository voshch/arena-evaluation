"""Deterministic conversion of pre-overhaul report manifests into the single grammar.

The pre-overhaul manifests declared a flat plot with ``data_key``/``differentiate``/
``group_by``/``filter``/``auto_differentiate``; the overhaul rejects every one of those
field names. This module reads such a manifest and produces a ``VizManifest`` in the new
grammar plus a migration report that accounts for **every** old plot: converted, or
dropped with a reason, or left as a ``TODO:`` marker that an agent must complete.

Nothing is guessed: a binding that cannot be resolved deterministically becomes a TODO
entry (and the plot is omitted from the skeleton), and the migration report lists it.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Mapping
from typing import AbstractSet, Final

import yaml

from arena_evaluation.presentation.data_sources import EPISODE_SUMMARY_VIEW
from arena_evaluation.presentation.viz_manifest import (
    ManifestGroup,
    PlotChannels,
    PlotSpec,
    SummarySpec,
    VizManifest,
)
from arena_evaluation.storage.registry.series import SERIES_TABLES

__all__ = [
    "LegacyManifest",
    "LegacyMigration",
    "LegacyPlot",
    "MetricResolution",
    "MigrationEntry",
    "classify_legacy_plot_declaration",
    "convert_legacy_file",
    "convert_manifest",
    "load_legacy_manifest",
    "load_metric_map",
    "render_manifest_yaml",
    "render_migration_report",
    "write_legacy_bundle",
]

CONVERTED: Final[str] = "converted"
DROPPED: Final[str] = "dropped"
TODO: Final[str] = "todo"

_DIMENSION_ALIASES: Final[dict[str, str]] = {
    "planner": "planner",
    "local_planner": "local_planner",
    "inter_planner": "inter_planner",
    "robot": "robot_id",
    "robot_id": "robot_id",
    "robot_model": "robot_model",
    "stage": "stage",
    "world": "world",
    "scenario": "scenario",
    "dispatcher": "dispatcher",
    "seed": "seed",
    "crowd_condition": "crowd_condition",
    "is_reference": "is_reference",
}
_TRAJECTORY_SUBJECTS: Final[frozenset[str]] = frozenset(
    {"path", "pedestrian_path", "robot_path"}
)
# Old series names carry a short scope prefix and often a suffix the registry renamed.
_SERIES_PREFIX_STEMS: Final[dict[str, str]] = {
    "char": "characterization",
    "power": "energy",
    "battery": "energy",
    "prox": "proxemics",
    "prox_ext": "proxemics_extended",
    "path": "path_metrics",
    "ped_path": "pedestrian_path_metrics",
    "motion": "motion_metrics",
    "acoustic": "acoustic_exposure",
    "social": "social_forces",
    "hhi": "hhi_contact",
    "clearance": "clearance",
    "gaze": "gaze",
}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LegacyPlot:
    """One plot declaration of the pre-overhaul manifest schema."""

    id: str
    type: str
    title: str
    data_key: str
    layout_group: str
    differentiate: str = ""
    secondary_data_key: str = ""
    group_by: tuple[str, ...] = ()
    filter: Mapping[str, object] = dataclasses.field(default_factory=dict)
    options: Mapping[str, object] = dataclasses.field(default_factory=dict)
    auto_differentiate: bool = False


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LegacyManifest:
    """A pre-overhaul manifest as declared."""

    name: str
    title: str
    description: str
    data_source: str
    units: Mapping[str, str]
    groups: tuple[ManifestGroup, ...]
    plots: tuple[LegacyPlot, ...]
    summary: tuple[Mapping[str, object], ...] = ()
    summary_group_by: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MetricResolution:
    """What one old metric or dimension name became."""

    legacy: str
    new: str | None
    status: str
    note: str = ""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MigrationEntry:
    """The disposition of one old plot."""

    plot_id: str
    legacy_type: str
    status: str
    new_id: str | None
    reason: str


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LegacyMigration:
    """A converted manifest plus the full accounting of its old plots."""

    source: pathlib.Path
    manifest: VizManifest
    entries: tuple[MigrationEntry, ...]
    metrics: tuple[MetricResolution, ...]
    carried_notes: tuple[str, ...] = ()

    @property
    def converted(self) -> int:
        return sum(entry.status == CONVERTED for entry in self.entries)

    @property
    def dropped(self) -> int:
        return sum(entry.status == DROPPED for entry in self.entries)

    @property
    def todo(self) -> int:
        return sum(entry.status == TODO for entry in self.entries)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MetricMap:
    """The checked-in old-name to new-name mapping."""

    renamed: Mapping[str, str]
    removed: Mapping[str, str]

    def resolve(self, legacy: str, known: AbstractSet[str]) -> MetricResolution:
        """Resolve one old metric name without guessing."""
        if legacy in self.renamed:
            return MetricResolution(
                legacy=legacy, new=self.renamed[legacy], status="renamed"
            )
        if legacy in self.removed:
            return MetricResolution(
                legacy=legacy, new=None, status="removed", note=self.removed[legacy]
            )
        if legacy in known:
            return MetricResolution(legacy=legacy, new=legacy, status="confirmed")
        return MetricResolution(
            legacy=legacy,
            new=None,
            status="unknown",
            note="TODO: not in the metric registry and no mapping entry; decide rename or drop.",
        )


def _spec(
    plot: LegacyPlot,
    plot_type: str,
    data: Mapping[str, object],
    channels: Mapping[str, str] | PlotChannels,
    options: Mapping[str, object] | None = None,
) -> PlotSpec:
    """Build a plot spec from a legacy declaration plus its converted binding."""
    return PlotSpec(
        id=plot.id,
        type=plot_type,
        title=plot.title,
        layout_group=plot.layout_group,
        data=dict(data),
        channels=channels,
        options=dict(options or {}),
    )


def _metric_map_path(candidate: pathlib.Path | None) -> pathlib.Path | None:
    """An explicit map file, or one found in a config root, or None for the built-in map."""
    if candidate is not None:
        return pathlib.Path(candidate)
    from arena_evaluation.storage.config_layout import package_config_roots

    for root in package_config_roots():
        for relative in (
            pathlib.Path("legacy") / "legacy_metric_map.yaml",
            pathlib.Path("reports") / "legacy_metric_map.yaml",
        ):
            path = root / relative
            if path.is_file():
                return path
    return None


def load_metric_map(path: pathlib.Path | None = None) -> MetricMap:
    """Load the legacy metric map: an explicit file, a config-root map, or the built-in one."""
    from .legacy_metric_map import LEGACY_METRIC_MAP

    resolved = _metric_map_path(path)
    raw: object = (
        yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if resolved is not None
        else LEGACY_METRIC_MAP
    )
    if not isinstance(raw, dict):
        raise ValueError("Legacy metric map must be a YAML mapping.")
    renamed = raw.get("renamed") or {}
    removed = raw.get("removed") or {}
    if not isinstance(renamed, dict) or not isinstance(removed, dict):
        raise ValueError("Legacy metric map 'renamed'/'removed' must be mappings.")
    return MetricMap(
        renamed={str(k): str(v) for k, v in renamed.items()},
        removed={str(k): str(v) for k, v in removed.items()},
    )


_LEGACY_MANIFEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "manifest_version",
        "name",
        "title",
        "description",
        "data_source",
        "groups",
        "units",
        "plots",
        "summary",
        "summary_group_by",
    }
)
_LEGACY_PLOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "type",
        "title",
        "data_key",
        "secondary_data_key",
        "differentiate",
        "auto_differentiate",
        "group_by",
        "filter",
        "layout_group",
        "options",
    }
)


def _mapping(value: object, *, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a YAML mapping, got {type(value).__name__}.")
    return value


def _reject_unknown(
    raw: Mapping[str, object], allowed: frozenset[str], *, where: str
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"{where} declares unknown legacy field(s) {unknown}; "
            f"the legacy schema allows {sorted(allowed)}."
        )


def load_legacy_manifest(path: pathlib.Path) -> LegacyManifest:
    """Read a pre-overhaul manifest strictly; unknown fields fail loudly."""
    resolved = pathlib.Path(path)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    document = _mapping(raw, where=str(resolved))
    _reject_unknown(document, _LEGACY_MANIFEST_KEYS, where=str(resolved))
    groups = tuple(
        ManifestGroup(
            id=str(_mapping(item, where="groups[]")["id"]),
            title=str(_mapping(item, where="groups[]")["title"]),
        )
        for item in document.get("groups") or ()
    )
    plots: list[LegacyPlot] = []
    for index, item in enumerate(document.get("plots") or ()):
        plot_raw = _mapping(item, where=f"{resolved}:plots[{index}]")
        _reject_unknown(plot_raw, _LEGACY_PLOT_KEYS, where=f"{resolved}:plots[{index}]")
        options = plot_raw.get("options") or {}
        plots.append(
            LegacyPlot(
                id=str(plot_raw.get("id", f"plot_{index}")),
                type=str(plot_raw.get("type", "")),
                title=str(plot_raw.get("title", "")),
                data_key=str(plot_raw.get("data_key", "")),
                layout_group=str(plot_raw.get("layout_group", "")),
                differentiate=str(plot_raw.get("differentiate") or ""),
                secondary_data_key=str(plot_raw.get("secondary_data_key") or ""),
                group_by=tuple(str(v) for v in plot_raw.get("group_by") or ()),
                filter=_mapping(plot_raw.get("filter") or {}, where="filter"),
                options=_mapping(options, where="options"),
                auto_differentiate=bool(plot_raw.get("auto_differentiate", False)),
            )
        )
    return LegacyManifest(
        name=str(document.get("name") or resolved.stem),
        title=str(document.get("title") or resolved.stem),
        description=str(document.get("description") or ""),
        data_source=str(document.get("data_source") or ""),
        units={str(k): str(v) for k, v in (document.get("units") or {}).items()},
        groups=groups,
        plots=tuple(plots),
    )


@functools.lru_cache(maxsize=1)
def _series_columns() -> dict[str, frozenset[str]]:
    return {
        stem: frozenset(table.polars_schema().names())
        for stem, table in SERIES_TABLES.items()
    }


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _SeriesBinding:
    stem: str
    column: str


def _resolve_series(
    name: str, *, prefer_stem: str | None = None
) -> _SeriesBinding | None:
    """Resolve ``timeseries_<prefix>_<column>`` to a series table stem and column."""
    if not name.startswith("timeseries_"):
        return None
    rest = name[len("timeseries_") :]
    columns = _series_columns()
    prefix = rest.split("_", 1)[0]
    stems: list[str] = []
    if prefer_stem in columns:
        stems.append(prefer_stem)
    if prefix in _SERIES_PREFIX_STEMS and _SERIES_PREFIX_STEMS[prefix] not in stems:
        stems.append(_SERIES_PREFIX_STEMS[prefix])
    stems.extend(stem for stem in columns if stem not in stems)
    candidates = [rest]
    if prefix in _SERIES_PREFIX_STEMS:
        stripped = rest.removeprefix(f"{prefix}_")
        if stripped != rest:
            candidates.append(stripped)
    for stem in stems:
        cols = columns[stem]
        for candidate in candidates:
            if candidate in cols:
                return _SeriesBinding(stem=stem, column=candidate)
            matches = sorted(col for col in cols if col.startswith(candidate + "_"))
            if len(matches) == 1:
                return _SeriesBinding(stem=stem, column=matches[0])
    return None


def _dimension(name: str) -> str | None:
    return _DIMENSION_ALIASES.get(name)


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def _where(filter_map: Mapping[str, object]) -> str:
    parts = []
    for key, value in filter_map.items():
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        literal = (
            "[" + ", ".join(_literal(v) for v in values) + "]"
            if len(values) > 1
            else _literal(values[0])
        )
        parts.append(
            f"{key} == {literal}" if len(values) == 1 else f"{key} in {literal}"
        )
    return " and ".join(parts)


def _resolve_filter(
    filter_map: Mapping[str, object], *, prefer_stem: str | None
) -> tuple[dict[str, object] | None, str | None]:
    """Map legacy filter keys to new columns; return (filter, unresolved key)."""
    resolved: dict[str, object] = {}
    for key, value in filter_map.items():
        if key.startswith("timeseries_"):
            binding = _resolve_series(key, prefer_stem=prefer_stem)
            if binding is None:
                return None, key
            resolved[binding.column] = value
            continue
        if _dimension(key) is None:
            return None, key
        resolved[str(_dimension(key))] = value
    return resolved, None


def _group_columns(plot: LegacyPlot) -> list[str]:
    option_groups = plot.options.get("group_by") or ()
    raw = (
        list(plot.group_by)
        or [str(v) for v in option_groups]
        or ([plot.differentiate] if plot.differentiate else [])
    )
    resolved = [str(_dimension(name) or "") for name in raw]
    return [name for name in resolved if name]


def _color(plot: LegacyPlot) -> str | None:
    return _dimension(plot.differentiate) or None


def _series_source(name: str) -> tuple[str, str] | None:
    """Resolve a series telemetry name to (source view, column), or None."""
    binding = _resolve_series(name)
    if binding is None:
        return None
    return f"v_series_{binding.stem}", binding.column


def _primary_source(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[str, str, str | None, str]:
    """Resolve the first data binding: (view, metric, series stem, problem)."""
    series = _series_source(plot.data_key)
    if series is not None:
        return series[0], series[1], series[0].removeprefix("v_series_"), ""
    metric, problem = _metric_or_todo(plot.data_key, metric_map, known)
    return EPISODE_SUMMARY_VIEW, metric or "", None, problem or ""


def _metric_or_todo(
    name: str, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[str | None, str | None]:
    if not name or name == "*":
        return None, f"plot declares no single metric (data_key={name!r})"
    resolution = metric_map.resolve(name, known)
    if resolution.new is None:
        if resolution.status == "removed":
            return None, f"removed: {name} ({resolution.note})"
        return None, f"TODO: unknown metric {name!r} ({resolution.note})"
    return resolution.new, None


def _resolve_option_metrics(
    raw: object,
    *,
    minimum: int,
    label: str,
    metric_map: MetricMap,
    known: AbstractSet[str],
) -> tuple[list[str], str, str]:
    """Resolve an ``options.metrics`` list; return (metrics, note, problem)."""
    if not isinstance(raw, list) or len(raw) < minimum:
        return [], "", f"TODO: {label} without at least {minimum} options.metrics"
    metrics: list[str] = []
    dropped: list[str] = []
    for metric in raw:
        new, problem = _metric_or_todo(str(metric), metric_map, known)
        if problem:
            dropped.append(f"{metric}: {problem}")
            continue
        metrics.append(str(new))
    if len(metrics) < minimum:
        return [], "", f"TODO: {label} without {minimum} resolvable metrics"
    return metrics, f"converted; dropped {dropped}" if dropped else "", ""


def _convert_table(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    options = dict(plot.options)
    if options.get("notes"):
        return None, (
            "notes-only table: the new grammar has no figures without data; the note text "
            "is preserved in migration_report.yaml as carried_notes"
        )
    columns = options.get("columns") or options.get("metrics") or []
    metrics: list[str] = []
    labels: dict[str, str] = {}
    formats: dict[str, str] = {}
    if isinstance(columns, list):
        for column in columns:
            if isinstance(column, dict):
                metric = str(column.get("metric", ""))
                if column.get("label"):
                    labels[metric] = str(column["label"])
                if column.get("format"):
                    formats[metric] = str(column["format"])
            else:
                metric = str(column)
            if metric:
                metrics.append(metric)
    resolved: list[str] = []
    for metric in metrics:
        new, problem = _metric_or_todo(metric, metric_map, known)
        if problem:
            return None, f"unresolved table metric {metric!r}: {problem}"
        resolved.append(str(new))
    group_cols = _group_columns(plot)
    if not resolved or not group_cols:
        return None, "TODO: table without resolvable metrics or without a group column"
    options_new: dict[str, object] = {}
    if labels:
        options_new["labels"] = {
            metric_map.renamed.get(k, k): v for k, v in labels.items()
        }
    if formats:
        options_new["format"] = {
            metric_map.renamed.get(k, k): v for k, v in formats.items()
        }
    return (
        _spec(
            plot,
            "table",
            {
                "computation": "group_summary",
                "params": {"metrics": resolved, "group_cols": group_cols},
            },
            PlotChannels(
                x=group_cols[0],
                y=["metric", "n", "mean", "std"],
                color=group_cols[1] if len(group_cols) > 1 else None,
            ),
            options_new,
        ),
        "",
    )


def _convert_bar(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    source, metric, prefer_stem, problem = _primary_source(plot, metric_map, known)
    if problem:
        return None, problem
    group_cols = _group_columns(plot)
    if not group_cols:
        return None, "TODO: bar without a group/differentiate column"
    filter_map, unresolved = _resolve_filter(plot.filter, prefer_stem=prefer_stem)
    if unresolved:
        return None, f"TODO: unresolved filter key {unresolved!r}"
    params: dict[str, object] = {
        "source": source,
        "metrics": [metric],
        "group_cols": group_cols,
    }
    if filter_map:
        params["filter"] = filter_map
    return (
        _spec(
            plot,
            "bar",
            {"computation": "group_summary", "params": params},
            {"x": group_cols[0], "y": "mean"},
            {"err_plus": "err_plus", "err_minus": "err_minus"},
        ),
        "",
    )


def _convert_distribution(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    metric, problem = _metric_or_todo(plot.data_key, metric_map, known)
    if problem:
        return None, problem
    group_cols = _group_columns(plot)
    filter_map, unresolved = _resolve_filter(plot.filter, prefer_stem=None)
    if unresolved:
        return None, f"unresolved filter key {unresolved!r}"
    params: dict[str, object] = {"metric": metric, "nbins": 20}
    if group_cols:
        params["group_cols"] = group_cols
    if filter_map:
        params["filter"] = filter_map
    return (
        _spec(
            plot,
            "histogram",
            {"computation": "distribution_bins", "params": params},
            PlotChannels(
                x="bin_center", y="count", color=group_cols[0] if group_cols else None
            ),
            {"bin_low": "bin_low", "bin_high": "bin_high"},
        ),
        "",
    )


def _convert_direct(
    plot: LegacyPlot,
    metric_map: MetricMap,
    known: AbstractSet[str],
    metric_tables: Mapping[str, str] | None = None,
) -> tuple[PlotSpec | None, str]:
    """Box/violin/scatter: one metric (or two) bound directly to a view."""
    source, metric, prefer_stem, problem = _primary_source(plot, metric_map, known)
    if problem:
        return None, problem
    secondary_raw = str(plot.options.get("y") or plot.secondary_data_key or "")
    secondary = None
    if secondary_raw:
        secondary_series = _series_source(secondary_raw)
        if secondary_series is not None:
            secondary = secondary_series[1]
        else:
            secondary, problem = _metric_or_todo(secondary_raw, metric_map, known)
            if problem:
                return None, f"TODO: unresolved secondary metric: {problem}"
    filter_map, unresolved = _resolve_filter(plot.filter, prefer_stem=prefer_stem)
    if unresolved:
        return None, f"TODO: unresolved filter key {unresolved!r}"
    color = _color(plot)
    if plot.type == "scatter":
        if secondary is None:
            return None, "TODO: scatter without options.y (the second axis metric)"
        channels: PlotChannels | dict[str, str] = PlotChannels(
            x=metric, y=secondary, color=color
        )
    else:
        channels = PlotChannels(x=color, y=metric)
    note = ""
    if prefer_stem is None and metric_tables is not None:
        fact_table = metric_tables.get(str(metric))
        if fact_table == "fact_pedestrians" and plot.type != "scatter":
            source = "fact_pedestrians"
            channels = PlotChannels(x="ped_policy", y=metric)
            note = (
                "converted; grouped by ped_policy because fact_pedestrians has no "
                "planner column (the old plot grouped by the planner label)"
            )
        elif (
            fact_table in ("fact_contacts", "fact_encounters")
            and plot.type != "scatter"
        ):
            source = fact_table
            channels = PlotChannels(x="robot_id", y=metric)
            note = f"converted; grouped by robot_id because {fact_table} has no planner column"
    data: dict[str, object] = {"source": source}
    where = _where(filter_map or {})
    if where:
        data["where"] = where
    options = {
        key: value
        for key, value in plot.options.items()
        if key in ("show_points", "barmode", "show_markers")
    }
    return _spec(plot, plot.type, data, channels, options), note


def _convert_radar(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    metrics, note, problem = _resolve_option_metrics(
        plot.options.get("metrics") or [],
        minimum=3,
        label="radar",
        metric_map=metric_map,
        known=known,
    )
    if problem:
        return None, problem
    group_col = _color(plot) or "planner"
    return (
        _spec(
            plot,
            "radar",
            {
                "computation": "radar_profile",
                "params": {"group_col": group_col, "metrics": metrics},
            },
            {"x": "metric", "y": "normalized_score", "color": group_col},
        ),
        note,
    )


def _convert_heatmap(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    raw_metrics = plot.options.get("metrics") or []
    if isinstance(raw_metrics, list) and len(raw_metrics) >= 2:
        metrics, note, problem = _resolve_option_metrics(
            raw_metrics,
            minimum=2,
            label="correlation heatmap",
            metric_map=metric_map,
            known=known,
        )
        if problem:
            return None, problem
        return (
            _spec(
                plot,
                "heatmap",
                {
                    "computation": "correlation_matrix",
                    "params": {"metrics": metrics},
                },
                {"x": "metric_x", "y": "metric_y", "color": "pearson_r"},
            ),
            note,
        )
    x_raw = str(plot.options.get("x") or "")
    y_raw = str(plot.options.get("y") or "")
    x_dim = _dimension(x_raw) if x_raw else None
    y_dim = _dimension(y_raw) if y_raw else None
    if x_dim and y_dim:
        metric, problem = _metric_or_todo(plot.data_key, metric_map, known)
        if problem:
            return None, problem
        return (
            _spec(
                plot,
                "heatmap",
                {
                    "computation": "group_summary",
                    "params": {"metrics": [metric], "group_cols": [x_dim, y_dim]},
                },
                {"x": x_dim, "y": y_dim, "color": "mean"},
            ),
            "",
        )
    return None, (
        "TODO: heatmap without options.metrics (correlation) or options.x/options.y "
        "(pivot); decide the two dimensions or the metric list to correlate"
    )


def _convert_series(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    """line/timeseries: a series_profile over one published series table."""
    del metric_map, known
    raw_metrics = plot.options.get("metrics") or [plot.data_key]
    if not isinstance(raw_metrics, list):
        raw_metrics = [raw_metrics]
    bindings: list[_SeriesBinding] = []
    for metric in raw_metrics:
        binding = _resolve_series(str(metric))
        if binding is None:
            return None, f"unresolved series column {metric!r}"
        bindings.append(binding)
    if not bindings:
        return None, "TODO: series plot without a metric"
    if len(bindings) > 1:
        names = ", ".join(str(metric) for metric in raw_metrics)
        return None, (
            f"TODO: timeseries declares {len(bindings)} metrics ({names}); emit one "
            "series_profile plot per metric"
        )
    primary = bindings[0]
    x_raw = str(plot.options.get("x") or "timeseries_time_s")
    x_binding = _resolve_series(x_raw, prefer_stem=primary.stem)
    if x_binding is None:
        return None, f"TODO: unresolved series x column {x_raw!r}"
    group_cols = _group_columns(plot)
    filter_map, unresolved = _resolve_filter(plot.filter, prefer_stem=primary.stem)
    if unresolved:
        return None, f"unresolved filter key {unresolved!r}"
    params: dict[str, object] = {
        "source": f"v_series_{primary.stem}",
        "x": x_binding.column,
        "metric": primary.column,
        "group_cols": group_cols,
    }
    if plot.options.get("bin_x") is not None:
        params["bin_x"] = plot.options["bin_x"]
    if filter_map:
        params["filter"] = filter_map
    return (
        _spec(
            plot,
            plot.type,
            {"computation": "series_profile", "params": params},
            PlotChannels(
                x="x", y="value", color=group_cols[0] if group_cols else None
            ),
            {"show_markers": True} if plot.type == "line" else {},
        ),
        "",
    )


def _convert_trajectory(
    plot: LegacyPlot, metric_map: MetricMap, known: AbstractSet[str]
) -> tuple[PlotSpec | None, str]:
    del metric_map, known
    if plot.data_key not in _TRAJECTORY_SUBJECTS:
        return (
            None,
            f"trajectory subject {plot.data_key!r} is not a published track stream",
        )
    subject = "robot" if plot.data_key == "path" else "pedestrian"
    query = (
        "robot_tracks_non_ref"
        if subject == "robot"
        else "pedestrian_arena_peds_non_ref"
    )
    color = "robot_id" if subject == "robot" else "ped_id"
    return (
        _spec(
            plot,
            "trajectory",
            {"stored_query": query},
            {"x": "x", "y": "y", "color": color},
        ),
        "",
    )


_ACOUSTIC_DROP: Final[str] = (
    "acoustic plot types are registered but no recipe binds them to a report, and the "
    "report contract test forbids acoustic types in shipped manifests (L4-010); bind it "
    "ad hoc with evaluation_render_plot or ship a recipe first"
)

_CONVERTERS = {
    "table": _convert_table,
    "bar": _convert_bar,
    "histogram": _convert_distribution,
    "box": _convert_direct,
    "violin": _convert_direct,
    "scatter": _convert_direct,
    "radar": _convert_radar,
    "heatmap": _convert_heatmap,
    "line": _convert_series,
    "timeseries": _convert_series,
    "trajectory": _convert_trajectory,
}


def _convert_plot(
    plot: LegacyPlot,
    metric_map: MetricMap,
    known: AbstractSet[str],
    metric_tables: Mapping[str, str] | None = None,
) -> tuple[PlotSpec | None, MigrationEntry]:
    converter = _CONVERTERS.get(plot.type)
    if plot.type in ("acoustic_field", "acoustic_field_animation"):
        problem = _ACOUSTIC_DROP
        status = DROPPED
    elif converter is None:
        problem = f"TODO: no conversion rule for legacy plot type {plot.type!r}."
        status = TODO
    else:
        spec, problem = (
            converter(plot, metric_map, known, metric_tables)
            if converter is _convert_direct
            else converter(plot, metric_map, known)
        )
        if spec is not None:
            return spec, MigrationEntry(
                plot_id=plot.id,
                legacy_type=plot.type,
                status=CONVERTED,
                new_id=spec.id,
                reason=problem or "converted",
            )
        status = TODO if problem.startswith("TODO") else DROPPED
    return None, MigrationEntry(
        plot_id=plot.id,
        legacy_type=plot.type,
        status=status,
        new_id=None,
        reason=problem,
    )


def convert_manifest(
    legacy: LegacyManifest,
    *,
    metric_map: MetricMap,
    known_metrics: AbstractSet[str],
    source: pathlib.Path | None = None,
    metric_tables: Mapping[str, str] | None = None,
) -> LegacyMigration:
    """Convert one legacy manifest, accounting for every old plot."""
    specs: list[PlotSpec] = []
    entries: list[MigrationEntry] = []
    metrics: dict[str, MetricResolution] = {}
    for plot in legacy.plots:
        spec, entry = _convert_plot(plot, metric_map, known_metrics, metric_tables)
        entries.append(entry)
        if spec is not None:
            specs.append(spec)
        for name in _referenced_metric_names(plot):
            resolution = metric_map.resolve(name, known_metrics)
            metrics.setdefault(name, resolution)
    for item in legacy.summary:
        summary_metric = str(item.get("metric") or "")
        if summary_metric:
            metrics.setdefault(
                summary_metric, metric_map.resolve(summary_metric, known_metrics)
            )
    groups = legacy.groups or tuple(
        ManifestGroup(id=plot.layout_group, title=plot.layout_group)
        for plot in legacy.plots
        if plot.layout_group
    )
    unique_groups = tuple({group.id: group for group in groups}.values())
    summary, summary_group_by, summary_stem = _convert_summary(
        legacy, metric_map, known_metrics
    )
    data_source = (
        f"v_series_{summary_stem}" if summary_stem is not None else EPISODE_SUMMARY_VIEW
    )
    manifest = VizManifest(
        name=legacy.name,
        title=legacy.title,
        description=legacy.description,
        data_source=data_source,
        summary=summary,
        summary_group_by=summary_group_by or None,
        groups=list(unique_groups),
        units={
            str(metric_map.renamed.get(name, name)): unit
            for name, unit in legacy.units.items()
            if unit and metric_map.resolve(name, known_metrics).status != "removed"
        },
        plots=specs,
    )
    carried_notes = tuple(
        f"{plot.id} [{item.get('label', '')}]: {item.get('value', '')}"
        for plot in legacy.plots
        for item in (plot.options.get("notes") or ())
        if isinstance(item, dict)
    )
    return LegacyMigration(
        source=source or pathlib.Path(f"{legacy.name}.yaml"),
        manifest=manifest,
        entries=tuple(entries),
        metrics=tuple(sorted(metrics.values(), key=lambda item: item.legacy)),
        carried_notes=carried_notes,
    )


def _convert_summary(
    legacy: LegacyManifest,
    metric_map: MetricMap,
    known: AbstractSet[str],
) -> tuple[list[SummarySpec], list[str], str | None]:
    """Carry a legacy summary over, translating series and metric names."""
    specs: list[SummarySpec] = []
    stem: str | None = None
    for item in legacy.summary:
        name = str(item.get("metric") or "")
        if not name:
            continue
        series = _resolve_series(name)
        if series is not None:
            metric = series.column
            stem = stem or series.stem
        else:
            resolution = metric_map.resolve(name, known)
            if resolution.new is None:
                continue
            metric = resolution.new
        specs.append(
            SummarySpec(
                metric=metric,
                label=str(item.get("label") or metric),
                format=str(item.get("format") or "{:.2f}"),
            )
        )
    groups: list[str] = []
    for value in legacy.summary_group_by:
        series = _resolve_series(value, prefer_stem=stem)
        if series is not None:
            groups.append(series.column)
            stem = stem or series.stem
            continue
        dimension = _dimension(value)
        if dimension is not None:
            groups.append(dimension)
    return specs, groups, stem


def _referenced_metric_names(plot: LegacyPlot) -> list[str]:
    names: list[str] = []
    if (
        plot.data_key
        and plot.data_key != "*"
        and not plot.data_key.startswith("timeseries_")
    ):
        names.append(plot.data_key)
    for key in ("metrics", "y", "x"):
        value = plot.options.get(key)
        values = [value] if isinstance(value, str) else (value or ())
        names.extend(str(v) for v in values if isinstance(v, str) and v != "*")
    for column in plot.options.get("columns") or ():
        if isinstance(column, dict) and column.get("metric"):
            names.append(str(column["metric"]))
    return [name for name in dict.fromkeys(names) if not name.startswith("timeseries_")]


def convert_legacy_file(
    path: pathlib.Path,
    *,
    metric_map: MetricMap,
    known_metrics: AbstractSet[str],
    metric_tables: Mapping[str, str] | None = None,
) -> LegacyMigration:
    """Load and convert one pre-overhaul manifest file."""
    legacy = load_legacy_manifest(path)
    return convert_manifest(
        legacy,
        metric_map=metric_map,
        known_metrics=known_metrics,
        source=pathlib.Path(path),
        metric_tables=metric_tables,
    )


def render_migration_report(migration: LegacyMigration) -> dict[str, object]:
    """The machine-readable accounting of one conversion."""
    return {
        "source": str(migration.source),
        "manifest": migration.manifest.name,
        "counts": {
            "total": len(migration.entries),
            "converted": migration.converted,
            "dropped": migration.dropped,
            "todo": migration.todo,
        },
        "plots": [
            {
                "id": entry.plot_id,
                "legacy_type": entry.legacy_type,
                "status": entry.status,
                "new_id": entry.new_id,
                "reason": entry.reason,
            }
            for entry in migration.entries
        ],
        "metrics": [dataclasses.asdict(item) for item in migration.metrics],
        "carried_notes": list(migration.carried_notes),
    }


def render_manifest_yaml(migration: LegacyMigration) -> str:
    """The new-grammar manifest as YAML."""
    payload = migration.manifest.model_dump(
        exclude_none=True, exclude={"manifest_version"}
    )
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def write_legacy_bundle(
    migration: LegacyMigration,
    output_dir: pathlib.Path,
    *,
    overwrite: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write ``<output_dir>/<name>/report.yaml`` and its ``migration_report.yaml``."""
    bundle = pathlib.Path(output_dir) / migration.manifest.name
    report_path = bundle / "report.yaml"
    migration_path = bundle / "migration_report.yaml"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"{report_path} already exists; pass overwrite=True to replace it."
        )
    bundle.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_manifest_yaml(migration), encoding="utf-8")
    migration_path.write_text(
        yaml.safe_dump(
            render_migration_report(migration), sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )
    return report_path, migration_path


def classify_legacy_plot_declaration(path: pathlib.Path) -> dict[str, object]:
    """Classify one old ``plot_declarations/`` file: convertible, best-effort or archive."""
    try:
        raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {
            "file": pathlib.Path(path).name,
            "code_driven_schema": False,
            "has_plots": False,
            "disposition": "archive",
            "reason": f"not machine-readable YAML ({exc.__class__.__name__}); documentation only",
        }
    document = raw if isinstance(raw, dict) else {}
    schema_keys = {
        "show_plots",
        "datasets",
        "results",
        "single_episode_metrics",
        "single_episode_series",
    }
    is_code_schema = bool(set(document) & schema_keys)
    return {
        "file": pathlib.Path(path).name,
        "code_driven_schema": is_code_schema,
        "has_plots": "plots" in document,
        "disposition": "convert"
        if "plots" in document
        else ("best-effort" if is_code_schema else "archive"),
        "reason": (
            "declares plots in the current grammar"
            if "plots" in document
            else "code-driven schema (show_plots/datasets/results/single_episode_*); no unambiguous plot specs"
            if is_code_schema
            else "documentation/template file, no runnable declaration"
        ),
    }
