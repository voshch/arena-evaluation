"""Declarative HTML and static plot report builder."""

from __future__ import annotations

import dataclasses
import datetime
import html as html_escape
import json
import logging
import pathlib
import re
from collections.abc import Sequence
from typing import Any

import jinja2
import markupsafe
import polars as pl
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import (
    resolve_metric_units,
    run_computation,
)
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.presentation.checks import (
    BuildChecks,
    CheckStage,
    DiagnosticInspector,
    PlotCheck,
    PlotDiagnostics,
)
from arena_evaluation.presentation.data_sources import EPISODE_SUMMARY_VIEW
from arena_evaluation.presentation.manifest_registry import is_inline, resolve_manifest
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.report_plot_block import (
    plot_block,
    plot_block_styles,
    plotly_bundle_script,
)
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest
from arena_evaluation.storage.config_layout import run_report_output_dir

__all__ = ["BuildResult", "PlotBuildRecord", "ReportBuilder"]

_STATUS_COLS: tuple[str, ...] = (
    "stage",
    "planner",
    "episode_id",
    "robot_id",
    "result",
    "null_reason",
)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PlotBuildRecord:
    """Provenance and check outcome for a single plot in a report build."""

    plot_id: str
    evidence_id: str = ""
    snapshot_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    snapshot_ids: tuple[str, ...] = ()
    row_count: int = 0
    passed: bool = True


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BuildResult:
    """Structured output of ReportBuilder.build()."""

    report_path: pathlib.Path
    checks: BuildChecks
    plots: tuple[PlotBuildRecord, ...]


def _unique_run_labels(dirs: tuple[pathlib.Path, ...]) -> tuple[str, ...]:
    names = [d.name for d in dirs]
    if len(set(names)) == len(names):
        return tuple(names)
    counts: dict[str, int] = {}
    labels: list[str] = []
    for name in names:
        idx = counts.get(name, 0) + 1
        counts[name] = idx
        labels.append(name if idx == 1 else f"{name}_{idx}")
    return tuple(labels)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _CatalogSources:
    """The benchmark catalogs one report reads, one catalog per source directory."""

    dirs: tuple[pathlib.Path, ...]
    catalogs: tuple[BenchmarkCatalog, ...]

    def __post_init__(self) -> None:
        if not self.catalogs:
            raise ValueError("A report needs at least one benchmark catalog.")

    @property
    def run_labels(self) -> tuple[str, ...]:
        return _unique_run_labels(self.dirs)

    def frame(self, table: str, *, tag_run: bool = False) -> pl.DataFrame | None:
        """Concatenate the named Star Schema table or view across every source, None when unknown."""
        frames: list[pl.DataFrame] = []
        multi = len(self.catalogs) > 1
        for label, catalog in zip(self.run_labels, self.catalogs, strict=True):
            if table not in catalog.table_names():
                if multi:
                    logging.getLogger(__name__).warning(
                        "Run %r is missing table %r.", label, table
                    )
                continue
            df = catalog.get_table(table).collect()
            if tag_run and multi:
                df = df.with_columns(pl.lit(label).alias("run"))
            frames.append(df)
        if not frames:
            return None
        return (
            frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        )

    def world_geometry(self) -> pl.DataFrame | None:
        """Load and concatenate dim_world_geometry across sources when registered."""
        return self.frame("dim_world_geometry")


def _status_table_data(df: pl.DataFrame) -> dict[str, Any] | None:
    """Structured table data for unevaluated episode rows, None when every row was evaluated."""
    schema = df.schema
    if schema.get("success") is None:
        return None
    rows = df.filter(pl.col("success").is_null())
    if rows.is_empty():
        return None
    cols = [c for c in _STATUS_COLS if schema.get(c) is not None]
    if not cols:
        return None
    sort_cols = [c for c in ("stage", "episode_id") if c in cols]
    if sort_cols:
        rows = rows.sort(sort_cols)
    rows = rows.select(cols)
    return {
        "heading": f"Episodes not fully evaluated ({len(rows)} of {len(df)})",
        "headers": [name.replace("_", " ") for name in cols],
        "rows": [
            ["NULL" if row.get(col) is None else str(row.get(col)) for col in cols]
            for row in rows.to_dicts()
        ],
    }


def _format_metric(value: float | None, template: str | None) -> str:
    if value is None or value != value:
        return "NULL"
    return template.format(value) if template else f"{value:.3f}"


def _summary_table_data(
    table: pl.DataFrame,
    group_cols: list[str],
    summary_specs: Sequence[Any],
) -> dict[str, Any]:
    """Pivot a long ``group_summary`` frame into a wide table: one row per group, one column per metric."""
    headers = [*group_cols, *[item.label for item in summary_specs]]
    grouped: dict[tuple[object, ...], dict[str, float | None]] = {}
    for row in table.to_dicts():
        key = tuple(row.get(col) for col in group_cols)
        bucket = grouped.setdefault(key, {})
        metric_name = str(row.get("metric", ""))
        mean_val = row.get("mean")
        bucket[metric_name] = (
            float(mean_val) if isinstance(mean_val, (int, float)) else None
        )

    rows: list[list[str]] = []
    for key, metric_map in grouped.items():
        cells = ["NULL" if val is None else str(val) for val in key]
        for item in summary_specs:
            cells.append(_format_metric(metric_map.get(item.metric), item.format))
        rows.append(cells)
    return {"headers": headers, "rows": rows}


def _render_markdown_light(text: str) -> markupsafe.Markup:
    """Minimal safe markdown -> HTML for agent notes (allows only http/https links)."""
    out = html_escape.escape(text, quote=True)
    out = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>',
        out,
    )
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return markupsafe.Markup(out.replace("\n", "<br>"))


def _resolve_manifest_and_name(
    manifest: VizManifest | str | pathlib.Path | None,
    benchmark_dir: pathlib.Path,
) -> tuple[VizManifest, str]:
    """Resolve a report manifest and its output directory name."""
    if isinstance(manifest, VizManifest):
        name = (manifest.name or "").strip()
        if not name:
            raise ValueError("Report manifest must declare a non-empty 'name'.")
        return manifest, name

    if isinstance(manifest, pathlib.Path):
        obj = VizManifest.load(manifest)
        name = (obj.name or "").strip()
        if not name:
            raise ValueError(
                f"Report manifest {manifest} must declare a non-empty 'name'."
            )
        return obj, name

    if isinstance(manifest, str):
        ref = manifest.strip()
        obj = resolve_manifest(ref, benchmark_dir=benchmark_dir)
        if is_inline(ref) or pathlib.Path(ref).exists():
            name = (obj.name or "").strip()
            if not name:
                raise ValueError(
                    "Inline or path report manifest must declare a non-empty 'name'."
                )
            return obj, name
        return obj, ref.removesuffix(".yaml")

    return resolve_manifest(None, benchmark_dir=benchmark_dir), "standard"


class ReportBuilder:
    """Generates interactive HTML reports and static plots from a VizManifest and BenchmarkCatalog."""

    def __init__(
        self,
        benchmark_dir: pathlib.Path,
        *,
        output_dir: pathlib.Path | None = None,
        manifest: VizManifest | str | pathlib.Path | None = None,
    ) -> None:
        self.benchmark_dir = pathlib.Path(benchmark_dir)
        self.output_root = (
            pathlib.Path(output_dir) if output_dir else self.benchmark_dir
        )
        manifest_obj, report_name = _resolve_manifest_and_name(
            manifest, self.benchmark_dir
        )
        self._manifest_obj: VizManifest = manifest_obj
        self.report_name = report_name
        self.output_dir = run_report_output_dir(self.output_root, report_name)
        self.plots_dir = self.output_dir / "plots"
        self.report_path = self.output_dir / "report.html"

        self._sources = _CatalogSources(
            dirs=(self.benchmark_dir,),
            catalogs=(BenchmarkCatalog.open(self.benchmark_dir),),
        )
        self._agent_notes: dict[str, str] | None = None
        self._build_checks = BuildChecks(plots_total=0, failures=())
        self._build_result: BuildResult | None = None

    @classmethod
    def from_dirs(
        cls,
        source_dirs: list[pathlib.Path],
        output_dir: pathlib.Path,
        *,
        manifest: VizManifest | str | pathlib.Path | None = None,
    ) -> ReportBuilder:
        """Build a ReportBuilder that resolves across multiple benchmark source directories."""
        instance = cls.__new__(cls)
        paths = tuple(pathlib.Path(src) for src in source_dirs) or (
            pathlib.Path(output_dir),
        )
        instance.benchmark_dir = paths[0]
        instance.output_root = pathlib.Path(output_dir)
        manifest_obj, report_name = _resolve_manifest_and_name(
            manifest, instance.benchmark_dir
        )
        instance._manifest_obj = manifest_obj
        instance.report_name = report_name
        instance.output_dir = run_report_output_dir(instance.output_root, report_name)
        instance.plots_dir = instance.output_dir / "plots"
        instance.report_path = instance.output_dir / "report.html"
        instance._agent_notes = None
        instance._build_checks = BuildChecks(plots_total=0, failures=())
        instance._build_result = None

        instance._sources = _CatalogSources(
            dirs=paths,
            catalogs=tuple(BenchmarkCatalog.open(p) for p in paths),
        )
        return instance

    @property
    def build_checks(self) -> BuildChecks:
        """The failed checks of the last build, the same rows the report appendix lists."""
        return self._build_checks

    @property
    def build_result(self) -> BuildResult | None:
        """Structured build provenance and check result from the last build."""
        return self._build_result

    def _resolve_for_spec(
        self, spec: PlotSpec, *, default_source: str = EPISODE_SUMMARY_VIEW
    ) -> tuple[pl.DataFrame, PlotDiagnostics]:
        """Resolve one plot across all source catalogs, retaining per-run provenance."""
        frames: list[pl.DataFrame] = []
        evidence_ids: list[str] = []
        snapshot_ids: list[str] = []
        multi = len(self._sources.dirs) > 1
        for label, src_dir, catalog in zip(
            self._sources.run_labels,
            self._sources.dirs,
            self._sources.catalogs,
            strict=True,
        ):
            resolver = DataResolver(
                catalog=catalog,
                benchmark_dir=src_dir,
            )
            resolved_spec = resolver.resolve_spec(spec)
            frame, diag = resolver.resolve(resolved_spec, default_source=default_source)
            if multi:
                frame = frame.with_columns(pl.lit(label).alias("run"))
            frames.append(frame)
            if diag.evidence_id:
                evidence_ids.append(diag.evidence_id)
            if diag.snapshot_id:
                snapshot_ids.append(diag.snapshot_id)

        combined = (
            frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        )
        checked = DiagnosticInspector.inspect(combined, spec)
        combined_diag = PlotDiagnostics(
            status=checked.status,
            issues=checked.issues,
            row_count=combined.height,
            evidence_id=evidence_ids[-1] if evidence_ids else "",
            snapshot_id=snapshot_ids[-1] if snapshot_ids else "",
            evidence_ids=tuple(evidence_ids),
            snapshot_ids=tuple(snapshot_ids),
        )
        return combined, combined_diag

    def _collect_units(self, manifest: VizManifest) -> dict[str, str]:
        metric_names: list[str] = []
        for spec in manifest.plots:
            for col in (spec.channels.x, spec.channels.y):
                if isinstance(col, str):
                    metric_names.append(col)
                elif isinstance(col, list):
                    metric_names.extend(col)
        units: dict[str, str] = {}
        if metric_names:
            units.update(
                resolve_metric_units(
                    self._sources.catalogs[0], list(dict.fromkeys(metric_names))
                )
            )
        units.update(manifest.units or {})
        return units

    def build(self) -> pathlib.Path | None:
        """Execute the report building process and return the generated HTML path."""
        # differs from overhaul/source:report_builder.py:261-375: multi-catalog per-plot resolution, no MetricRegistry, keeps evidence_ids & snapshot_ids
        manifest = self._manifest_obj
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        units = self._collect_units(manifest)
        world_geometry = self._sources.world_geometry()
        plotly_renderer = PlotlyRenderer(units=units)
        static_renderer = StaticRenderer(units=units)

        plot_entries: list[dict[str, Any]] = []
        failures: list[PlotCheck] = []
        records: list[PlotBuildRecord] = []

        def record_failure(spec: PlotSpec, stage: CheckStage, detail: str) -> None:
            failures.append(
                PlotCheck(plot_id=spec.id, title=spec.title, stage=stage, detail=detail)
            )

        spec_resolver = DataResolver(benchmark_dir=self.benchmark_dir)
        for raw_spec in manifest.plots:
            spec = raw_spec
            plot_failed = False
            diag = PlotDiagnostics()
            try:
                if raw_spec.uses:
                    spec = spec_resolver.resolve_spec(raw_spec)
                plot_df, diag = self._resolve_for_spec(
                    spec, default_source=manifest.data_source
                )
            except Exception as exc:
                record_failure(spec, CheckStage.LOAD, str(exc))
                records.append(PlotBuildRecord(plot_id=spec.id, passed=False))
                continue

            note_html, agent_note_html = self._plot_note_markup(spec)

            if diag.status == "FAILED":
                detail = (
                    "; ".join(diag.issues) if diag.issues else "Quality gate rejected"
                )
                record_failure(spec, CheckStage.QUALITY, detail)
                plot_entries.append(
                    {
                        "group": spec.layout_group,
                        "title": spec.title,
                        "gate_error": detail,
                        "html": None,
                        "note_html": note_html,
                        "agent_note_html": agent_note_html,
                    }
                )
                records.append(
                    PlotBuildRecord(
                        plot_id=spec.id,
                        evidence_id=diag.evidence_id,
                        snapshot_id=diag.snapshot_id,
                        evidence_ids=diag.evidence_ids,
                        snapshot_ids=diag.snapshot_ids,
                        row_count=diag.row_count,
                        passed=False,
                    )
                )
                continue

            png_path = self.plots_dir / f"{spec.id}.png"
            try:
                static_renderer.render(
                    spec,
                    plot_df,
                    png_path,
                    run_dir=self.output_dir,
                    world_geometry=world_geometry,
                    diagnostics=diag,
                )
                if not png_path.exists():
                    record_failure(
                        spec,
                        CheckStage.STATIC,
                        "Static renderer produced no output file",
                    )
                    plot_failed = True
            except Exception as exc:
                record_failure(spec, CheckStage.STATIC, str(exc))
                plot_failed = True

            if spec.options.get("vector_svg", False):
                svg_path = self.plots_dir / f"{spec.id}.svg"
                try:
                    static_renderer.render(
                        spec,
                        plot_df,
                        svg_path,
                        run_dir=self.output_dir,
                        world_geometry=world_geometry,
                        diagnostics=diag,
                    )
                except Exception as exc:
                    record_failure(spec, CheckStage.STATIC, f"vector SVG: {exc}")
                    plot_failed = True

            html_chunks: list[str] = []
            try:
                html_chunk = plotly_renderer.render(
                    spec,
                    plot_df,
                    run_dir=self.output_dir,
                    world_geometry=world_geometry,
                    diagnostics=diag,
                )
                if not html_chunk:
                    record_failure(
                        spec,
                        CheckStage.INTERACTIVE,
                        "Interactive renderer returned empty output",
                    )
                    plot_failed = True
                else:
                    html_chunks = (
                        html_chunk if isinstance(html_chunk, list) else [html_chunk]
                    )
            except Exception as exc:
                record_failure(spec, CheckStage.INTERACTIVE, str(exc))
                plot_failed = True

            if not plot_failed and png_path.is_file() and html_chunks:
                plot_entries.append(
                    {
                        "group": spec.layout_group,
                        "title": spec.title,
                        "gate_error": None,
                        "html": markupsafe.Markup(
                            plot_block(
                                plot_id=spec.id,
                                title=spec.title,
                                interactive_html="\n".join(html_chunks),
                                png_path=png_path,
                            )
                        ),
                        "note_html": note_html,
                        "agent_note_html": agent_note_html,
                    }
                )

            records.append(
                PlotBuildRecord(
                    plot_id=spec.id,
                    evidence_id=diag.evidence_id,
                    snapshot_id=diag.snapshot_id,
                    evidence_ids=diag.evidence_ids,
                    snapshot_ids=diag.snapshot_ids,
                    row_count=diag.row_count,
                    passed=not plot_failed,
                )
            )

        status_table, summary_table = self._build_summary_tables(manifest, failures)
        self._build_checks = BuildChecks(
            plots_total=len(manifest.plots), failures=tuple(failures)
        )
        self._build_result = BuildResult(
            report_path=self.report_path,
            checks=self._build_checks,
            plots=tuple(records),
        )

        report_title = manifest.title or self.output_root.name
        html_content = self._assemble_html(
            status_table=status_table,
            summary_table=summary_table,
            plot_entries=plot_entries,
            report_title=report_title,
            failed_checks=self._build_checks.failures,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(html_content, encoding="utf-8")
        return self.report_path

    def _build_summary_tables(
        self, manifest: VizManifest, failures: list[PlotCheck]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Build status and summary table payloads from manifest.data_source and manifest.summary."""
        # differs from overhaul/source:report_builder.py:489-531: delegates aggregation to analytics group_summary
        source_name = manifest.data_source or EPISODE_SUMMARY_VIEW
        source_df = self._sources.frame(source_name, tag_run=True)
        status_table = _status_table_data(source_df) if source_df is not None else None
        if not manifest.summary:
            return status_table, None

        if source_df is None or source_df.is_empty():
            failures.append(
                PlotCheck(
                    plot_id="summary",
                    title="Performance Summary",
                    stage=CheckStage.LOAD,
                    detail=f"Summary source {source_name!r} is missing or empty.",
                )
            )
            return status_table, None

        metrics = [item.metric for item in manifest.summary]
        missing_metrics = [m for m in metrics if source_df.schema.get(m) is None]
        if missing_metrics:
            failures.append(
                PlotCheck(
                    plot_id="summary",
                    title="Performance Summary",
                    stage=CheckStage.LOAD,
                    detail=f"Summary metric(s) {missing_metrics!r} missing from {source_name!r}.",
                )
            )
            return status_table, None

        if manifest.summary_group_by is None:
            failures.append(
                PlotCheck(
                    plot_id="summary",
                    title="Performance Summary",
                    stage=CheckStage.LOAD,
                    detail="Summary manifest must declare 'summary_group_by'.",
                )
            )
            return status_table, None

        raw_groups = (
            [manifest.summary_group_by]
            if isinstance(manifest.summary_group_by, str)
            else list(manifest.summary_group_by)
        )

        multi = len(self._sources.dirs) > 1
        if multi and "run" not in raw_groups:
            group_cols = ["run", *raw_groups]
        else:
            group_cols = raw_groups

        missing_groups = [g for g in group_cols if source_df.schema.get(g) is None]
        if missing_groups:
            failures.append(
                PlotCheck(
                    plot_id="summary",
                    title="Performance Summary",
                    stage=CheckStage.LOAD,
                    detail=f"Summary group_by column(s) {missing_groups!r} missing from {source_name!r}.",
                )
            )
            return status_table, None

        if not group_cols:
            return status_table, None

        comp_group_cols = [c for c in group_cols if c != "run" or not multi]

        frames: list[pl.DataFrame] = []
        for run_label, src_dir, catalog in zip(
            self._sources.run_labels,
            self._sources.dirs,
            self._sources.catalogs,
            strict=True,
        ):
            if source_name not in catalog.table_names():
                failures.append(
                    PlotCheck(
                        plot_id="summary",
                        title="Performance Summary",
                        stage=CheckStage.LOAD,
                        detail=f"Run {run_label!r} is missing source table {source_name!r}.",
                    )
                )
                continue
            store = EvidenceStore(src_dir)
            res = run_computation(
                "group_summary",
                catalog,
                params={
                    "source": source_name,
                    "metrics": metrics,
                    "group_cols": comp_group_cols,
                },
                evidence_store=store,
                purpose="summary",
            )
            frame = res.frame
            if multi:
                frame = frame.with_columns(pl.lit(run_label).alias("run"))
                frame = frame.select(["run", *[c for c in frame.columns if c != "run"]])
            frames.append(frame)

        if not frames:
            return status_table, None
        summary_df = (
            frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        )
        return status_table, _summary_table_data(
            summary_df, group_cols, manifest.summary
        )

    def _load_agent_notes(self) -> dict[str, str]:
        if self._agent_notes is not None:
            return self._agent_notes
        notes: dict[str, str] = {}
        search_dirs = [self.benchmark_dir]
        if self.output_dir != self.benchmark_dir:
            search_dirs.append(self.output_dir)
        for sdir in search_dirs:
            for filename in ("notes.json", "insights.json"):
                path = sdir / filename
                if not path.is_file():
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, str):
                                notes[str(k)] = v
                            elif isinstance(v, dict):
                                text = (
                                    v.get("insight") or v.get("note") or v.get("text")
                                )
                                if text:
                                    notes[str(k)] = str(text)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                plot_id = (
                                    item.get("plot_id")
                                    or item.get("id")
                                    or item.get("label")
                                )
                                text = (
                                    item.get("insight")
                                    or item.get("note")
                                    or item.get("value")
                                    or item.get("text")
                                )
                                if plot_id and text:
                                    notes[str(plot_id)] = str(text)
                except Exception:
                    continue
        self._agent_notes = notes
        return notes

    def _plot_note_markup(
        self, spec: PlotSpec
    ) -> tuple[markupsafe.Markup | None, markupsafe.Markup | None]:
        opts = spec.options or {}
        note = opts.get("note")
        if note is None and opts.get("notes_key"):
            notes_path = self.benchmark_dir / "notes.yaml"
            if notes_path.is_file():
                try:
                    data = yaml.safe_load(notes_path.read_text(encoding="utf-8"))
                    rows = (
                        data
                        if isinstance(data, list)
                        else [
                            {"label": str(k), "value": str(v)}
                            for k, v in (data or {}).items()
                        ]
                    )
                    for row in rows:
                        if isinstance(row, dict) and str(row.get("label", "")) == str(
                            opts["notes_key"]
                        ):
                            note = row.get("value")
                            break
                except Exception:
                    note = None

        note_html = _render_markdown_light(str(note)) if note else None
        agent_notes = self._load_agent_notes()
        agent_html = (
            _render_markdown_light(agent_notes[spec.id])
            if spec.id and spec.id in agent_notes
            else None
        )
        return note_html, agent_html

    def _assemble_html(
        self,
        *,
        status_table: dict[str, Any] | None,
        summary_table: dict[str, Any] | None,
        plot_entries: list[dict[str, Any]],
        report_title: str,
        failed_checks: tuple[PlotCheck, ...],
    ) -> str:
        overview_plots: list[dict[str, Any]] = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in plot_entries:
            if entry["group"] == "overview":
                overview_plots.append(entry)
            else:
                grouped.setdefault(str(entry["group"]), []).append(entry)

        plot_groups = [
            {"title": name, "plots": items} for name, items in grouped.items()
        ]
        template_dir = pathlib.Path(__file__).parent
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("report_template.html.j2")
        return template.render(
            benchmark_id=report_title,
            report_dir=str(self.output_dir),
            generated_on=datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            status_table=status_table,
            summary_table=summary_table,
            overview_plots=overview_plots,
            plot_groups=plot_groups,
            failed_checks=failed_checks,
            plotly_bundle=markupsafe.Markup(plotly_bundle_script()),
            plot_block_styles=markupsafe.Markup(plot_block_styles()),
        )
