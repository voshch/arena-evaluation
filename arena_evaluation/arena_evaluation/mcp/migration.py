"""MCP migration gateway: pre-overhaul report and plot declarations to the current grammar.

The conversion itself belongs to the L4 reporting layer (``presentation/legacy_manifest``)
so the MCP can only *run* it, never re-implement its rules. Every entry the converter
cannot bind deterministically stays ``todo``; this gateway never invents a binding, and it
writes the converted bundle only when the caller explicitly asks for it.
"""

from __future__ import annotations

import dataclasses
import importlib
import pathlib
from typing import Annotated, Any

import yaml
from arena_mcp.core import ToolError
from pydantic import Field

from arena_evaluation.storage.config_layout import run_analysis_root

from .base import (
    OptionalRunId,
    append_audit_event,
    assert_safe_analysis_write,
    atomic_write_text,
    component,
    domain,
    run_path,
)

__all__ = [
    "evaluation_manifest_mappings",
    "evaluation_migrate_manifest",
]


def _described(kind: Any, description: str) -> Any:
    """Annotate a tool parameter with its published description."""
    return Annotated[kind, Field(description=description)]


def _converter() -> Any:
    """Import the L4 legacy converter lazily and fail with a hint when it is not installed."""
    try:
        return importlib.import_module("arena_evaluation.presentation.legacy_manifest")
    except ImportError as exc:
        raise ToolError(
            "the legacy manifest converter is not available in this build",
            hint=(
                "requires arena_evaluation.presentation.legacy_manifest (the L4 migration "
                "converter); install the reporting layer that ships the converter, then "
                "call evaluation_migrate_manifest again"
            ),
        ) from exc


def _convert(source: str) -> tuple[Any, Any]:
    """Run the L4 converter on one legacy manifest: its module and the migration object."""
    converter = _converter()
    path = pathlib.Path(source).expanduser()
    if not path.is_file():
        raise ToolError(
            f"no legacy manifest file at {path}",
            hint="pass the path to a pre-overhaul configs/benchmark/manifests/<name>.yaml file",
        )
    path = path.resolve()
    from arena_evaluation.processing.metrics.dictionary import metric_dictionary

    dictionary = metric_dictionary()
    # The L4 converter takes in-memory metric maps, so this is the explicit boundary
    # where the columnar dictionary becomes the converter's inputs.
    names = dictionary.get_column("metric").to_list()  # agent-contracts: ignore AC004
    fact_tables = dictionary.get_column(
        "fact_table"
    ).to_list()  # agent-contracts: ignore AC004
    metrics = list(zip(names, fact_tables, strict=True))
    known = frozenset(str(name) for name, _ in metrics)
    metric_tables = {str(name): str(table or "") for name, table in metrics}
    try:
        migration = converter.convert_legacy_file(
            path,
            metric_map=converter.load_metric_map(None),
            known_metrics=known,
            metric_tables=metric_tables,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ToolError(
            f"cannot convert legacy manifest {path.name!r}: {exc}",
            hint="the converter accepts the pre-overhaul manifest grammar; fix the manifest or migrate it by hand",
        ) from exc
    return converter, migration


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_migrate_manifest(
    source: _described(
        str, "path to a pre-overhaul report manifest YAML file to convert"
    ),
    target: _described(
        str | None,
        "report bundle name to write under <run>/analysis/reports; defaults to the legacy name",
    ) = None,
    *,
    run_id: OptionalRunId = None,
    dry_run: _described(
        bool,
        "when true only return the skeleton and the conversion report; nothing is written",
    ) = True,
    overwrite: _described(
        bool,
        "replace an existing report.yaml in the target bundle",
    ) = False,
) -> dict[str, Any]:
    """Convert a legacy manifest into the current grammar, reporting every unresolved plot.

    The response carries the new-grammar skeleton, the per-plot accounting (converted,
    dropped with a reason, or todo), and the old->new metric resolutions. A ``todo`` entry
    is an unresolved binding: the agent must resolve it with
    ``evaluation_metrics``/``evaluation_describe_metric`` and a stored query, never by
    editing the skeleton blindly. The target bundle is written only with ``dry_run=False``
    and a run id, under ``<run>/analysis/reports/<target>/``.
    """
    converter, migration = _convert(source)
    target_name = component(
        target or str(migration.manifest.name), "target report name"
    )
    if migration.manifest.name != target_name:
        manifest = migration.manifest.model_copy(update={"name": target_name})
        migration = dataclasses.replace(migration, manifest=manifest)
    skeleton = converter.render_manifest_yaml(migration)
    migration_report = converter.render_migration_report(migration)

    written: dict[str, str] | None = None
    if not dry_run:
        if run_id is None:
            raise ToolError(
                "writing a migrated manifest needs a run id",
                hint="pass run_id so the bundle lands under <run>/analysis/reports/<target>/, or keep dry_run=True",
            )
        run_dir = run_path(run_id)
        bundle = run_analysis_root(run_dir) / "reports" / target_name
        report_path = bundle / "report.yaml"
        migration_path = bundle / "migration_report.yaml"
        assert_safe_analysis_write(run_dir, report_path)
        if report_path.exists() and not overwrite:
            raise ToolError(
                f"report bundle {target_name!r} already exists in run {run_dir.name!r}",
                hint="pass overwrite=True to replace it, or migrate into a new target name",
            )
        try:
            bundle.mkdir(parents=True, exist_ok=True)
            atomic_write_text(report_path, skeleton)
            atomic_write_text(
                migration_path,
                yaml.safe_dump(migration_report, sort_keys=False, allow_unicode=True),
            )
        except OSError as exc:
            raise ToolError(
                f"cannot write migrated manifest into run {run_dir.name!r}: {exc}",
                hint="the run's analysis folder must be a writable directory",
            ) from exc
        append_audit_event(
            run_dir,
            role="reporter",
            tool="evaluation_migrate_manifest",
            args={"source": str(migration.source), "target": target_name},
            produced_ids=[str(report_path)],
        )
        written = {"report": str(report_path), "migration_report": str(migration_path)}

    return {
        "source": str(migration.source),
        "run_id": run_id,
        "target": target_name,
        "dry_run": dry_run,
        "skeleton_yaml": skeleton,
        "migration_report": migration_report,
        "summary": dict(migration_report.get("counts") or {}),
        "written": written,
        "hint": (
            "resolve every 'todo' plot with evaluation_metrics and evaluation_describe_metric, "
            "author a stored query with evaluation_query_save, then validate with "
            "evaluation_validate_manifest and evaluation_lint_declarations(deep=True)"
        ),
    }


@domain.tool(read_only=True)
def evaluation_manifest_mappings(
    manifest: _described(
        str, "path to a pre-overhaul report manifest YAML file to map"
    ),
) -> dict[str, Any]:
    """The old->new metric and plot mapping table for one legacy manifest, without writing."""
    converter, migration = _convert(manifest)
    migration_report = converter.render_migration_report(migration)
    plots = [
        {
            "legacy_plot": str(entry["id"]),
            "legacy_type": str(entry["legacy_type"]),
            "status": str(entry["status"]),
            "new_plot": entry["new_id"],
            "reason": str(entry["reason"]),
        }
        for entry in migration_report.get("plots") or ()
    ]
    metrics = [
        {
            "legacy_metric": str(resolution["legacy"]),
            "new_metric": resolution["new"],
            "status": str(resolution["status"]),
            "note": str(resolution["note"]),
        }
        for resolution in migration_report.get("metrics") or ()
    ]
    return {
        "source": str(migration.source),
        "manifest": migration.manifest.name,
        "metrics": metrics,
        "plots": plots,
        "summary": dict(migration_report.get("counts") or {}),
        "carried_notes": list(migration_report.get("carried_notes") or ()),
        "hint": (
            "use evaluation_metrics and evaluation_describe_metric to resolve ambiguous or "
            "todo bindings; never guess a metric name"
        ),
    }
