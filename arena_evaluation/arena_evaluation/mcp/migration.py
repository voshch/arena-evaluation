"""MCP migration gateway: pre-overhaul report and plot declarations to the current grammar.

The conversion itself belongs to the L4 reporting layer (``presentation/legacy_manifest``)
so the MCP can only *run* it, never re-implement its rules. Every entry the converter
cannot bind deterministically stays ``todo``; this gateway never invents a binding, and it
writes the converted bundle only when the caller explicitly asks for it.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Annotated, Any, Final

import pydantic
import yaml
from arena_mcp.core import ToolError

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

_CONVERTER_MODULE: Final[str] = "arena_evaluation.presentation.legacy_manifest"


def _converter() -> Any:
    """Import the L4 legacy converter lazily and fail with a hint when it is not installed."""
    import importlib

    try:
        return importlib.import_module(_CONVERTER_MODULE)
    except ImportError as exc:
        raise ToolError(
            "the legacy manifest converter is not available in this build",
            hint=(
                "requires arena_evaluation.presentation.legacy_manifest (the L4 migration "
                "converter); install the reporting layer that ships the converter, then "
                "call evaluation_migrate_manifest again"
            ),
        ) from exc


def _manifest_path(source: str) -> pathlib.Path:
    """Return the existing pre-overhaul manifest file for ``source``."""
    path = pathlib.Path(source).expanduser()
    if not path.is_file():
        raise ToolError(
            f"no legacy manifest file at {path}",
            hint="pass the path to a pre-overhaul configs/benchmark/manifests/<name>.yaml file",
        )
    return path.resolve()


def _convert(source: str) -> Any:
    """Run the L4 converter on one legacy manifest and return its migration object."""
    converter = _converter()
    path = _manifest_path(source)
    from arena_evaluation.processing.metrics.dictionary import metric_dictionary

    dictionary = metric_dictionary()
    # The L4 converter takes in-memory metric maps, so this is the explicit boundary
    # where the columnar dictionary becomes the converter's inputs.
    metric_names = dictionary.get_column("metric")
    fact_table_names = dictionary.get_column("fact_table")
    names = metric_names.to_list()  # agent-contracts: ignore AC004
    fact_tables = fact_table_names.to_list()  # agent-contracts: ignore AC004
    known = frozenset(str(name) for name in names)
    metric_tables = {
        str(name): str(table or "")
        for name, table in zip(names, fact_tables, strict=True)
    }
    try:
        return converter.convert_legacy_file(
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


def _renamed(migration: Any, target: str) -> Any:
    """Return ``migration`` with its manifest name replaced by ``target`` when they differ."""
    if migration.manifest.name == target:
        return migration
    manifest = migration.manifest.model_copy(update={"name": target})
    return dataclasses.replace(migration, manifest=manifest)


@domain.tool(read_only=False, destructive=False, idempotent=False)
def evaluation_migrate_manifest(
    source: Annotated[
        str,
        pydantic.Field(
            description="path to a pre-overhaul report manifest YAML file to convert"
        ),
    ],
    target: Annotated[
        str | None,
        pydantic.Field(
            description="report bundle name to write under <run>/analysis/reports; defaults to the legacy name"
        ),
    ] = None,
    *,
    run_id: OptionalRunId = None,
    dry_run: Annotated[
        bool,
        pydantic.Field(
            description="when true only return the skeleton and the conversion report; nothing is written"
        ),
    ] = True,
    overwrite: Annotated[
        bool,
        pydantic.Field(
            description="replace an existing report.yaml in the target bundle"
        ),
    ] = False,
) -> dict[str, Any]:
    """Convert a legacy manifest into the current grammar, reporting every unresolved plot.

    The response carries the new-grammar skeleton, the per-plot accounting (converted,
    dropped with a reason, or todo), and the old->new metric resolutions. A ``todo`` entry
    is an unresolved binding: the agent must resolve it with
    ``evaluation_metrics``/``evaluation_describe_metric`` and a stored query, never by
    editing the skeleton blindly. The target bundle is written only with ``dry_run=False``
    and a run id, under ``<run>/analysis/reports/<target>/``.
    """
    migration = _convert(source)
    target_name = component(
        target or str(migration.manifest.name), "target report name"
    )
    converted = _renamed(migration, target_name)
    converter = _converter()
    skeleton = converter.render_manifest_yaml(converted)
    migration_report = converter.render_migration_report(converted)

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
    manifest: Annotated[
        str,
        pydantic.Field(
            description="path to a pre-overhaul report manifest YAML file to map"
        ),
    ],
) -> dict[str, Any]:
    """The old->new metric and plot mapping table for one legacy manifest, without writing."""
    migration = _convert(manifest)
    converter = _converter()
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
