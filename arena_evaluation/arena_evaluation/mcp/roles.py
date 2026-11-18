"""Agent role definitions, tool allowlists, and role resource for the evaluation harness."""

from __future__ import annotations

import json
from typing import Any, Final

from .base import domain

ROLES: Final[dict[str, dict[str, Any]]] = {
    "lead": {
        "purpose": "Coordinate investigation, inspect run status/schema, and review findings.",
        "tools": [
            "evaluation_runs",
            "evaluation_status",
            "evaluation_schema",
            "evaluation_metrics",
            "evaluation_describe_metric",
            "evaluation_findings",
            "evaluation_tables",
            "evaluation_lint_declarations",
            "evaluation_migrate_manifest",
            "evaluation_manifest_mappings",
            "evaluation_validate_manifest",
        ],
    },
    "triage": {
        "purpose": "Assess episode usability, missingness, and excluded rows before analysis.",
        "tools": [
            "evaluation_status",
            "evaluation_progress",
            "evaluation_schema",
            "evaluation_summarize",
            "evaluation_analysis",
            "evaluation_tables",
        ],
    },
    "analyst": {
        "purpose": "Run statistical queries, named analyses, sensitivity reruns, and propose findings.",
        "tools": [
            "evaluation_schema",
            "evaluation_query",
            "evaluation_queries",
            "evaluation_analysis",
            "evaluation_sensitivity",
            "evaluation_tables",
            "evaluation_refresh_table",
            "evaluation_slice",
            "evaluation_worst_encounters",
            "evaluation_finding_create",
            "evaluation_findings",
        ],
    },
    "attribution": {
        "purpose": "Drill into contact/encounter slices and compute summed-parts closing shares.",
        "tools": [
            "evaluation_schema",
            "evaluation_slice",
            "evaluation_worst_encounters",
            "evaluation_analysis",
            "evaluation_query",
            "evaluation_finding_create",
        ],
    },
    "verifier": {
        "purpose": "Mechanically verify findings against evidence and adjudicate status.",
        "tools": [
            "evaluation_findings",
            "evaluation_finding_verify",
            "evaluation_finding_status",
            "evaluation_evidence",
            "evaluation_query",
            "evaluation_schema",
        ],
    },
    "reporter": {
        "purpose": "Attach verified findings to reports, render plots, and assemble reviewed reports.",
        "tools": [
            "evaluation_findings",
            "evaluation_finding_attach",
            "evaluation_plot_declarations",
            "evaluation_read_plot_declaration",
            "evaluation_render_plot",
            "evaluation_manifests",
            "evaluation_report_draft",
            "evaluation_report_plot",
            "evaluation_report_view",
            "evaluation_report_review",
            "evaluation_report_assemble",
            "evaluation_evidence",
            "evaluation_save_plot_declaration",
            "evaluation_migrate_manifest",
            "evaluation_manifest_mappings",
            "evaluation_lint_declarations",
        ],
    },
    "migration": {
        "purpose": "Port a pre-overhaul report manifest to the current grammar and prove every binding.",
        "tools": [
            "evaluation_migrate_manifest",
            "evaluation_manifest_mappings",
            "evaluation_metrics",
            "evaluation_describe_metric",
            "evaluation_schema",
            "evaluation_tables",
            "evaluation_query",
            "evaluation_queries",
            "evaluation_query_save",
            "evaluation_validate_manifest",
            "evaluation_lint_declarations",
            "evaluation_render_plot",
            "evaluation_report_draft",
            "evaluation_report_plot",
            "evaluation_report_view",
            "evaluation_report_review",
            "evaluation_report_assemble",
            "evaluation_evidence",
            "evaluation_findings",
            "evaluation_finding_create",
            "evaluation_finding_verify",
            "evaluation_finding_status",
        ],
    },
    "operator": {
        "purpose": "Manage live benchmark runs, processes, recording, and destructive workspace actions.",
        "tools": [
            "evaluation_benchmark",
            "evaluation_preflight",
            "evaluation_stop",
            "evaluation_ps",
            "evaluation_kill",
            "evaluation_process",
            "evaluation_extract",
            "evaluation_pipeline",
            "evaluation_record_start",
            "evaluation_record_stop",
            "evaluation_create_suite",
            "evaluation_create_contest",
            "evaluation_create_manifest",
            "evaluation_query_save",
            "evaluation_report",
            "evaluation_write_notes",
            "evaluation_merge_runs",
        ],
    },
}


AGENT_AREA_TOOLS: Final[dict[str, str]] = {
    "evaluation_finding_status": "analysis",
    "evaluation_finding_attach": "analysis",
    "evaluation_query_save": "analysis",
    "evaluation_refresh_table": "analysis",
    "evaluation_render_plot": "analysis",
    "evaluation_save_plot_declaration": "analysis",
    "evaluation_report_plot": "analysis",
    "evaluation_report_assemble": "reports",
}


@domain.resource("arena://evaluation/roles")
def roles_resource() -> str:
    """Published role definitions, tool allowlists, and agent-area destructive tools."""
    return json.dumps({"roles": ROLES, "agent_area_tools": AGENT_AREA_TOOLS})
