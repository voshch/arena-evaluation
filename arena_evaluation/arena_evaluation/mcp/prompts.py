"""MCP workflow and per-role prompts generated from the role registry."""

from __future__ import annotations

from .base import domain
from .roles import ROLES

_REFUSALS = (
    "Refuse to run a tool outside the role allowlist, to publish a finding without a "
    "pinned evidence id, and to quote any number that evaluation_evidence cannot reproduce."
)


@domain.prompt()
def evaluation_analysis_workflow(question: str, run_id: str) -> str:
    """End-to-end hypothesis-to-verified-finding workflow prompt for an evaluation run."""
    return (
        f"Investigate question {question!r} on benchmark run {run_id!r}.\n"
        "1. Call evaluation_schema(run_id) to inspect available tables and units.\n"
        "2. Run evaluation_analysis('triage', {}, run_id=run_id) to check usable episodes.\n"
        "3. Run evaluation_analysis or evaluation_query with pin_evidence=True / save_as.\n"
        "4. Record claims with evaluation_finding_create(..., structured_numbers=..., author='analyst').\n"
        "5. Have the verifier role run evaluation_finding_verify and evaluation_finding_status.\n"
        f"{_REFUSALS}"
    )


@domain.prompt()
def evaluation_manifest_migration_workflow(
    run_id: str, source: str, report_name: str = "migrated_report"
) -> str:
    """Port one pre-overhaul report manifest with migrate, resolve, lint, render, and assemble."""
    return (
        f"Port legacy report manifest {source!r} into run {run_id!r} as {report_name!r}.\n"
        "1. evaluation_migrate_manifest(source, target=report_name, run_id=run_id, dry_run=True): "
        "read skeleton_yaml, the per-plot accounting, and every 'todo'.\n"
        "2. For each todo or ambiguous binding use evaluation_metrics(run_id), "
        "evaluation_describe_metric(metric), and evaluation_tables(run_id) to find the real "
        "metric, table, and direction. Never invent a metric name; if the converter cannot "
        "resolve it, leave it out and say why.\n"
        "3. Author the missing derivation with evaluation_query(run_id, ...) and persist it with "
        "evaluation_query_save(run_id, name, description, sql).\n"
        "4. evaluation_validate_manifest(yaml_content=<skeleton with resolved bindings>) and fix "
        "every schema error.\n"
        "5. evaluation_lint_declarations(run_id, deep=True): every plot must pass, or its reason "
        "must be stated in the report.\n"
        "6. evaluation_render_plot(run_id, declaration=<plot>, output_format='png'): look at the "
        "returned image and its checks; reject blank or unbound plots.\n"
        "7. evaluation_report_draft, evaluation_report_plot, evaluation_report_view, "
        "evaluation_report_review, then evaluation_report_assemble.\n"
        "8. Every quoted number must come from evaluation_evidence; the migration report is "
        "not evidence.\n"
        f"{_REFUSALS}"
    )


def _make_role_prompt(role_name: str, purpose: str, tools: list[str]) -> None:
    def _prompt(run_id: str) -> str:
        return (
            f"Role: {role_name}\n"
            f"Purpose: {purpose}\n"
            f"Run: {run_id}\n"
            f"Allowed tools: {', '.join(tools)}\n"
            f"{_REFUSALS}"
        )

    _prompt.__name__ = f"evaluation_role_{role_name}"
    _prompt.__doc__ = f"Role prompt for the {role_name} evaluation agent."
    domain.prompt()(_prompt)


for _role_name, _spec in ROLES.items():
    _make_role_prompt(_role_name, _spec["purpose"], list(_spec["tools"]))
