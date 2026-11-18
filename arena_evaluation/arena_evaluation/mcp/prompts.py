"""MCP workflow and per-role prompts generated from the role registry."""

from __future__ import annotations

from .base import domain
from .roles import ROLES


@domain.prompt()
def evaluation_analysis_workflow(question: str, run_id: str) -> str:
    """End-to-end hypothesis-to-verified-finding workflow prompt for an evaluation run."""
    return (
        f"Investigate question {question!r} on benchmark run {run_id!r}.\n"
        "1. Call evaluation_schema(run_id) to inspect available tables and units.\n"
        "2. Run evaluation_analysis('triage', {}, run_id=run_id) to check usable episodes.\n"
        "3. Run evaluation_analysis or evaluation_query with pin_evidence=True / save_as.\n"
        "4. Record claims with evaluation_finding_create(..., structured_numbers=..., author='analyst').\n"
        "5. Have the verifier role run evaluation_finding_verify and evaluation_finding_status."
    )


def _make_role_prompt(role_name: str, purpose: str, tools: list[str]) -> None:
    def _prompt(run_id: str) -> str:
        return (
            f"Role: {role_name}\n"
            f"Purpose: {purpose}\n"
            f"Run: {run_id}\n"
            f"Allowed tools: {', '.join(tools)}"
        )

    _prompt.__name__ = f"evaluation_role_{role_name}"
    _prompt.__doc__ = f"Role prompt for the {role_name} evaluation agent."
    domain.prompt()(_prompt)


for _role_name, _spec in ROLES.items():
    _make_role_prompt(_role_name, _spec["purpose"], list(_spec["tools"]))
