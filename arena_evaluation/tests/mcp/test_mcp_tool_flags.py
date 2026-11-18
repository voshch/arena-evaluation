from __future__ import annotations

import os

import asyncio
import typing

import pydantic
import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent

from arena_evaluation.mcp import domain
from arena_evaluation.mcp.query import evaluation_query
from arena_evaluation.mcp.slices import evaluation_slice, evaluation_worst_encounters

TOOL_CLASSES: dict[str, str] = {
    "evaluation_acoustic": "overwrite",
    "evaluation_aggregate": "read",
    "evaluation_benchmark": "overwrite",
    "evaluation_compare_planners": "read",
    "evaluation_config_schema": "read",
    "evaluation_config_template": "read",
    "evaluation_console": "read",
    "evaluation_contests": "read",
    "evaluation_correlation": "read",
    "evaluation_create_contest": "overwrite",
    "evaluation_create_manifest": "overwrite",
    "evaluation_create_suite": "overwrite",
    "evaluation_dashboard": "write",
    "evaluation_describe_metric": "read",
    "evaluation_evidence": "read",
    "evaluation_extract": "overwrite",
    "evaluation_kill": "overwrite",
    "evaluation_lint_declarations": "read",
    "evaluation_manifests": "read",
    "evaluation_merge_runs": "write",
    "evaluation_metrics": "read",
    "evaluation_pipeline": "overwrite",
    "evaluation_plot": "overwrite",
    "evaluation_plot_declarations": "read",
    "evaluation_preflight": "overwrite",
    "evaluation_process": "overwrite",
    "evaluation_progress": "read",
    "evaluation_ps": "read",
    "evaluation_queries": "read",
    "evaluation_query": "write",
    "evaluation_query_metrics": "read",
    "evaluation_query_save": "overwrite",
    "evaluation_read_episode": "read",
    "evaluation_read_metrics": "read",
    "evaluation_read_notes": "read",
    "evaluation_read_plot_declaration": "read",
    "evaluation_record_start": "overwrite",
    "evaluation_record_stop": "overwrite",
    "evaluation_recorded_topics": "read",
    "evaluation_render_plot": "overwrite",
    "evaluation_report": "overwrite",
    "evaluation_runs": "read",
    "evaluation_save_plot_declaration": "overwrite",
    "evaluation_schema": "read",
    "evaluation_slice": "read",
    "evaluation_status": "read",
    "evaluation_stop": "overwrite",
    "evaluation_suites": "read",
    "evaluation_summarize": "read",
    "evaluation_top_n": "read",
    "evaluation_validate_contest": "read",
    "evaluation_validate_manifest": "read",
    "evaluation_validate_suite": "read",
    "evaluation_watch": "read",
    "evaluation_worst_encounters": "read",
    "evaluation_write_notes": "overwrite",
}

PydanticTool = typing.Callable[..., pydantic.BaseModel]

MODEL_TOOLS = (
    evaluation_slice,
    evaluation_worst_encounters,
)


def test_every_tool_is_classified_with_matching_flags():
    registered = {spec.name: spec for spec in domain.tools}
    assert set(TOOL_CLASSES) == set(registered)

    for name, category in TOOL_CLASSES.items():
        ann = registered[name].annotations
        if category == "read":
            assert ann.read_only_hint is True, name
            assert ann.destructive_hint is False, name
            assert ann.idempotent_hint is True, name
        elif category == "write":
            assert ann.read_only_hint is False, name
            assert ann.destructive_hint is False, name
        elif category == "overwrite":
            assert ann.read_only_hint is False, name
            assert ann.destructive_hint is True, name
        else:
            raise AssertionError(f"unknown category {category!r} for {name}")



@pytest.mark.parametrize("tool", (evaluation_slice, evaluation_worst_encounters))
def test_a_model_returning_tool_stays_typed_for_its_output_schema(tool: PydanticTool):
    annotation = typing.get_type_hints(tool)["return"]
    assert isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)
