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
from arena_evaluation.mcp.findings import (
    evaluation_finding_attach,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_findings,
)
from arena_evaluation.mcp.query import evaluation_query
from arena_evaluation.mcp.slices import evaluation_slice, evaluation_worst_encounters

TOOL_CLASSES: dict[str, str] = {
    "evaluation_acoustic": "overwrite",
    "evaluation_aggregate": "read",
    "evaluation_analysis": "write",
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
    "evaluation_finding_attach": "overwrite",
    "evaluation_finding_create": "write",
    "evaluation_finding_status": "overwrite",
    "evaluation_finding_verify": "write",
    "evaluation_findings": "read",
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
    "evaluation_refresh_table": "overwrite",
    "evaluation_render_plot": "overwrite",
    "evaluation_report": "overwrite",
    "evaluation_report_assemble": "overwrite",
    "evaluation_report_draft": "write",
    "evaluation_report_plot": "overwrite",
    "evaluation_report_review": "write",
    "evaluation_report_view": "read",
    "evaluation_runs": "read",
    "evaluation_save_plot_declaration": "overwrite",
    "evaluation_schema": "read",
    "evaluation_sensitivity": "write",
    "evaluation_slice": "read",
    "evaluation_status": "read",
    "evaluation_stop": "overwrite",
    "evaluation_suites": "read",
    "evaluation_summarize": "read",
    "evaluation_tables": "read",
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
    evaluation_findings,
    evaluation_finding_create,
    evaluation_finding_status,
    evaluation_finding_attach,
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


def test_append_insight_is_gone_and_the_finding_tools_replace_it():
    names = {spec.name for spec in domain.tools}

    assert "evaluation_append_insight" not in names
    assert {
        "evaluation_finding_create",
        "evaluation_finding_status",
        "evaluation_finding_attach",
        "evaluation_findings",
    } <= names


@pytest.mark.parametrize("tool", MODEL_TOOLS)
def test_a_model_returning_tool_stays_typed_for_its_output_schema(tool: PydanticTool):
    annotation = typing.get_type_hints(tool)["return"]

    assert isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)


def test_the_capped_query_contract_keeps_its_inline_fields():
    annotation = typing.get_type_hints(evaluation_query)["return"]

    assert typing.get_origin(annotation) is dict


def test_mcp_server_publishes_output_schemas_and_structured_content():
    server = MCPServer("test-eval")
    domain.install(server)

    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    for fn in MODEL_TOOLS:
        t = tools[fn.__name__]
        assert t.output_schema is not None, f"{t.name} missing output_schema"
        assert "properties" in t.output_schema, t.name


def test_mcp_server_image_tools_return_image_content_and_structured_payload(tmp_path):
    import polars as pl
    from arena_evaluation.mcp.reporting import (
        evaluation_report_draft,
        evaluation_report_plot,
    )

    run_dir = tmp_path / "benchmarks" / "run-img-01"
    run_dir.mkdir(parents=True)
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)

    from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
    from arena_evaluation.storage.registry.facts import FACT_TABLES

    de_schema = DIMENSION_TABLES["dim_episodes"].polars_schema()
    pl.DataFrame(
        [
            {col: None for col in de_schema.names()}
            | {
                "run_id": "run-img-01",
                "episode_id": eid,
                "planner": planner,
                "world": "w1",
                "scenario": "s1",
                "duration_s": dur,
            }
            for eid, planner, dur in [(0, "dwb", 10.0), (1, "mppi", 12.0)]
        ],
        schema=de_schema,
    ).write_parquet(run_dir / "dim_episodes.parquet")

    fe_schema = FACT_TABLES["episodes"].polars_schema()
    pl.DataFrame(
        [
            {col: None for col in fe_schema.names()}
            | {"run_id": "run-img-01", "episode_id": eid, "result": "COMPLETED"}
            for eid in (0, 1)
        ],
        schema=fe_schema,
    ).write_parquet(run_dir / "fact_episodes.parquet")

    dm_schema = DIMENSION_TABLES["dim_metrics"].polars_schema()
    pl.DataFrame(
        [
            {col: None for col in dm_schema.names()}
            | {
                "metric": "time_to_goal",
                "scope": "robots",
                "fact_table": "fact_robots",
                "dtype": "Float64",
                "unit": "s",
                "direction": "lower",
            }
        ],
        schema=dm_schema,
    ).write_parquet(run_dir / "dim_metrics.parquet")

    server = MCPServer("test-eval-img")
    domain.install(server)

    res_plot = asyncio.run(
        server.call_tool(
            "evaluation_render_plot",
            {
                "run_id": "run-img-01",
                "spec_yaml": (
                    "id: test_bar\ntitle: Test Bar\ntype: bar\n"
                    "data:\n  source: dim_episodes\n"
                    "channels:\n  x: episode_id\n  y: duration_s\n"
                ),
                "output_format": "png",
            },
        )
    )
    assert any(isinstance(c, ImageContent) for c in res_plot.content)
    assert res_plot.structured_content is not None
    assert res_plot.structured_content["format"] == "png"
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"

    draft_id = evaluation_report_draft(run_id="run-img-01", title="T")["draft_id"]
    evaluation_report_plot(
        run_id="run-img-01",
        draft_id=draft_id,
        plot={
            "id": "p1",
            "title": "P1",
            "type": "bar",
            "data": {"source": "dim_episodes"},
            "channels": {"x": "episode_id", "y": "duration_s"},
        },
    )
    res_view = asyncio.run(
        server.call_tool(
            "evaluation_report_view",
            {"run_id": "run-img-01", "draft_id": draft_id, "plot_id": "p1"},
        )
    )
    assert any(isinstance(c, ImageContent) for c in res_view.content)
    assert res_view.structured_content is not None
    assert res_view.structured_content["plot_id"] == "p1"


def test_evaluation_domain_registers_same_resources_and_prompts_as_domain():
    from arena_mcp.core import Domain
    from arena_evaluation.mcp.base import EvaluationDomain

    def make_pair(dom_cls):
        dom = dom_cls("toy", "toy domain")

        @dom.tool(read_only=True)
        def toy_ping() -> dict[str, int]:
            """Ping."""
            return {"ok": 1}

        @dom.resource(
            "arena://toy/info", mime_type="application/json", description="info"
        )
        def toy_info() -> str:
            return "{}"

        @dom.prompt(description="help")
        def toy_help() -> str:
            return "help"

        srv = MCPServer(f"srv-{dom_cls.__name__}")
        dom.install(srv)
        return srv

    base_srv = make_pair(Domain)
    eval_srv = make_pair(EvaluationDomain)

    async def inspect_pair():
        return (
            [str(r.uri) for r in await base_srv.list_resources()],
            [str(r.uri) for r in await eval_srv.list_resources()],
            [p.name for p in await base_srv.list_prompts()],
            [p.name for p in await eval_srv.list_prompts()],
        )

    b_res, e_res, b_prm, e_prm = asyncio.run(inspect_pair())
    assert b_res == e_res == ["arena://toy/info"]
    assert b_prm == e_prm == ["toy_help"]
