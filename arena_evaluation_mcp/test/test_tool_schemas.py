"""Validate all tool input_schema definitions and the dispatch seam."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")
pytest.importorskip("polars")
pytest.importorskip("arena_evaluation")
pytest.importorskip("arena_evaluation_mcp")


def _get_tools():
    from arena_evaluation_mcp.eval_bridge import EvalBridge
    from arena_evaluation_mcp.tools import build_tools_list

    bridge = EvalBridge()
    return build_tools_list(bridge)


def _validate_json_schema(schema: dict) -> list[str]:
    """Basic JSON Schema validation. Returns list of issues."""
    issues = []
    if not isinstance(schema, dict):
        return ["schema is not a dict"]
    if schema.get("type") != "object":
        issues.append("schema type is not 'object'")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        issues.append("properties is not a dict")
    req = schema.get("required", [])
    if req is not None and not isinstance(req, list):
        issues.append("required is not a list")
    # Check that required fields exist in properties
    if isinstance(req, list) and isinstance(props, dict):
        for r in req:
            if r not in props:
                issues.append(f"required field '{r}' not in properties")
    return issues


class TestToolSchemas:
    """Validate all tool input_schema definitions."""

    def test_sdk_field_names(self):
        """Attribute is input_schema; the wire alias is inputSchema."""
        for tool in _get_tools():
            assert isinstance(tool.input_schema, dict), f"Tool '{tool.name}' lacks input_schema"
            dumped = tool.model_dump(by_alias=True)
            assert "inputSchema" in dumped, f"Tool '{tool.name}' lacks inputSchema alias"

    def test_all_tools_have_valid_schemas(self):
        tools = _get_tools()
        assert len(tools) >= 28, f"Expected at least 28 tools, got {len(tools)}"

        for tool in tools:
            issues = _validate_json_schema(tool.input_schema or {})
            assert not issues, f"Tool '{tool.name}' has schema issues: {issues}"

    def test_tool_names_are_unique(self):
        tools = _get_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names)), f"Duplicate tool names: {names}"

    def test_discovery_tools_have_no_required_fields(self):
        """Discovery tools that take no arguments should have empty required list."""
        discovery_tools = [
            "list_available_maps",
            "list_available_robots",
            "list_available_planners",
            "list_available_inter_planners",
            "list_available_task_modes",
            "list_available_manifests",
            "list_available_metrics",
        ]
        tools = {t.name: t for t in _get_tools()}
        for name in discovery_tools:
            assert name in tools, f"Missing discovery tool: {name}"
            schema = tools[name].input_schema or {}
            required = schema.get("required", [])
            assert not required, f"{name} should have no required fields, got {required}"

    def test_configure_tools_require_name_and_yaml_content(self):
        """create_suite, create_contest need name + yaml_content."""
        tools = {t.name: t for t in _get_tools()}
        for name in ["create_suite", "create_contest"]:
            assert name in tools
            required = tools[name].input_schema.get("required", [])
            for field in ["name", "yaml_content"]:
                assert field in required, f"{name} must require '{field}'"

    def test_validate_tools_require_yaml_content(self):
        """All validate_* tools need yaml_content."""
        tools = {t.name: t for t in _get_tools()}
        for name in ["validate_suite", "validate_contest", "validate_manifest"]:
            assert name in tools
            required = tools[name].input_schema.get("required", [])
            assert "yaml_content" in required, f"{name} must require 'yaml_content'"

    def test_execute_tools_have_benchmark_id(self):
        """run_processing and run_report need benchmark_id."""
        tools = {t.name: t for t in _get_tools()}
        for name in ["run_processing", "run_report"]:
            if name in tools:
                required = tools[name].input_schema.get("required", [])
                assert "benchmark_id" in required, f"{name} must require 'benchmark_id'"

    def test_run_benchmark_requires_suite_and_contest(self):
        tools = {t.name: t for t in _get_tools()}
        tool = tools.get("run_benchmark")
        assert tool is not None
        required = tool.input_schema.get("required", [])
        for field in ["suite", "contest"]:
            assert field in required, f"run_benchmark must require '{field}'"

    def test_manifest_tool_enum_matches_available(self):
        """The template enum in create_manifest should be a non-empty list."""
        tools = {t.name: t for t in _get_tools()}
        tool = tools.get("create_manifest")
        assert tool is not None
        props = tool.input_schema.get("properties", {})
        template = props.get("template", {})
        enum_vals = template.get("enum", [])
        assert len(enum_vals) >= 3, f"Expected at least 3 manifest templates, got {enum_vals}"

    def test_notes_tool_mode_enum(self):
        tools = {t.name: t for t in _get_tools()}
        tool = tools.get("write_notes")
        assert tool is not None
        props = tool.input_schema.get("properties", {})
        mode = props.get("mode", {})
        assert set(mode.get("enum", [])) == {"replace", "append", "merge"}


class TestDispatchSeam:
    def test_unknown_tool_is_structured_error(self):
        from mcp.types import CallToolRequestParams

        from arena_evaluation_mcp.eval_bridge import EvalBridge
        from arena_evaluation_mcp.tools import dispatch_tool_call

        bridge = EvalBridge()
        params = CallToolRequestParams(name="zz_no_such_tool", arguments={})
        result = asyncio.run(dispatch_tool_call(params, bridge))
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert "unknown tool" in payload["error"]

    def test_traversal_is_structured_error(self):
        from mcp.types import CallToolRequestParams

        from arena_evaluation_mcp.eval_bridge import EvalBridge
        from arena_evaluation_mcp.tools import dispatch_tool_call

        bridge = EvalBridge()
        params = CallToolRequestParams(
            name="read_notes",
            arguments={"benchmark_id": "../escape"},
        )
        result = asyncio.run(dispatch_tool_call(params, bridge))
        assert result.is_error
        payload = json.loads(result.content[0].text)
        assert "invalid name" in payload["error"]

    def test_kill_processes_tool_schema_and_dispatch(self, monkeypatch):
        from mcp.types import CallToolRequestParams
        from arena_evaluation_mcp.eval_bridge import EvalBridge
        from arena_evaluation_mcp.tools import dispatch_tool_call

        bridge = EvalBridge()
        monkeypatch.setattr(bridge, "kill_processes", lambda **kwargs: [{"pid": 999, "status": "killed"}])

        params = CallToolRequestParams(name="kill_processes", arguments={"pids": [999], "force": True})
        result = asyncio.run(dispatch_tool_call(params, bridge))
        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["results"] == [{"pid": 999, "status": "killed"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
