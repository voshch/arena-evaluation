"""Domain-wide contract (H-006): every evaluation tool returns an object, never a bare list."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

pytest.importorskip("arena_mcp")

import arena_evaluation.mcp as mcp_package


def _evaluation_tools() -> dict[str, object]:
    """Every evaluation_* function defined in the MCP package, keyed by tool name."""
    tools: dict[str, object] = {}
    for info in pkgutil.iter_modules(mcp_package.__path__):
        module = importlib.import_module(f"{mcp_package.__name__}.{info.name}")
        for name, obj in vars(module).items():
            if name.startswith("evaluation_") and inspect.isfunction(obj) and obj.__module__ == module.__name__:
                tools[name] = obj
    return tools


def test_the_tool_scan_finds_the_known_tools():
    """Sanity check: the scan sees the tools this contract is about, so an empty offender list means something."""
    tools = _evaluation_tools()
    assert {
        "evaluation_report",
        "evaluation_queries",
        "evaluation_finding_verify",
    } <= set(tools)


def test_no_evaluation_tool_returns_a_bare_list():
    """Every evaluation tool publishes an object, so MCP carries structured content for all of them."""
    offenders = sorted(name for name, tool in _evaluation_tools().items() if str(inspect.signature(tool).return_annotation).startswith("list["))

    assert offenders == []
