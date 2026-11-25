"""Every ``raise ToolError(...)`` in the MCP domain names the next action it wants."""

from __future__ import annotations

import ast
import pathlib

MCP_DIR = pathlib.Path(__file__).resolve().parents[2] / "arena_evaluation" / "mcp"


def _raise_sites() -> list[tuple[str, int, bool]]:
    """(file, line, has_hint) for every ``raise ToolError(...)`` in the MCP domain modules."""
    sites: list[tuple[str, int, bool]] = []
    for path in sorted(MCP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            function = node.exc.func
            name = function.id if isinstance(function, ast.Name) else (function.attr if isinstance(function, ast.Attribute) else "")
            if name not in ("ToolError", "_query_error"):
                continue
            sites.append(
                (
                    path.name,
                    node.lineno,
                    ("hint" in {keyword.arg for keyword in node.exc.keywords}) or (name == "_query_error"),
                )
            )
    return sites


def test_every_tool_error_raise_carries_a_hint() -> None:
    sites = _raise_sites()
    assert len(sites) >= 95, f"expected the MCP domain's ToolError raises, found {len(sites)}"
    missing = [f"{name}:{line}" for name, line, has_hint in sites if not has_hint]
    assert not missing, "raise ToolError without hint= at " + ", ".join(missing)
