"""The storage layer imports nothing above it, in any of its subpackages."""

from __future__ import annotations

import ast
import pathlib

import arena_evaluation.storage as storage_package

_STORAGE_ROOT: pathlib.Path = pathlib.Path(storage_package.__file__).parent
_HIGHER_LAYERS: tuple[str, ...] = (
    "processing",
    "presentation",
    "analytics",
    "metrics",
    "benchmark",
    "mcp",
)
_HIGHER_MODULES: tuple[str, ...] = tuple(
    f"arena_evaluation.{layer}" for layer in _HIGHER_LAYERS
)


def _imports_of(path: pathlib.Path) -> list[str]:
    """Every module a file names, relative imports that climb out of storage as their absolute name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.ImportFrom) and node.level >= 2 and node.module:
            modules.append(f"arena_evaluation.{node.module}")
    return modules


def test_storage_imports_no_higher_layer_in_any_subpackage() -> None:
    violations = [
        f"{path.relative_to(_STORAGE_ROOT)} imports {module}"
        for path in sorted(_STORAGE_ROOT.rglob("*.py"))
        for module in _imports_of(path)
        if module.startswith(_HIGHER_MODULES)
    ]
    assert not violations, f"storage imports from a higher layer: {violations}"
