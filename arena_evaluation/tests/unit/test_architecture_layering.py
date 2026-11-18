"""Architecture layering guards with strict allow-lists and pinned cross-layer edges."""

from __future__ import annotations

import ast
import pathlib

import arena_evaluation.analytics as analytics_package
import arena_evaluation.experiments as experiments_package
import arena_evaluation.presentation as presentation_package
import arena_evaluation.storage as storage_package

_PKG_ROOT: pathlib.Path = pathlib.Path(storage_package.__file__).parents[1]
_STORAGE_ROOT: pathlib.Path = pathlib.Path(storage_package.__file__).parent
_ANALYTICS_ROOT: pathlib.Path = pathlib.Path(analytics_package.__file__).parent
_EXPERIMENTS_ROOT: pathlib.Path = pathlib.Path(experiments_package.__file__).parent
_PRESENTATION_ROOT: pathlib.Path = pathlib.Path(presentation_package.__file__).parent

_STORAGE_ALLOWED: tuple[str, ...] = ("arena_evaluation.storage",)
_ANALYTICS_ALLOWED: tuple[str, ...] = (
    "arena_evaluation.storage",
    "arena_evaluation.analytics",
)
_EXPERIMENTS_ALLOWED: tuple[str, ...] = (
    "arena_evaluation.storage",
    "arena_evaluation.analytics",
    "arena_evaluation.experiments",
)
_EXPERIMENTS_PINNED: dict[tuple[str, str], set[str]] = {
    ("beckon.py", "arena_evaluation.benchmark.config"): {"_parse_duration"},
}
_PRESENTATION_ALLOWED: tuple[str, ...] = (
    "arena_evaluation.storage",
    "arena_evaluation.analytics",
    "arena_evaluation.presentation",
)
_PRESENTATION_PINNED: dict[tuple[str, str], set[str]] = {
    ("manifest_registry.py", "arena_evaluation.benchmark.tree"): {"ManifestIdentifier"},
    (
        "plot_types/acoustic_field.py",
        "arena_evaluation.processing.acoustics.door_map",
    ): {
        "_entity_matches_door",
        "barrier_door_tl",
        "build_pixel_tl",
        "load_barrier_map",
    },
    (
        "plot_types/acoustic_field.py",
        "arena_evaluation.processing.acoustics.door_state",
    ): {
        "DoorStateTimeline",
    },
    (
        "plot_types/acoustic_field.py",
        "arena_evaluation.processing.acoustics.impedance_grid",
    ): {
        "compute_acoustic_field",
        "downsample_mask",
        "downsample_occupancy",
    },
}


def _resolve_package(path: pathlib.Path, level: int) -> str:
    rel = path.relative_to(_PKG_ROOT)
    parts = rel.parts[:-1]
    if level > 1:
        parts = parts[: max(0, len(parts) - (level - 1))]
    return ".".join(("arena_evaluation", *parts))


def _imports_of(path: pathlib.Path) -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(module_name, imported_names)`` for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, ()))
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            if node.level == 0 and node.module:
                imports.append((node.module, names))
            elif node.level >= 1:
                pkg = _resolve_package(path, node.level)
                target = f"{pkg}.{node.module}" if node.module else pkg
                imports.append((target, names))
    return imports


def _is_internal(module: str) -> bool:
    return module == "arena_evaluation" or module.startswith("arena_evaluation.")


def _matches_allowed(module: str, allowed: tuple[str, ...]) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in allowed
    )


def test_storage_imports_only_storage() -> None:
    violations = [
        f"{path.relative_to(_STORAGE_ROOT)} imports {module}"
        for path in sorted(_STORAGE_ROOT.rglob("*.py"))
        for module, _ in _imports_of(path)
        if _is_internal(module) and not _matches_allowed(module, _STORAGE_ALLOWED)
    ]
    assert not violations, f"storage imports outside allow-list: {violations}"


def test_analytics_imports_only_storage_and_analytics() -> None:
    violations = [
        f"{path.relative_to(_ANALYTICS_ROOT)} imports {module}"
        for path in sorted(_ANALYTICS_ROOT.rglob("*.py"))
        for module, _ in _imports_of(path)
        if _is_internal(module) and not _matches_allowed(module, _ANALYTICS_ALLOWED)
    ]
    assert not violations, f"analytics imports outside allow-list: {violations}"


def test_experiments_imports_only_allowed_layers_and_pinned_edge() -> None:
    violations: list[str] = []
    seen_pinned: set[tuple[str, str]] = set()
    for path in sorted(_EXPERIMENTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_EXPERIMENTS_ROOT).as_posix()
        for module, names in _imports_of(path):
            if not _is_internal(module) or _matches_allowed(
                module, _EXPERIMENTS_ALLOWED
            ):
                continue
            key = (rel, module)
            if key in _EXPERIMENTS_PINNED and set(names) == _EXPERIMENTS_PINNED[key]:
                seen_pinned.add(key)
            else:
                violations.append(f"{rel} imports {module} ({list(names)})")

    assert not violations, f"experiments imports outside allow-list: {violations}"
    assert seen_pinned == set(_EXPERIMENTS_PINNED), (
        f"pinned edge(s) missing: {set(_EXPERIMENTS_PINNED) - seen_pinned}"
    )


def test_presentation_imports_only_allowed_layers_and_pinned_edge() -> None:
    violations: list[str] = []
    seen_pinned: set[tuple[str, str]] = set()
    for path in sorted(_PRESENTATION_ROOT.rglob("*.py")):
        rel = path.relative_to(_PRESENTATION_ROOT).as_posix()
        for module, names in _imports_of(path):
            if not _is_internal(module) or _matches_allowed(
                module, _PRESENTATION_ALLOWED
            ):
                continue
            key = (rel, module)
            if key in _PRESENTATION_PINNED and set(names) == _PRESENTATION_PINNED[key]:
                seen_pinned.add(key)
            else:
                violations.append(f"{rel} imports {module} ({list(names)})")

    assert not violations, f"presentation imports outside allow-list: {violations}"
    assert seen_pinned == set(_PRESENTATION_PINNED), (
        f"pinned edge(s) missing: {set(_PRESENTATION_PINNED) - seen_pinned}"
    )


_FORBIDDEN_STAT_ATTRS: frozenset[str] = frozenset(
    {
        "mean",
        "std",
        "var",
        "median",
        "quantile",
        "percentile",
        "corr",
        "corrcoef",
        "group_by",
        "histogram",
        "histplot",
        "kdeplot",
        "regplot",
        "lmplot",
        "Histogram",
        "pivot_table",
        "nan_to_num",
        "linkage",
        "theta_star",
    }
)
_FORBIDDEN_STAT_MODULES: tuple[str, ...] = ("scipy.stats", "scipy.cluster")


def test_presentation_computes_no_statistics() -> None:
    violations: list[str] = []
    for path in sorted(_PRESENTATION_ROOT.rglob("*.py")):
        rel = path.relative_to(_PRESENTATION_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, _ in _imports_of(path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_STAT_MODULES
            ):
                violations.append(f"{rel} imports forbidden stats module {module}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in _FORBIDDEN_STAT_ATTRS:
                        violations.append(
                            f"{rel}:{node.lineno} calls .{node.func.attr}()"
                        )
                    if node.func.attr == "fill_null" and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Constant) and arg0.value in (0, 0.0):
                            violations.append(
                                f"{rel}:{node.lineno} calls .fill_null({arg0.value!r})"
                            )
                elif (
                    isinstance(node.func, ast.Name)
                    and node.func.id in _FORBIDDEN_STAT_ATTRS
                ):
                    violations.append(f"{rel}:{node.lineno} calls {node.func.id}()")
                for kw in node.keywords:
                    if kw.arg == "estimator" and not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is None
                    ):
                        violations.append(
                            f"{rel}:{node.lineno} passes estimator= (non-None)"
                        )
    assert not violations, (
        f"presentation must bind, not compute statistics: {violations}"
    )


_MCP_ROOT: pathlib.Path = _PKG_ROOT / "mcp"
_MCP_ALLOWED: tuple[str, ...] = (
    "arena_evaluation.storage",
    "arena_evaluation.analytics",
    "arena_evaluation.presentation",
    "arena_evaluation.benchmark",
    "arena_evaluation.mcp",
)
_MCP_PINNED: dict[tuple[str, str], set[str]] = {}


def test_mcp_imports_only_allowed_layers() -> None:
    violations: list[str] = []
    seen_pinned: set[tuple[str, str]] = set()
    for path in sorted(_MCP_ROOT.rglob("*.py")):
        rel = path.relative_to(_MCP_ROOT).as_posix()
        for module, names in _imports_of(path):
            if not _is_internal(module) or _matches_allowed(module, _MCP_ALLOWED):
                continue
            key = (rel, module)
            if key in _MCP_PINNED and set(names) == _MCP_PINNED[key]:
                seen_pinned.add(key)
            else:
                violations.append(f"{rel} imports {module} ({list(names)})")
    assert not violations, f"mcp imports outside allow-list: {violations}"
    assert seen_pinned == set(_MCP_PINNED), (
        f"pinned edge(s) missing: {set(_MCP_PINNED) - seen_pinned}"
    )


def test_mcp_reads_no_parquet_directly() -> None:
    violations: list[str] = []
    for path in sorted(_MCP_ROOT.rglob("*.py")):
        rel = path.relative_to(_MCP_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("read_parquet", "scan_parquet", "write_parquet")
            ):
                violations.append(f"{rel}:{node.lineno} calls .{node.func.attr}()")
    assert not violations, (
        f"mcp must use BenchmarkCatalog, not direct Parquet I/O: {violations}"
    )


_NON_MCP_TO_MCP_PINNED: set[tuple[str, str]] = {
    ("benchmark/dashboard_runtime.py", "arena_evaluation.mcp.runs"),
    ("benchmark/dashboard_runtime.py", "arena_evaluation.mcp.catalog"),
    ("benchmark/dashboard_runtime.py", "arena_evaluation.mcp.authoring"),
}


def test_non_mcp_packages_never_import_mcp() -> None:
    violations = [
        f"{path.relative_to(_PKG_ROOT).as_posix()} imports {module}"
        for path in sorted(_PKG_ROOT.rglob("*.py"))
        if not path.is_relative_to(_MCP_ROOT)
        for module, _ in _imports_of(path)
        if (
            module == "arena_evaluation.mcp"
            or module.startswith("arena_evaluation.mcp.")
        )
        and (path.relative_to(_PKG_ROOT).as_posix(), module)
        not in _NON_MCP_TO_MCP_PINNED
    ]
    assert not violations, f"non-mcp modules import mcp: {violations}"
