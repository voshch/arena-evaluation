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
#: The one accepted L1->L2 edge (decision X-007): the fact schema self-loads the metric definitions
#: (L1-019/X-001). The pin fixes the importing module and the exact symbol; any other
#: storage -> processing import fails, and moving the registry into storage is post-PR work.
_STORAGE_PINNED: dict[tuple[str, str], set[str]] = {
    (
        "layout.py",
        "arena_evaluation.processing.metrics.registry",
    ): {"load_metric_definitions"},
}
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
    (
        "plot_types/acoustic_field.py",
        "arena_evaluation.processing.acoustics.constants",
    ): {
        "MIC_DISTANCE_M",
        "ROBOT_ACOUSTICS_TOPIC",
        "WALL_TRANSMISSION_LOSS_DB",
    },
    (
        "plot_types/acoustic_field.py",
        "arena_evaluation.processing.acoustics.episode_series",
    ): {
        "episode_number",
        "load_aligned_episode",
    },
}

#: The one alignment implementation and the metric-layer module that re-exports it (FIX2).
_ALIGNMENT_MODULES: frozenset[str] = frozenset(
    {
        "processing/alignment.py",
        "processing/metrics/common/alignment.py",
    }
)
# Direct alignment sites that still bypass the one alignment module. Each is owned by a row
# that must route it through `alignment` (X-026); delete the entry once the file no longer
# aligns streams itself, the assertion below fails until then.
_KNOWN_DIRECT_ALIGNMENT_SITES: dict[str, str] = {
    "processing/metrics/social/encounter_costs.py": "X-026 / PR2",
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
    violations: list[str] = []
    seen_pinned: set[tuple[str, str]] = set()
    for path in sorted(_STORAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_STORAGE_ROOT).as_posix()
        for module, names in _imports_of(path):
            if not _is_internal(module) or _matches_allowed(module, _STORAGE_ALLOWED):
                continue
            key = (rel, module)
            if key in _STORAGE_PINNED and set(names) == _STORAGE_PINNED[key]:
                seen_pinned.add(key)
            else:
                violations.append(f"{rel} imports {module} ({list(names)})")
    assert not violations, f"storage imports outside allow-list: {violations}"
    assert seen_pinned == set(_STORAGE_PINNED), (
        f"pinned edge(s) missing: {set(_STORAGE_PINNED) - seen_pinned}"
    )


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


def test_streams_are_aligned_only_in_the_alignment_module() -> None:
    """Alignment is one module: presentation binds published series and nothing re-pairs streams itself."""
    violations: list[str] = []
    seen_known: set[str] = set()
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(_PKG_ROOT).as_posix()
        if rel in _ALIGNMENT_MODULES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"join_asof", "interp"}
        ]
        if not lines:
            continue
        if rel in _KNOWN_DIRECT_ALIGNMENT_SITES:
            seen_known.add(rel)
            continue
        violations.append(f"{rel}:{lines}")

    assert not violations, f"streams aligned outside {sorted(_ALIGNMENT_MODULES)}: {violations}"
    assert seen_known == set(_KNOWN_DIRECT_ALIGNMENT_SITES), (
        f"site(s) no longer aligning directly, delete their entry: {set(_KNOWN_DIRECT_ALIGNMENT_SITES) - seen_known}"
    )


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
_MCP_PINNED: dict[tuple[str, str], set[str]] = {
    ("analysis.py", "arena_evaluation.processing.sensitivity"): {"rerun_sensitivity"},
    ("migration.py", "arena_evaluation.processing.metrics.dictionary"): {
        "metric_dictionary"
    },
}


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


def _module_name(path: pathlib.Path) -> str:
    """The dotted name of a package module, ``__init__`` included as its package."""
    parts = list(path.relative_to(_PKG_ROOT).parts[:-1])
    if path.stem != "__init__":
        parts.append(path.stem)
    return ".".join(("arena_evaluation", *parts))


def _module_graph() -> dict[str, set[str]]:
    """Internal imports between package modules, resolved to module names."""
    modules = {_module_name(path): path for path in sorted(_PKG_ROOT.rglob("*.py"))}
    known = set(modules)
    graph: dict[str, set[str]] = {name: set() for name in known}
    for name, path in modules.items():
        for target, names in _imports_of(path):
            for candidate in (target, *(f"{target}.{alias}" for alias in names)):
                if not _is_internal(candidate):
                    continue
                resolved = candidate
                while resolved not in known and "." in resolved:
                    resolved = resolved.rsplit(".", 1)[0]
                if resolved in known:
                    graph[name].add(resolved)
    return graph


def _strongly_connected(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's strongly connected components of ``graph``, each component sorted."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    counter = 0

    def visit(node: str) -> None:
        nonlocal counter
        index[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in sorted(graph[node]):
            if neighbour not in index:
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
            elif neighbour in on_stack:
                low[node] = min(low[node], index[neighbour])
        if low[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return components


#: The reviewed module import cycles; the metrics one is the type-only X-008 cycle.
_ALLOWED_CYCLES: frozenset[frozenset[str]] = frozenset(
    {
        # Type-only (TYPE_CHECKING) pair: exports annotates the builder it exports.
        frozenset(
            {
                "arena_evaluation.presentation.exports",
                "arena_evaluation.presentation.report_builder",
            }
        ),
        # The manifest identifier loads the model it identifies; the model resolves the
        # default report by name. Presentation-internal, no cross-layer edge (X-005).
        frozenset(
            {
                "arena_evaluation.presentation.manifest_identifier",
                "arena_evaluation.presentation.viz_manifest",
            }
        ),
        frozenset(
            {
                "arena_evaluation.processing.metrics.base",
                "arena_evaluation.processing.metrics.ecological.mechanism_common",
            }
        ),
        frozenset(
            {
                "arena_evaluation.analytics.analyses",
                "arena_evaluation.analytics.analyses.procedures",
                "arena_evaluation.analytics.catalog",
                "arena_evaluation.analytics.computations",
                "arena_evaluation.analytics.derived",
                "arena_evaluation.analytics.ranking",
                "arena_evaluation.analytics.sql_engine",
                "arena_evaluation.analytics.stats",
            }
        ),
        frozenset(
            {
                "arena_evaluation.benchmark.cli",
                "arena_evaluation.benchmark.dashboard",
                "arena_evaluation.benchmark.dashboard_runtime",
                "arena_evaluation.benchmark.runner",
                "arena_evaluation.mcp.runs",
            }
        ),
        frozenset(
            {
                "arena_evaluation.processing.metrics",
                "arena_evaluation.processing.metrics.registry",
            }
        ),
    }
)


def test_module_import_graph_has_only_reviewed_cycles() -> None:
    """Pin every module-granularity cycle so a new one fails review before it becomes architecture."""
    cycles = {
        frozenset(component)
        for component in _strongly_connected(_module_graph())
        if len(component) > 1
    }
    unreviewed = cycles - _ALLOWED_CYCLES
    assert not unreviewed, (
        f"new module import cycle(s): {sorted(sorted(cycle) for cycle in unreviewed)}"
    )
    resolved = _ALLOWED_CYCLES - cycles
    assert not resolved, (
        "reviewed cycle(s) no longer present; update _ALLOWED_CYCLES: "
        f"{sorted(sorted(cycle) for cycle in resolved)}"
    )
