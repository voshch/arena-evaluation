"""MCP Presentation Gateway Tools (Layer 5)."""

from __future__ import annotations

import base64
import dataclasses
import html
import json
import pathlib
from typing import Annotated, Any, Literal

import polars.exceptions
import matplotlib

matplotlib.use("Agg")
import pydantic
import yaml
from arena_mcp.core import ToolError
from mcp.types import CallToolResult, ImageContent, TextContent

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.evidence import EvidenceStore
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.analytics.sql_engine import PolarsSQLEngine
from arena_evaluation.presentation.plotly_renderer import PlotlyRenderer
from arena_evaluation.presentation.resolver import (
    DataResolver,
    deep_merge,
    find_recipe_path,
    recipe_search_dirs,
)
from arena_evaluation.presentation.static_renderer import StaticRenderer
from arena_evaluation.presentation.viz_manifest import PlotSpec
from arena_evaluation.storage.config_layout import (
    bundle_path,
    get_config_kind,
    run_analysis_root,
)
from arena_evaluation.storage.star_schema import has_star_schema

from .base import (
    OptionalRunId,
    YamlText,
    assert_safe_analysis_write,
    atomic_write_text,
    component,
    domain,
    parse_yaml,
    run_path,
)

__all__ = [
    "evaluation_plot_declarations",
    "evaluation_read_plot_declaration",
    "evaluation_render_plot",
    "evaluation_save_plot_declaration",
]


def _plot_bundle_files(root: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Return ``(name, plot_yaml_path)`` entries inside one config root's ``plots/`` directory."""
    plots_dir = root / get_config_kind("plot").folder
    if not plots_dir.is_dir():
        return []
    plot_filename = get_config_kind("plot").filename
    entries: list[tuple[str, pathlib.Path]] = []
    for sub in sorted(plots_dir.iterdir()):
        if sub.is_dir():
            candidate = sub / plot_filename
            if candidate.is_file():
                entries.append((sub.name, candidate))
    return entries


@domain.tool(read_only=True)
def evaluation_plot_declarations() -> list[dict[str, Any]]:
    """List all registered declarative plot declarations and recipes with encodings and data sources."""
    results: dict[str, dict[str, Any]] = {}
    for directory in recipe_search_dirs():
        for name, file_path in _plot_bundle_files(directory):
            if name in results:
                continue
            try:
                raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or "type" not in raw:
                    continue
                spec = PlotSpec.model_validate(raw)
                source_val = spec.data.source if spec.data else None
                stored_query_val = (
                    spec.data.stored_query
                    if (spec.data and spec.data.stored_query)
                    else None
                )
                results[name] = {
                    "id": spec.id or name,
                    "title": spec.title,
                    "type": spec.type,
                    "source": source_val,
                    "stored_query": stored_query_val,
                    "encodings": spec.channels.model_dump(exclude_unset=True),
                    "options": dict(spec.options),
                }
            except (yaml.YAMLError, pydantic.ValidationError, ValueError, OSError):
                continue
    return [results[k] for k in sorted(results)]


@domain.tool(read_only=True)
def evaluation_read_plot_declaration(
    name: Annotated[
        str,
        pydantic.Field(description="Name or stem of registered plot declaration"),
    ],
    run_id: OptionalRunId = None,
) -> dict[str, Any]:
    """Read a registered plot declaration YAML document and its validated specification."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    extra = (run_analysis_root(run_path(run_id)),) if run_id is not None else ()
    path = find_recipe_path(safe_name, extra_dirs=extra)
    if path is None:
        raise ToolError(
            f"Plot declaration {name!r} not found",
            hint="evaluation_plot_declarations lists available declarations",
        )

    text = path.read_text(encoding="utf-8")
    parsed = parse_yaml(text, f"plot declaration '{safe_name}'")
    if not isinstance(parsed, dict):
        raise ToolError(
            f"Plot declaration '{safe_name}' must be a YAML mapping",
            hint="make the declaration a YAML mapping with id, type and data",
        )

    try:
        spec = PlotSpec.model_validate(parsed)
    except (yaml.YAMLError, pydantic.ValidationError, ValueError, OSError) as exc:
        raise ToolError(
            f"Invalid PlotSpec schema in declaration '{safe_name}': {exc}",
            hint="fix the PlotSpec fields; evaluation_plot_declarations shows a valid declaration",
        ) from exc

    return {
        "name": safe_name,
        "yaml": text,
        "spec": spec.model_dump(),
    }


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_save_plot_declaration(
    name: Annotated[
        str,
        pydantic.Field(description="Unique identifier for the plot declaration"),
    ],
    yaml_text: YamlText,
    *,
    run_id: OptionalRunId = None,
) -> dict[str, Any]:
    """Validate against PlotSpec and save a plot declaration recipe to the run's analysis/plots or package catalog."""
    safe_name = component(name.removesuffix(".yaml").removesuffix(".yml"), "name")
    parsed = parse_yaml(yaml_text, "yaml_text")
    if not isinstance(parsed, dict):
        raise ToolError(
            "Plot declaration YAML must be a mapping",
            hint="make yaml_text a YAML mapping with id, type and data",
        )

    try:
        spec = PlotSpec.model_validate(parsed)
    except (yaml.YAMLError, pydantic.ValidationError, ValueError, OSError) as exc:
        raise ToolError(
            f"Invalid PlotSpec schema: {exc}",
            hint="fix the PlotSpec fields; evaluation_plot_declarations shows a valid document",
        ) from exc

    if run_id is not None:
        file_path = bundle_path(run_analysis_root(run_path(run_id)), "plot", safe_name)
    else:
        dirs = recipe_search_dirs()
        target_root = next((d for d in dirs if d.is_dir()), dirs[0])
        file_path = bundle_path(target_root, "plot", safe_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(file_path, yaml_text.strip() + "\n")

    return {
        "status": "saved",
        "name": safe_name,
        "path": str(file_path),
        "id": spec.id or safe_name,
        "type": spec.type,
        "title": spec.title,
    }


@domain.tool(read_only=False, destructive=True, idempotent=True)
def evaluation_render_plot(
    run_id: OptionalRunId = None,
    *,
    declaration: Annotated[
        str | None,
        pydantic.Field(
            description="Name of registered plot recipe/declaration to render"
        ),
    ] = None,
    spec_yaml: Annotated[
        str | None,
        pydantic.Field(description="Inline YAML string defining a PlotSpec to render"),
    ] = None,
    overrides: Annotated[
        dict[str, Any] | None,
        pydantic.Field(
            description="Optional dictionary to deep-merge over declaration"
        ),
    ] = None,
    output_format: Annotated[
        Literal["png", "svg", "pdf", "html"],
        pydantic.Field(description="Desired output format"),
    ] = "png",
) -> CallToolResult:
    """Render a plot recipe or inline spec directly into a standalone artifact file with quality diagnostics."""
    if declaration is None and spec_yaml is None:
        raise ToolError(
            "Must provide either 'declaration' or 'spec_yaml'",
            hint="evaluation_plot_declarations lists available recipes",
        )
    if declaration is not None and spec_yaml is not None:
        raise ToolError(
            "Cannot provide both 'declaration' and 'spec_yaml'",
            hint="pass declaration or spec_yaml, not both",
        )

    benchmark_path = run_path(run_id)
    if not has_star_schema(benchmark_path):
        raise ToolError(
            f"run {benchmark_path.name!r} has no Star Schema tables",
            hint="evaluation_process builds them",
        )

    if declaration is not None:
        safe_decl = component(
            declaration.removesuffix(".yaml").removesuffix(".yml"), "declaration"
        )
        recipe_path = find_recipe_path(
            safe_decl, extra_dirs=(run_analysis_root(benchmark_path),)
        )
        if recipe_path is None:
            raise ToolError(
                f"Plot declaration {declaration!r} not found",
                hint="evaluation_plot_declarations lists available recipes",
            )
        raw_recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
        if not isinstance(raw_recipe, dict):
            raise ToolError(
                f"Recipe '{safe_decl}' is not a valid YAML mapping",
                hint="make the recipe a YAML mapping with id, type and data",
            )
        merged = deep_merge(raw_recipe, overrides or {})
        try:
            spec = PlotSpec.model_validate(merged)
        except (yaml.YAMLError, pydantic.ValidationError, ValueError) as exc:
            raise ToolError(
                f"Invalid PlotSpec schema in recipe '{safe_decl}': {exc}",
                hint="fix the recipe's PlotSpec fields; evaluation_plot_declarations shows a valid recipe",
            ) from exc
    else:
        parsed = parse_yaml(spec_yaml or "", "spec_yaml")
        if not isinstance(parsed, dict):
            raise ToolError(
                "spec_yaml must be a YAML mapping",
                hint="make spec_yaml a YAML mapping with id, type and data",
            )
        merged = deep_merge(parsed, overrides or {})
        try:
            spec = PlotSpec.model_validate(merged)
        except (yaml.YAMLError, pydantic.ValidationError, ValueError) as exc:
            raise ToolError(
                f"Invalid PlotSpec schema: {exc}",
                hint="fix the PlotSpec fields; evaluation_plot_declarations shows a valid document",
            ) from exc

    catalog = BenchmarkCatalog.open(benchmark_path)
    sql_engine = PolarsSQLEngine(catalog)
    evidence_store = EvidenceStore(benchmark_path)
    q_catalog = QueryCatalog(run_dir=benchmark_path)

    resolver = DataResolver(
        catalog=catalog,
        sql_engine=sql_engine,
        evidence_store=evidence_store,
        query_catalog=q_catalog,
        benchmark_dir=benchmark_path,
    )

    try:
        df, diagnostics = resolver.resolve(spec)
    except (
        KeyError,
        ValueError,
        OSError,
        RuntimeError,
        polars.exceptions.PolarsError,
    ) as exc:
        raise ToolError(
            f"Failed to resolve data for plot '{spec.id}': {exc}",
            hint="check the spec's data source with evaluation_read_metrics or evaluation_evidence",
        ) from exc

    if df.is_empty():
        plot_id = spec.id or declaration or "plot"
        raise ToolError(
            f"Data source or query for plot '{plot_id}' returned 0 rows; cannot render plot with an empty dataset",
            hint="Verify filter conditions or run 'evaluation_query' to check row counts.",
        )

    plots_dir = run_analysis_root(benchmark_path) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    clean_stem = component(spec.id or declaration or "plot", "plot_name")
    out_file = plots_dir / f"{clean_stem}.{output_format}"
    assert_safe_analysis_write(benchmark_path, out_file)

    if output_format == "html":
        import plotly.offline

        plotly_renderer = PlotlyRenderer(units={})
        try:
            html_body = plotly_renderer.render(spec, df, run_dir=benchmark_path)
        except (
            ValueError,
            KeyError,
            RuntimeError,
            polars.exceptions.PolarsError,
        ) as exc:
            raise ToolError(
                f"Cannot render plot {spec.id!r}: {exc}",
                hint="check PlotSpec type, channels, and data source; raw-row bindings require valid aggregation or x/y columns",
            ) from exc
        if html_body is None:
            raise ToolError(
                f"Plotly renderer failed to produce output for '{spec.id}'",
                hint="use output_format='html' only for a plotly-compatible spec; try png",
            )
        if isinstance(html_body, list):
            html_body = "\n".join(html_body)


        plotly_js = plotly.offline.get_plotlyjs()
        full_html = (
            f"<!DOCTYPE html>\n<html>\n<head>\n<title>{html.escape(spec.title or clean_stem)}</title>\n"
            f"<script>{plotly_js}</script>\n"
            "</head>\n<body style='margin: 0; padding: 12px; font-family: sans-serif;'>\n"
            f"{html_body}\n</body>\n</html>\n"
        )
        atomic_write_text(out_file, full_html)
    else:
        static_renderer = StaticRenderer(units={})
        try:
            static_renderer.render(
                spec,
                df,
                out_path=out_file,
                run_dir=benchmark_path,
                diagnostics=diagnostics,
            )
        except (
            ValueError,
            KeyError,
            RuntimeError,
            polars.exceptions.PolarsError,
        ) as exc:
            raise ToolError(
                f"Cannot render plot {spec.id!r}: {exc}",
                hint="check PlotSpec type, channels, and data source; raw-row bindings require valid aggregation or x/y columns",
            ) from exc
        if not out_file.is_file() or out_file.stat().st_size == 0:
            detail = f": {'; '.join(diagnostics.issues)}" if diagnostics.issues else ""
            raise ToolError(
                f"Static renderer failed to produce output at {out_file}{detail}",
                hint="check the plot data and try output_format='png'",
            )

    summary_payload: dict[str, Any] = {
        "plot_path": str(out_file),
        "format": output_format,
        "spec_id": spec.id or clean_stem,
        "diagnostics": dataclasses.asdict(diagnostics),
    }
    content: list[Any] = [TextContent(type="text", text=json.dumps(summary_payload))]
    if output_format == "png":
        encoded = base64.b64encode(out_file.read_bytes()).decode("ascii")
        content.append(ImageContent(type="image", data=encoded, mimeType="image/png"))
    return CallToolResult(content=content, structuredContent=summary_payload)
