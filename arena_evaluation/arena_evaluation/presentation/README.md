# presentation (Layer 4: Report and Plot Generation)

Reads processed metric data (Parquet) and produces interactive HTML reports and static PNG/SVG plots offline.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder`: resolves manifests, loads data sources, renders `report.html` and `plots/` |
| `manifest_registry.py` | Resolves report manifests by name, path, or inline YAML |
| `viz_manifest.py` | `VizManifest`, `PlotSpec`, `PlotDataSpec`, `PlotChannels`, `PlotSliceSpec` models |
| `data_sources.py` | Declared Star Schema tables, canonical views, and series views |
| `predicate.py` | AST-to-Polars compiler for `data.where` row predicates |
| `resolver.py` | `DataResolver` for views, SQL queries, computations, evidence, and recipes |
| `plotly_renderer.py` | Dispatches `PlotSpec` to interactive chart renderers |
| `static_renderer.py` | Dispatches `PlotSpec` to static seaborn/Matplotlib PNG renderers |
| `color_utils.py` | Accessibility color palette loader and global plotting styles |
| `report_template.html.j2` | Jinja2 HTML template for reports |
| `plot_types/` | Individual chart type implementations |

---

## Usage

```bash
# Standard metrics report
arena evaluation report --benchmark-dir /opt/arena_ws/data/my_benchmark

# Named declarative manifest
arena evaluation run --benchmark-dir /opt/arena_ws/data/my_benchmark --report-manifest characterization

# List available named manifests
arena evaluation run --list-manifests

# Merge multiple benchmarks into one report
arena evaluation report --benchmark-dir /opt/arena_ws/data/bench1 /opt/arena_ws/data/bench2 --output-dir ./merged_report
```

Programmatically:

```python
from arena_evaluation.presentation.report_builder import ReportBuilder
from pathlib import Path

builder = ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/my_benchmark")],
    output_dir=Path("."),
    manifest="characterization",
)
builder.build()
```

---

## Declarative Report Manifests

Report layouts are defined in `configs/reports/<name>/report.yaml`:

| Manifest | Focus |
|---|---|
| `standard` | Default benchmark report (safety, efficiency, motion, smoothness, social, ecological) |
| `ecological` | Energy consumption, power splits, battery behavior |
| `social` | Proxemics, gaze, pedestrian interaction |
| `safety` | Success rate, collisions, trajectories |
| `characterization` | Open-loop energy/acoustic profiles |

### Resolution Precedence (`manifest_registry.py`)

1. Inline YAML (starts with `{` or `[`).
2. Explicit YAML file path.
3. Run-local bundle in `<run>/analysis/reports/<name>/report.yaml`.
4. Named bundle via asset chain (`<imported>`, `<share>/configs/reports/<name>/report.yaml`, `<source>/configs/reports/<name>/report.yaml`).
5. Default `standard` resolved through the same chain.

### Manifest Schema

```yaml
manifest_version: "1.0"
name: characterization
title: Open-Loop Characterization Report
data_source: v_series_characterization
groups:
  - {id: power_curves, title: Power vs. Velocity Curves}
summary:
  - {metric: power_total_w, label: Mean Power, format: "{:.1f}"}
summary_group_by: [phase_kind, robot_model]
units:
  power_total_w: W
  leq_af_dba: dBA
plots: [ ... ]
```

### PlotSpec Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique plot identifier and PNG filename |
| `type` | `str` | Plot type (`violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`, `timeseries`, `line`, `table`, `hierarchy`) |
| `title` | `str` | Plot title |
| `uses` | `str \| None` | Plot bundle name from `plots/<name>/plot.yaml` |
| `data` | `PlotDataSpec \| None` | Binding (`source`, `query`, `stored_query`, `computation`, `params`, `evidence_id`, `where`) |
| `channels` | `PlotChannels` | Visual channels (`x`, `y`, `color`, `facet_col`, `facet_row`, `size`, `symbol`, `split`, `hierarchy`) |
| `slice` | `PlotSliceSpec \| None` | Entity or window slice (`scope`, `episode_id`, `encounter_id`, `contact_id`) |
| `options` | `dict` | Renderer options |
| `checks` | `dict` | Technical check expectations (`require_columns`) |
| `layout_group` | `str \| None` | Report section identifier |

---

## Plot Types

- `violin` / `box`: Distribution of a metric per group.
- `bar`: Mean +/- std bar chart (supports `stacked: true`).
- `histogram`: Smooth area distribution per group.
- `scatter`: X-Y scatter (expands list columns).
- `timeseries`: Wide-format per-episode list columns vs time axis.
- `line`: Long-format line charts with optional confidence bands (`error_y`).
- `table`: Declarative HTML summary table.
- `trajectory`: Robot or pedestrian paths rendered over `dim_world_geometry` (with optional `world` filter).
- `radar`: Normalized multi-metric profile.
- `heatmap`: Correlation matrix or pivot grid.
- `hierarchy`: Treemap / sunburst hierarchical breakdown.

---

## Adding a Plot Type

1. Create `plot_types/<mytype>.py`, subclass `BasePlotRenderer`, and define `PLOT_TYPE`.
2. Implement `render_plotly(df, ...)` and `render_static(df, out_path, ...)`.
3. Register the class in `plot_types/__init__.py` (`PLOT_RENDERERS`).
4. Ensure renderers preserve nulls cleanly and handle empty or missing columns via `PlotChecks` / `DiagnosticInspector`.

