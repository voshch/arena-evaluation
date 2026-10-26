# presentation (Layer 5: Report and Plot Generation)

Reads processed metric data (Parquet) and produces interactive HTML reports, static PNG plots, and optional animated GIFs offline.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder`: resolves manifests, loads data sources, renders `report.html` and `plots/` |
| `manifest_registry.py` | Resolves report manifests by name, path, or inline YAML |
| `viz_manifest.py` | `VizManifest` and `PlotSpec` models and default manifest loader |
| `plotly_renderer.py` | Dispatches `PlotSpec` to interactive chart renderers |
| `seaborn_renderer.py` | Dispatches `PlotSpec` to static PNG renderers |
| `color_utils.py` | Accessibility color palette loader and global plotting styles |
| `dimension_detector.py` | Auto-detects varying dimensions and builds compound labels |
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

Report layouts are defined in `configs/benchmark/manifests/*.yaml`:

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
3. Named manifest in `configs/benchmark/manifests/<name>.yaml`.
4. Legacy `benchmark_dir/viz_manifest.yaml`.
5. `report_manifest.yaml` in benchmark directory.
6. Default `standard`.

### Manifest Schema

```yaml
manifest_version: "1.0"
name: characterization
title: Open-Loop Characterization Report
data_source: characterization_samples
groups:
  - {id: power_curves, title: Power vs. Velocity Curves}
summary:
  - {metric: mean_power_total_w, label: Mean Power, format: "{:.1f}"}
summary_group_by: [phase_kind, robot]
units:
  p_total_w: W
  leq_af_dba: dBA
plots: [ ... ]
```

### PlotSpec Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique plot identifier and PNG filename |
| `type` | `str` | Plot type (`violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`, `timeseries`, `line`, `table`) |
| `title` | `str` | Plot title |
| `data_key` | `str` | Column name in data source (or `"*"` for multi-metric plots) |
| `group_by` | `str \| list[str]` | Grouping columns |
| `differentiate` | `str` | Color differentiation column (default: `"planner"`) |
| `auto_differentiate` | `bool` | Auto-detect varying dimensions and construct compound labels |
| `filter` | `dict` | Row filter before plotting (e.g. `{"is_reference": false}`) |
| `options` | `dict` | Renderer options |
| `layout_group` | `str` | Report section identifier |
| `data_source` | `str \| None` | Per-plot data source override |

---

## Plot Types

- `violin` / `box`: Distribution of a metric per group.
- `bar`: Mean +/- std bar chart (supports `stacked: true`).
- `histogram`: Smooth area distribution per group.
- `scatter`: X-Y scatter (expands list columns).
- `timeseries`: Wide-format per-episode list columns vs time axis.
- `line`: Long-format line charts with optional confidence bands (`error_y`).
- `table`: Declarative HTML summary table.
- `trajectory`: paths over the world map, broken at teleports (resets, elevator rides, stair crossings), with doors and elevator cabins drawn from the world. Options: `show_map`, `overlay_markers`, `overlay_pedestrians`, `overlay_theta_star`, `overlay_reference`, `overlay_mechanisms` (default on), `facet` (a column, one panel per value), `crop` (view limited to the tracks, default on with `facet`). With `--generate-gifs` the static plot is replayed on sim time: paths grow, pedestrians walk, doors and cabin doors follow the recorded `mechanism_states`, goal and collision marks appear when their episode ends. Doors animate only on a panel that holds one episode, so `facet: local_planner` compares planners.
- `radar`: Normalized multi-metric profile.
- `heatmap`: Correlation matrix or pivot grid.

---

## Adding a Plot Type

1. Create `plot_types/<mytype>.py`, subclass `BasePlotRenderer`, and define `PLOT_TYPE`.
2. Implement `render_plotly(df)` and `render_seaborn(df, out_path)`.
3. Register class in `plot_types/__init__.py`, `plotly_renderer.py`, and `seaborn_renderer.py`.
4. Ensure renderers handle empty or missing columns safely.

