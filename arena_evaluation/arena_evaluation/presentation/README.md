# presentation (Layer 4: Report and Plot Generation)

Reads the Star Schema lakehouse through the benchmark catalog and produces interactive
HTML reports, static PNG/SVG plots, and document exports offline.

---

## Files

| File | Purpose |
|---|---|
| `report_builder.py` | `ReportBuilder`: resolves manifests, loads data sources, renders `report.html` and `plots/` |
| `manifest_registry.py` | Resolves report manifests by name, path, or inline YAML |
| `viz_manifest.py` | `VizManifest`, `PlotSpec`, `PlotDataSpec`, `PlotChannels`, `PlotSliceSpec`, `PlotChecks` models |
| `data_sources.py` | Declared Star Schema tables, canonical views, and series views |
| `predicate.py` | AST-to-Polars compiler for `data.where` row predicates |
| `resolver.py` | `DataResolver` for views, SQL queries, computations, evidence, and recipes |
| `checks.py` | `DiagnosticInspector` technical quality gate, `PlotRenderError`, `image_is_blank` |
| `report_plot_block.py` | One plot block (interactive figure + static PNG) and the inline/sibling Plotly bundle |
| `lint.py` | `lint_declarations`: schema-valid lint and per-manifest bindability against a run |
| `exports.py` | `export_report_latex`: escaped booktabs tables and figures of an approved report |
| `legacy_manifest.py` | Deterministic converter of pre-overhaul report declarations |
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

# Animated trajectory GIFs (opt-in; report-level flag, wired by the CLI/MCP)
python - <<'PY'
from pathlib import Path
from arena_evaluation.presentation.report_builder import ReportBuilder

ReportBuilder(
    Path("/opt/arena_ws/data/my_benchmark"),
    output_dir=Path("./out"),
    manifest="standard",
    generate_gifs=True,
).build()
PY
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
builder.export_latex(Path("report.tex"))       # approved tables + figures, escaped
print(builder.build_checks.failed_count, builder.build_checks.failures)

# External-asset report: plotly.min.js sibling + <img src="plots/<id>.png"> instead of inline bytes
ReportBuilder.from_dirs(
    source_dirs=[Path("/opt/arena_ws/data/my_benchmark")],
    output_dir=Path("./out"),
    manifest="everything",
    external_assets=True,
).build()
```

The MCP reviewed loop (`mcp/reporting.py`) builds through this same builder: one page writer, one
`BuildChecks` appendix, one LaTeX export. `evaluation_report_assemble` adds the verified findings,
the excluded/not-used findings appendix and the requested exports on top of the builder page.

---

## Declarative Report Manifests

Report layouts are defined in `configs/reports/<name>/report.yaml`:

| Manifest | Focus |
|---|---|
| `standard` | Default benchmark report (safety, efficiency, motion, smoothness, social, ecological) |
| `everything` | Cross-domain matrix of every shipped view and metric |
| `failure_modes` | Diagnostic triage, verification tables, per-domain checks |
| `ecological` | Energy consumption, power splits, battery, transport cost |
| `social` | Proxemics, gaze, pedestrian interaction, acoustic comfort |
| `safety` | Success rate, collisions, trade-offs, trajectories |
| `characterization` | Open-loop energy/acoustic profiles (series telemetry) |

### Resolution Precedence (`manifest_registry.py`)

1. Inline YAML (starts with `{` or `[`).
2. Explicit YAML file path (a missing path raises `FileNotFoundError`; it never falls back).
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

`summary` is the single page-level "Performance Summary" table; `table` **plots** are
in-section tables. Use one mechanism per report: a manifest with `summary` should not also
repeat the same metrics as an `overview_table`, and a manifest without `summary` should
carry an explicit table plot where the headline numbers belong.

`units` holds only metrics the report actually draws; regenerate it from the metric
registry rather than hand-writing names (a stale entry silently mislabels an axis).

### PlotSpec Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique plot identifier and PNG filename |
| `type` | `str` | Plot type (`violin`, `box`, `bar`, `histogram`, `scatter`, `trajectory`, `radar`, `heatmap`, `timeseries`, `line`, `table`, `hierarchy`, `interval`, `rank_swap`, `breakdown`, `trajectory_slice`, `acoustic_field`, `acoustic_field_animation`) |
| `title` | `str` | Plot title |
| `uses` | `str \| None` | Plot bundle name from `plots/<name>/plot.yaml` |
| `data` | `PlotDataSpec \| None` | Binding (`source`, `query`, `stored_query`, `computation`, `params`, `evidence_id`, `where`) |
| `channels` | `PlotChannels` | Visual channels (`x`, `y`, `color`, `facet_col`, `facet_row`, `size`, `symbol`, `hierarchy`) |
| `slice` | `PlotSliceSpec \| None` | Entity slice (`episode_id`, `encounter_id`, `contact_id`) |
| `options` | `dict` | Renderer options |
| `checks` | `PlotChecks` | Declared check expectations (`require_columns`, `min_rows`, `allow_all_null`); an unknown key is rejected |
| `layout_group` | `str \| None` | Report section identifier |

Declared channels are also the implicit `require_columns` set: a missing declared column
is a quality-gate failure, not a renderer crash.

---

## Plot Types

- `violin` / `box`: Distribution of a metric per group.
- `bar`: Mean +/- error bar chart (`err_plus`/`err_minus` options; `barmode` for grouping).
- `histogram`: Binned distribution (`distribution_bins`, `bin_low`/`bin_high`).
- `scatter`: X-Y scatter with optional colour/size channels.
- `timeseries`: 1NF series telemetry (`series_profile`) against the recorded time axis.
- `line`: Series profile lines with optional markers (`show_markers`).
- `table`: Declarative table of the bound frame (scorecards, summaries).
- `trajectory`: Robot or pedestrian paths over `dim_world_geometry`; `generate_gifs: true` also writes an animated GIF.
- `radar`: Normalized multi-metric profile.
- `heatmap`: Correlation matrix or pivot grid.
- `hierarchy`: Treemap / sunburst hierarchical breakdown.
- `interval`: Point estimates with pre-computed bounds (`options.low`/`options.high`), optionally dodged per `color`.
- `rank_swap`: Pre-computed rank-swap counts/rates per entity.
- `breakdown`: Stacked attribution shares per `x` and `color` component.
- `trajectory_slice`: One pre-recorded XY path per frame.
- `acoustic_field`: Spatial acoustic field, shipped via `configs/plots/acoustic_field/plot.yaml` and bound in `everything`.
- `acoustic_field_animation`: Animated acoustic field (registered; not bound in a shipped report, see L4-010).
  Both acoustic classes expose their static artifact as `render_seaborn` (D-06) and `render_static` delegates to it.

---

## Quality Gate

`DiagnosticInspector` runs before any renderer:

1. the resolved frame is non-empty;
2. every `require_columns` and every declared channel column exists;
3. no declared numeric channel is **100 % NULL** — the issue names the metric and the
   recorded `<column>_null_reason` (or the row-level `null_reason`);
4. `checks.min_rows` holds.

After the static render, the builder rejects the plot when no file was written, when a
renderer returns without writing (`StaticRenderer` raises `PlotRenderError`), or when the
PNG is one flat colour. Every failure is a `PlotCheck` in the report appendix and in
`BuildResult.checks`; `failed_count` counts declared plots only, while `has_failures`
also covers the page-level summary table.

---

## Exports

- `report.html` + `plots/*.png` (+ `*.svg` via `options.vector_svg`) + inline Plotly bundle.
- External-asset mode (`external_assets=True`): the bundle moves to a sibling `plotly.min.js` and every
  figure becomes `<img src="plots/<id>.png">`, so the HTML holds markup instead of megabytes of
  base64 + script.
- Size budget: **inline ≤ 300 KB per plot** (the `everything` report measured 21.6 MB / 84 plots =
  251 KB/plot, ~96 % of it base64 PNGs); use `external_assets=True` for reports over ~15 MB or
  ~50 plots. External mode measured **0.74 MB HTML + 4.85 MB `plotly.min.js`** for the same 84 plots
  (9 KB/plot of HTML) with the PNGs unchanged on disk (12 MB) — a ~29× smaller page.
- `ReportBuilder.export_latex(path)` writes the **approved** tables and rendered figures
  of the last build as one LaTeX document: booktabs tables with every `_ & % $ # { } ~ ^ \`
  escaped, and `\includegraphics` blocks for the PNGs. The MCP reviewed-report loop
  (`mcp/reporting.py`) calls this same exporter.

---

## Legacy migration

The pre-overhaul manifests (`configs/benchmark/manifests/*.yaml`) are rejected by the
current grammar. Convert them deterministically:

```bash
python scripts/migrate_legacy_manifests.py configs/benchmark/manifests/social.yaml --dry-run
python scripts/migrate_legacy_manifests.py configs/benchmark/manifests/*.yaml --output-dir /tmp/migrated
python scripts/migrate_legacy_manifests.py --legacy-plots plot_declarations/all_metrics.yaml
python scripts/reconcile_legacy_manifests.py old/social.yaml configs/reports/social/report.yaml
```

Each conversion writes `<name>/report.yaml` (a skeleton) plus `migration_report.yaml`
that accounts for **every** old plot as `converted`, `dropped` (with a reason) or `todo`
(with the binding that must still be decided). Old metric names are translated through
`presentation/legacy_metric_map.py` (or an external `--metric-map` YAML with the same
`renamed:`/`removed:` shape); a name that is neither mapped nor in the metric registry
stays a `TODO:` — the converter never guesses a rename. The map ships inside the package
rather than under `configs/legacy/` because the config root admits only the five
registered bundle kinds.

The old idiom for a grouping dimension was `differentiate: <dim>` (sometimes implicit
via `auto_differentiate`); the new idiom is an explicit channel:

```yaml
# old
data_key: time_to_goal
differentiate: local_planner
# new
data:
  source: v_episode_summary
channels:
  x: planner
  y: time_to_goal
```

`group_by` becomes `channels.facet_col` (bar/scatter) or `params.group_cols`
(aggregations); `filter` becomes `data.where` or `params.filter`; `secondary_data_key`
becomes a second `y` entry or `options.y`.

---

## Adding a Plot Type

1. Create `plot_types/<mytype>.py`, subclass `BasePlotRenderer`, and define `PLOT_TYPE`.
2. Implement `build_plotly_figure` / `build_static_figure` (the public `render_plotly` /
   `render_static` surfaces call them) and raise `PlotRenderError` when the frame cannot
   be drawn.
3. Register the class in `plot_types/__init__.py` (`RENDERER_REGISTRY`) and add one
   dispatch test through `StaticRenderer` + `PlotlyRenderer`.
4. Preserve nulls cleanly and declare the checks in `PlotChecks` / `DiagnosticInspector`.
5. The two acoustic classes keep D-06 explicit by implementing `render_seaborn` and letting
   `render_static` delegate to it; other types implement `render_static` directly.
