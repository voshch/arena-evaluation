# arena_evaluation
> ROS 2 package for recording, processing, and visualising navigation and ecological evaluation metrics for Arena robots.

Evaluation pipeline converting simulation data into structured metric reports across recording, processing, and presentation layers.

---

Suites take their task-mode names and params from the selected task runtime, see
[task runtimes](../../arena_runtime/arena_runtime/arena_runtime/env/README.md#task-runtimes).
The characterization metric rebuilds the default schedule of a recording that
carries none from `arena_robots.characterization`, the module the
`characterization` robot mode publishes from. Both read the envelope from the
robot instance's recorded caps snapshot, so a robot with cap overrides is swept and
scored with its own limits.

## Architecture Overview

```
+---------------------------------------------------------------------+
|                          Arena Simulation                           |
|  Gazebo / task_generator / nav2 / planner / task mode               |
+-------------------------+-------------------------------------------+
                         | ROS 2 topics
                         v
+---------------------------------------------------------------------+
|  Layer 1 - Ingestion  (ingestion/)                                  |
|  DataRecorderNode writes one MCAP file per episode into             |
|  benchmark/episodes/episode_XXX/. Episode lifecycle (start/stop) is |
|  driven by the benchmark runner via the `start_episode` service.    |
|  Output: episodes/episode_XXX/episode_XXX.mcap + .yaml              |
+-------------------------+-------------------------------------------+
                         | MCAP file (one per episode)
                         v
+---------------------------------------------------------------------+
|  Layer 1 - Processing  (processing/)                                |
|  MCAPReader -> native tables (topics/*.parquet) -> dimension rows   |
|  -> dim_*.parquet at the benchmark root, once per benchmark.        |
|  Each episode is read once; telemetry keeps its native timestamps.  |
|  Layer 2 metric calculators populate fact tables and series parquet.   |
|  Output: topics/*.parquet per episode, dim_*.parquet per benchmark  |
+-------------------------+-------------------------------------------+
                         | Parquet files
                         v
+---------------------------------------------------------------------+
|  Layer 4 - Presentation  (presentation/)                            |
|  ReportBuilder reads a single-grammar report.yaml,                  |
|  resolves plot data per BenchmarkCatalog via DataResolver (views,   |
|  SQL queries, or analytics computations), checks PlotDiagnostics    |
|  via DiagnosticInspector, and renders HTML + static PNG/SVG plots.  |
|  Output: reports/<name>/report.html + plots/*.png (Plotly inline)   |
+---------------------------------------------------------------------+
```

**Layer 3, Analysis (`analytics/`):** `BenchmarkCatalog.open(benchmark_dir)` registers the Star Schema dimensions, facts, telemetry, series, and `evidence_index` alongside the six canonical lazy views (`v_episode_summary`, `v_planner_scorecard`, `v_contacts`, `v_contact_summary`, `v_encounter_summary`, `v_pedestrian_summary`) and the `v_series_*` views. Guarded read-only SQL (`PolarsSQLEngine`), registered computations (`analytics/computations.py`), and the content-addressed `EvidenceStore` persist query/computation results under `analysis/evidence/<evidence_id>.parquet` indexed at `analysis/evidence_index.parquet`.

---

## Directory Structure

```
arena_evaluation/
|-- arena_evaluation/          <- Python package root
|   |-- ingestion/             <- Layer 1: Live recording (ROS node)
|   |   |-- recorder.py        <- DataRecorderNode (service-driven episode lifecycle)
|   |   |-- episode_writer.py  <- EpisodeWriter (episode folder, MCAP writer, metadata finalize)
|   |   |-- metadata.py        <- IngestionMetadata (per-episode metadata)
|   |   |-- migrate.py         <- Migration of runs recorded before the native layout
|   |   `-- topics.py          <- Topic definitions and type registry
|   |-- storage/               <- Layer 1: Registry, dimensions and path management
|   |   |-- registry/          <- One Table class per stored table (telemetry, dimensions, facts)
|   |   |-- star_schema.py     <- Dimension tables and benchmark-wide lazy scans from the registry
|   |   |-- contracts.py       <- Dimension contract (schema, keys, reasons of NULL)
|   |   |-- schemas.py         <- RunMetadata, EpisodeDescriptor, RobotParams, TopicBundle
|   |   |-- folder_manager.py  <- Path resolution and episode discovery
|   |   |-- manifest.py        <- MetadataWriter (YAML read/write)
|   |   |-- planner_names.py   <- split_planner_name helper
|   |   |-- data_root.py       <- benchmarks root and latest benchmark helpers
|   |   |-- atomic.py          <- Atomic file replacement and locks
|   |   `-- exceptions.py      <- Package exception types
|   |-- processing/            <- Layer 1: Extraction and dimension rows
|   |   |-- mcap_reader.py     <- MCAPReader: one recording into flat tables, one bundle per robot
|   |   |-- telemetry/         <- Shapes recorded messages into flat rows, written as ZSTD Parquet
|   |   |-- parquet_store.py   <- TopicParquetStore: the extraction cache of an episode
|   |   |-- pose_anchor.py     <- Dense, sparse or absent ground-truth pose stream
|   |   |-- dimension_rows.py  <- The dimension rows of one episode
|   |   |-- map_registry.py    <- The map image of a world, resolved and cached
|   |   |-- pipeline.py        <- ProcessingPipeline: extracts, then writes the benchmark's dimensions
|   |   |-- metrics/           <- Metric calculators (layer 2)
|   |   |-- acoustics/         <- Acoustic propagation solver and door geometry (layer 2)
|   |   |-- path/              <- Reference path solver (layer 2)
|   |   `-- pose_segments.py   <- Teleport detection (layer 2)
|   |-- analytics/             <- Layer 3: BenchmarkCatalog, canonical views, guarded SQL, query store
|   |-- presentation/          <- Layer 4: Report and plot generation
|   |   |-- report_builder.py  <- Generates report.html and plots
|   |   |-- manifest_registry.py <- Resolves report manifests
|   |   |-- viz_manifest.py    <- VizManifest model and default loader
|   |   |-- plotly_renderer.py <- Interactive HTML chart dispatcher
|   |   |-- static_renderer.py <- Static PNG/SVG chart dispatcher
|   |   |-- color_utils.py     <- Palette loader and styling
|   |   |-- report_template.html.j2 <- Jinja2 report template
|   |   `-- plot_types/        <- acoustic_field, bar, box, breakdown, heatmap, hierarchy, histogram, interval, line, radar, rank_swap, scatter, table, timeseries, trajectory, trajectory_slice, violin (17 types)
|   |-- benchmark/             <- Benchmark runner and CLI
|   |   |-- runner.py          <- BenchmarkRunner
|   |   |-- config.py          <- Suite and Contest parsers
|   |   |-- state.py           <- Run manifest, progress, resume
|   |   |-- step.py            <- Step grid model
|   |   |-- debug.py           <- Process introspection
|   |   |-- profiler.py        <- PipelineProfiler (CPU, GPU, RAM)
|   |   `-- cli.py             <- Benchmark management CLI
|   |-- mcp/                   <- evaluation domain of the Arena MCP server
|   |-- cli.py                 <- Evaluation pipeline CLI
|   `-- cli_acoustic.py        <- Acoustic analysis subcommands
|-- config/
|   |-- data_recorder_config.yaml  <- Topic throttle frequencies
|   `-- mcap_writer_options.yaml   <- MCAP writer tuning
|-- configs/
|   |-- color_palette.yaml          <- Color palette (flat config asset)
|   |-- suites/<name>/suite.yaml     <- Suite definitions
|   |-- contests/<name>/contest.yaml <- Contest definitions
|   |-- reports/<name>/report.yaml   <- Report manifests (standard, ecological, social, safety, reference, characterization, failure_modes, everything)
|   |-- plots/<name>/plot.yaml       <- Reusable plot declarations (`uses: <name>`)
|   `-- queries/<name>/query.yaml    <- Stored SQL queries (`stored_query: <name>`)
`-- tests/
    |-- unit/                  <- Unit tests (no ROS required)
    |-- integration/           <- Integration tests
    `-- test_benchmark_*.py    <- Benchmark runner tests

arena_evaluation_msgs/         <- Sibling package: ROS 2 message and service definitions
|-- msg/BenchmarkState.msg     <- Live benchmark progress
`-- srv/                       <- RecordEpisode, ChangeDirectory services
```

### Robot facts

The recorder writes the effective caps of every robot instance to `<run_dir>/robots/<robot ns>/caps_snapshot.yaml` (`arena-caps/1`: `effective`, `provenance`, `hash`). The metrics read the mobile, power and acoustics models of a robot only from that snapshot through `RobotParams`, and the recorder subscribes the controller topics from its `body` block. The radius and mass are the recorded fleet parameters. A unit test fails on any read of robot definition files by path outside the migration. See the [ingestion](arena_evaluation/ingestion/README.md#robot-caps-snapshot) and [storage](arena_evaluation/storage/README.md) READMEs.

---

## Workflow

### 1. Run a benchmark (live simulation)

```bash
arena evaluation benchmark --suite basic --contest basic
```

Outputs are written to `$ARENA_DATA_DIR/benchmarks/<run_id>/episodes/episode_XXX/`.

Open-loop characterization:

```bash
arena evaluation benchmark --suite characterization --contest characterization
```

Edge cases: generated scenarios run as `scenario` stages against the `edge_case_differential` (jackal) or `edge_case_omnidirectional` (mpo700) contest, and the `edge_case` metric family scores them from the recording.

```bash
arena evaluation benchmark --suite <edge-case suite> --contest edge_case_differential
```

### 2. Extract (offline)

```bash
arena evaluation extract --benchmark-dir <run_id>
```

Reads every episode's MCAP once into its `topics/` folder and writes the benchmark's `dim_*.parquet` at its root.

### 3. Process metrics

```bash
arena evaluation process --benchmark-dir <run_id>
arena evaluation process --run-dir <run>/episodes/episode_004
```

Writes the per-episode `metrics/fact_*.parquet` and `metrics/series_*.parquet`, then the benchmark's
dimensions and contract report. The extraction cache is reused; `--force-extract` re-reads the MCAP,
and `--force-process` recomputes a run whose tables already carry the current layout stamp, replacing every episode's fact tables before the contract judges them.

### 4. Report

```bash
arena evaluation report --benchmark-dir <run_id> --report-manifest standard
arena evaluation report --benchmark-dir A B --output-dir ./merged_report
```

Writes `reports/report.html` and `reports/plots/*.png`, plus GIF side artefacts with
`--generate-gifs`; `--export svg|pdf|latex` also exports the approved plots and tables. There is no
`metrics.parquet`/`combined_metrics.parquet` any more: reports read the Star Schema through
`BenchmarkCatalog`.

### 5. Migrate a run recorded before the native layout

```bash
arena evaluation migrate <run dir> --dry-run
arena evaluation migrate <run dir> --report migration_report.yaml
```

See [docs/migrate.md](docs/migrate.md) for the supported generations, the NULL-and-reason mapping, the
descriptor synthesis rule, `migration_report.yaml` and the limits.

Report manifests are declarative YAML bundles in `configs/reports/<name>/report.yaml`; the seven
built-in ones are `standard`, `ecological`, `social`, `safety`, `characterization`, `failure_modes`
and `everything`.

```bash
arena evaluation run --list-manifests
arena evaluation run --benchmark-dir X --report-manifest ecological
arena evaluation run --benchmark-dir X --report-manifest '{name: inline, plots: [...]}'
```

---

## CLI Reference

```
usage: evaluation <command> [flags]

Commands:
  extract   Layer 1: extract topics from MCAP into Parquet and write the benchmark dimensions
  process   Layer 2: compute metrics and write the Star Schema fact and series tables
  run       Full pipeline: extract -> process -> report + plots
  report    Generate report.html from the existing Star Schema tables
  plot      Generate static PNG plots only
  migrate   Convert a run recorded before the native layout, in place
  validate  Check a suite or contest definition without running it
  acoustic  Acoustic analysis subcommands (list, animate, snapshot)
```

| Flag | Description |
|---|---|
| `--benchmark-dir DIR...` | Benchmark root directories: `extract`, `process`, `report` and `plot` handle every episode (and every robot) under them |
| `--run-dir DIR...` | Single `episode_NNN` directories: `process` and `extract` handle each alone, `report` and `plot` merge their Star Schema tables |
| `--output-dir DIR` | Output directory for reports and plots (defaults to the first input dir) |
| `--workers N` | Worker count for parallel extraction and processing (`-1` = auto CPU count) |
| `--profile` | Write `pipeline_profile.yaml` with per-phase resource stats |
| `--force-extract` | Force re-extraction of MCAP files (`process`) |
| `--force-process` | Recompute a run whose Star Schema tables already carry the current layout stamp (`process`) |
| `--report-manifest NAME\|PATH\|{...}` | Report layout: named manifest, YAML path, or inline YAML (`run`, `report`, `plot`) |
| `--list-manifests` | List available named report manifests |
| `--export {svg,pdf,latex}` | Also export approved plots as SVG/PDF and/or the approved tables and figures as LaTeX (repeatable) |
| `--generate-gifs` | Render each trajectory or animation plot's GIF side artefact |
| `--dry-run`, `--force`, `--report PATH` | `migrate` only: detect without writing, re-extract anyway, write the report elsewhere |
| `--suite FILE` \| `--contest FILE` | `validate` only: check one suite or contest bundle without running it |

`process` and `run` exit 1 when a benchmark or episode directory produced no metrics; a contract
violation exits non-zero and names the table and column.

---

## MCP domain

The package contributes the `evaluation` domain to the Arena MCP server (`arena_evaluation/mcp/`, registered under the `arena_mcp.domains` entry point). `python3 -m arena_mcp ls evaluation` lists its tools with one line each.

- Catalogs come from the live registries and resolvers: task modes and their parameter schemas from the task-mode registry, metrics and their declared directions from the metric registry, drivers from the bringup and adapter registries, nav2 planner configs from `arena_robots`, bridge planners from `arena_planners`, worlds, suites, contests and manifests through their `Identifier` resolvers (with `network=true` the buckets too).
- Validation runs the runner's own parsers (`Suite.parse`, `Contest.parse` plus the runner's nav2 preflight, `VizManifest`, `parse_scenario`). Created suites, contests and manifests go to the write target of their resolver chain, a manifest for one run into that run directory, scenarios into the world the map name resolves to.
- Benchmarks (`arena evaluation benchmark`), preflights and soaks (`arena planners test`), processing, extraction, reports, plots and acoustic renders run as jobs of the real CLI verbs. A benchmark always gets an explicit run id, and the start waits until the runner prepared its run directory, so a config error comes back as an error. Poll jobs with `jobs_status` and `jobs_log`, stop a benchmark with `evaluation_stop`, which interrupts its runner process group and escalates only after a grace window.
- Run state is read from `$ARENA_DATA_DIR/benchmarks/<run_id>` with the package's readers: the manifest's steps, the union of every lane's state file and the folded progress files, so shared runs show their lanes while they run. Analysis tools group by contestant (the `planner` column) and leave reference episodes out unless asked, and they rank only over outputs whose calculator declares a `DIRECTIONS` entry (69 tools in the domain today).
- Recording control calls an env's recorder (`<task generator>/start_episode`) and reads `<task generator>/state/recorded_topics`, env-scoped tools take `env_id`.

Tests live in `tests/mcp/` and drive a real `python3 -m arena_mcp serve` over stdio against run directories built with the package's writers.

The domain is implemented via `EvaluationDomain` (`arena_evaluation/mcp/base.py`), which publishes Pydantic `BaseModel` output schemas and `structuredContent` alongside `ImageContent` plot views. Role allowlists and agent-area destructive tools (`<run>/analysis/`, `<run>/reports/`) are declared in `arena_evaluation/mcp/roles.py` (`ROLES`, `AGENT_AREA_TOOLS`), guarded by `assert_safe_analysis_write` and logged to `<run>/analysis/audit.jsonl`. Tests discovering `arena_mcp.domains` require the package entry point (`arena_evaluation.mcp:domain`) to be visible on `PYTHONPATH`.

---

## Building the C++ solver libraries

Two C++17 sources are loaded through `ctypes` and must be compiled before an acoustic metric or a route plan can run:

| source | artefact | loaded by |
|---|---|---|
| `arena_evaluation/processing/acoustics/solver.cpp` | `arena_evaluation/processing/acoustics/solver.so` | `processing/acoustics/impedance_grid.py` |
| `arena_evaluation/processing/path/solver.cpp` | `arena_evaluation/processing/path/solver.so` | `processing/path/theta_star.py` |

`arena build` compiles both: colcon installs this package with `setup.py develop`, and the `build_ext` command in `setup.py` configures `CMakeLists.txt` with `-DARENA_SOLVERS_ONLY=ON` and builds the two targets into the package tree. The same two libraries can be built on their own:

```bash
cmake -S arena_evaluation -B build/solvers -DARENA_SOLVERS_ONLY=ON
cmake --build build/solvers --parallel
```

This needs `cmake` and a C++17 compiler (`g++` or `clang++`), both part of the Arena environment. The loaded library exports `arena_acoustic_solver_abi`, which the wrapper verifies before binding: a library built from another model raises `MetricInputError` instead of publishing a different attenuation. `ARENA_ACOUSTIC_SOLVER_SO` overrides the file that is loaded and is the supported way to use a library built elsewhere.

---

## Running Tests

The unit tests use the solver libraries above and need no running ROS graph:

```bash
pytest tests/unit -v
pytest tests/ -v
```

---

## Packaging

`arena build` (colcon) and `setup.py build`/`install` are the supported build paths. A pure-Python wheel is not supported: `pyproject.toml` sets `package = false` for `uv`, and the two solver libraries are platform-specific artefacts produced by the build, not by wheel metadata.

---

## Sub-package Documentation

- [ingestion/README.md](arena_evaluation/ingestion/README.md): recording, topics, episode lifecycle, migration
- [processing/README.md](arena_evaluation/processing/README.md): extraction, native tables, pose stream, dimension rows
- [processing/metrics/README.md](arena_evaluation/processing/metrics/README.md): metric calculator framework
- [docs/migrate.md](docs/migrate.md): recording and report-declaration migration
- [presentation/README.md](arena_evaluation/presentation/README.md): declarative manifests, plot types
- [storage/README.md](arena_evaluation/storage/README.md): registry, dimensions and contracts, paths, metadata
- [benchmark/README.md](arena_evaluation/benchmark/README.md): runner, state, CLI, profiler
- [configs/README.md](configs/README.md): suites, contests, reports, plots, queries
- [arena_evaluation_msgs/README.md](../arena_evaluation_msgs/README.md): ROS 2 message and service definitions

