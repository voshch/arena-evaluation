# arena_evaluation
> ROS 2 package for recording, processing, and visualising navigation and ecological evaluation metrics for Arena robots.

Evaluation pipeline converting simulation data into structured metric reports across recording, processing, and presentation layers.

---

Suites take their task-mode names and params from the selected task runtime, see
[task runtimes](../../arena_runtime/arena_runtime/arena_runtime/env/README.md#task-runtimes).
The characterization metric rebuilds the default schedule of a recording that
carries none from `arena_robots.characterization`, the module the
`characterization` robot mode publishes from.

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
|  Layer 2 metric calculators and their report return with PR2, PR4.  |
|  Output: topics/*.parquet per episode, dim_*.parquet per benchmark  |
+-------------------------+-------------------------------------------+
                         | Parquet files
                         v
+---------------------------------------------------------------------+
|  Layer 5 - Presentation  (presentation/)                            |
|  ReportBuilder reads parquet selected by a declarative manifest     |
|  (configs/benchmark/manifests/*.yaml) and generates HTML reports,   |
|  static PNG plots, and optional GIFs.                               |
|  Output: report.html + plots/*.png (+ plots/*.gif)                  |
+-------------------------+-------------------------------------------+
```

**Layer 3 — Analysis (`analytics/`):** `BenchmarkCatalog.open(benchmark_dir)` registers the Star Schema dimensions, facts, telemetry, series, and `evidence_index` alongside canonical lazy views (`v_episode_summary`, `v_planner_scorecard`, `v_contacts`, `v_contact_summary`, `v_encounter_summary`, and `v_series_*`). Guarded read-only SQL (`PolarsSQLEngine`), registered computations (`analytics/computations.py`), and the content-addressed `EvidenceStore` persist query/computation results under `analysis/evidence/<evidence_id>.parquet` indexed at `analysis/evidence_index.parquet`.

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
|   |   |-- metrics/           <- Metric calculators (layer 2, returns with PR2)
|   |   |-- acoustics/         <- Acoustic propagation solver and door geometry (layer 2)
|   |   |-- path/              <- Reference path solver (layer 2)
|   |   `-- pose_segments.py   <- Teleport detection (layer 2)
|   |-- analytics/             <- Layer 3: BenchmarkCatalog, canonical views, guarded SQL, query store
|   |-- presentation/          <- Layer 5: Report and plot generation
|   |   |-- report_builder.py  <- Generates report.html and plots
|   |   |-- manifest_registry.py <- Resolves report manifests
|   |   |-- viz_manifest.py    <- VizManifest model and default loader
|   |   |-- plotly_renderer.py <- Interactive HTML chart dispatcher
|   |   |-- seaborn_renderer.py<- Static PNG chart dispatcher
|   |   |-- color_utils.py     <- Palette loader and styling
|   |   |-- dimension_detector.py <- Auto-detects varying dimensions for compound labels
|   |   |-- report_template.html.j2 <- Jinja2 report template
|   |   `-- plot_types/        <- violin, box, bar, histogram, scatter, trajectory, radar, heatmap, timeseries, line
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
|   |-- mcap_writer_options.yaml   <- MCAP writer tuning
|   `-- color_palette.yaml         <- Color palette
|-- configs/declarations/query_declarations/ <- Named read-only SQL query declarations
|-- configs/benchmark/
|   |-- suites/                <- Suite definitions
|   |-- contests/              <- Contest definitions
|   `-- manifests/             <- Report manifests (standard, ecological, social, safety, characterization)
`-- tests/
    |-- unit/                  <- Unit tests (no ROS required)
    |-- integration/           <- Integration tests
    `-- test_benchmark_*.py    <- Benchmark runner tests

arena_evaluation_msgs/         <- Sibling package: ROS 2 message and service definitions
|-- msg/BenchmarkState.msg     <- Live benchmark progress
`-- srv/                       <- RecordEpisode, ChangeDirectory services
```

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

### 3. Migrate a run recorded before the native layout

```bash
arena evaluation migrate <run dir>
```

See the Migration section of [ingestion/README.md](arena_evaluation/ingestion/README.md). The metric reports (`process`, `run`, `report`, `plot`) return with PR2 and PR4.

### Report manifests (returns with PR4)

Report layouts are declarative YAML files in `configs/benchmark/manifests/`. Built-in manifests: `standard`, `ecological`, `social`, `safety`, `characterization`.

```bash
arena evaluation run --list-manifests
arena evaluation run --benchmark-dir X --report-manifest ecological
arena evaluation run --benchmark-dir X --report-manifest '{name: inline, plots: [...]}'
```

---

## CLI Reference

```
usage: arena evaluation <command> [--run-dir DIR | --benchmark-dir DIR] [--output-dir DIR]
                            [--workers N] [--report-manifest NAME|PATH|{...}] [--list-manifests]

Commands:
  extract   Extract topics from MCAP into Parquet files and write the benchmark's dimensions
  migrate   Convert a run recorded before the native layout to the current layout, in place
  process   Compute metrics and write metrics.parquet (returns with PR2)
  run       Full pipeline: extract -> process -> report + plots (returns with PR2 and PR4)
  report    Generate report.html from existing metrics.parquet (returns with PR4)
  plot      Generate static PNG plots only (returns with PR4)
  acoustic  Acoustic analysis subcommands (list, animate, snapshot)
```

| Flag | Description |
|---|---|
| `--benchmark-dir DIR...` | Benchmark root directories, every episode under them |
| `--run-dir DIR...` | Single `episode_NNN` directories: `process` and `extract` handle each alone, `report` and `plot` merge their `metrics.parquet` |
| `--output-dir DIR` | Output directory for reports and plots (defaults to first input dir) |
| `--workers N` | Worker count for parallel processing (`-1` = auto CPU count) |
| `--force-extract` | Force re-extraction of MCAP files |
| `--report-manifest NAME\|PATH\|{...}` | Report layout: named manifest, YAML path, or inline YAML |
| `--list-manifests` | List available named report manifests |

`process` and `run` exit 1 when a benchmark or episode directory produced no metrics.

---

## MCP domain

The package contributes the `evaluation` domain to the Arena MCP server (`arena_evaluation/mcp/`, registered under the `arena_mcp.domains` entry point). `python3 -m arena_mcp ls evaluation` lists its tools with one line each.

- Catalogs come from the live registries and resolvers: task modes and their parameter schemas from the task-mode registry, metrics and their declared directions from the metric registry, drivers from the bringup and adapter registries, nav2 planner configs from `arena_robots`, bridge planners from `arena_planners`, worlds, suites, contests and manifests through their `Identifier` resolvers (with `network=true` the buckets too).
- Validation runs the runner's own parsers (`Suite.parse`, `Contest.parse` plus the runner's nav2 preflight, `VizManifest`, `parse_scenario`). Created suites, contests and manifests go to the write target of their resolver chain, a manifest for one run into that run directory, scenarios into the world the map name resolves to.
- Benchmarks (`arena evaluation benchmark`), preflights and soaks (`arena planners test`), processing, extraction, reports, plots and acoustic renders run as jobs of the real CLI verbs. A benchmark always gets an explicit run id, and the start waits until the runner prepared its run directory, so a config error comes back as an error. Poll jobs with `jobs_status` and `jobs_log`, stop a benchmark with `evaluation_stop`, which interrupts its runner process group and escalates only after a grace window.
- Run state is read from `$ARENA_DATA_DIR/benchmarks/<run_id>` with the package's readers: the manifest's steps, the union of every lane's state file and the folded progress files, so shared runs show their lanes while they run. Analysis tools group by contestant (the `planner` column) and leave reference episodes out unless asked, and they rank only over outputs whose calculator declares `OUTPUT_DIRECTIONS`.
- Recording control calls an env's recorder (`<task generator>/start_episode`) and reads `<task generator>/state/recorded_topics`, env-scoped tools take `env_id`.

Tests live in `tests/mcp/` and drive a real `python3 -m arena_mcp serve` over stdio against run directories built with the package's writers.

---

## Running Tests

Unit tests are pure Python and do not require ROS.

```bash
pytest tests/unit -v
pytest tests/ -v
```

---

## Sub-package Documentation

- [ingestion/README.md](arena_evaluation/ingestion/README.md): recording, topics, episode lifecycle, migration
- [processing/README.md](arena_evaluation/processing/README.md): extraction, native tables, pose stream, dimension rows
- [processing/metrics/README.md](arena_evaluation/processing/metrics/README.md): metric calculator framework
- [presentation/README.md](arena_evaluation/presentation/README.md): declarative manifests, plot types
- [storage/README.md](arena_evaluation/storage/README.md): registry, dimensions and contracts, paths, metadata
- [benchmark/README.md](arena_evaluation/benchmark/README.md): runner, state, CLI, profiler
- [configs/benchmark/README.md](configs/benchmark/README.md): suites, contests, characterization
- [arena_evaluation_msgs/README.md](../arena_evaluation_msgs/README.md): ROS 2 message and service definitions

