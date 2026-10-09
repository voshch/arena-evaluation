# arena_evaluation
> ROS 2 package for recording, processing, and visualising navigation and ecological evaluation metrics for Arena robots.

Evaluation pipeline converting simulation data into structured metric reports across recording, processing, and presentation layers.

---

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
|  Layer 3 - Processing  (processing/)                                |
|  MCAPReader -> TopicParquetStore -> TopicAligner -> MetricRegistry  |
|  -> ParquetStore                                                    |
|  Reads episode MCAPs offline, aligns multi-rate topics onto the     |
|  odom axis, and computes metrics.                                   |
|  Output: metrics.parquet + combined_metrics.parquet                 |
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

---

## Directory Structure

```
arena_evaluation/
|-- arena_evaluation/          <- Python package root
|   |-- ingestion/             <- Layer 1: Live recording (ROS node)
|   |   |-- recorder.py        <- DataRecorderNode (service-driven episode lifecycle)
|   |   |-- metadata.py        <- IngestionMetadata (per-episode YAML context)
|   |   `-- topics.py          <- Topic definitions and type registry
|   |-- storage/               <- Layer 2: Shared schemas and path management
|   |   |-- schemas.py         <- Data models (RunMetadata, PlotSpec, VizManifest, ...)
|   |   |-- folder_manager.py  <- Path resolution and run discovery
|   |   |-- manifest.py        <- MetadataWriter (YAML read/write)
|   |   |-- planner_names.py   <- split_planner_name helper
|   |   |-- data_root.py       <- benchmarks_root and latest_benchmark helpers
|   |   `-- exceptions.py      <- Pipeline exceptions
|   |-- processing/            <- Layer 3: Offline metric computation
|   |   |-- mcap_reader.py     <- MCAP to TopicBundle parser
|   |   |-- topic_aligner.py   <- Aligns multi-rate topics via join_asof
|   |   |-- parquet_store.py   <- Reads/writes topic cache and metrics
|   |   |-- pipeline.py        <- ProcessingPipeline orchestrator
|   |   |-- map_registry.py    <- Discovers and caches map images (PGM to PNG)
|   |   `-- metrics/           <- Metric calculators (performance, social, ecological, naturalness)
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
|   |-- cli.py                 <- Evaluation pipeline CLI
|   `-- cli_acoustic.py        <- Acoustic analysis subcommands
|-- config/
|   |-- data_recorder_config.yaml  <- Topic throttle frequencies
|   |-- mcap_writer_options.yaml   <- MCAP writer tuning
|   `-- color_palette.yaml         <- Color palette
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

### 2. Process and report (offline)

```bash
# Standard metrics report
arena evaluation run --benchmark-dir <run_id>

# Characterization report
arena evaluation run --benchmark-dir <run_id> --report-manifest characterization
```

### Report manifests

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
  extract   Extract topics from MCAP into Parquet files
  process   Compute metrics and write metrics.parquet
  run       Full pipeline: extract -> process -> report + plots
  report    Generate report.html from existing metrics.parquet
  plot      Generate static PNG plots only
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

---

## Running Tests

Unit tests are pure Python and do not require ROS.

```bash
pytest tests/unit -v
pytest tests/ -v
```

---

## Sub-package Documentation

- [ingestion/README.md](arena_evaluation/ingestion/README.md): recording, topics, episode lifecycle
- [processing/README.md](arena_evaluation/processing/README.md): extraction, alignment, metrics
- [processing/metrics/README.md](arena_evaluation/processing/metrics/README.md): metric calculator framework
- [presentation/README.md](arena_evaluation/presentation/README.md): declarative manifests, plot types
- [storage/README.md](arena_evaluation/storage/README.md): schemas, paths, metadata
- [benchmark/README.md](arena_evaluation/benchmark/README.md): runner, state, CLI, profiler
- [configs/benchmark/README.md](configs/benchmark/README.md): suites, contests, characterization
- [arena_evaluation_msgs/README.md](../arena_evaluation_msgs/README.md): ROS 2 message and service definitions

