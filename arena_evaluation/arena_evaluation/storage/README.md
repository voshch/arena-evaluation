# storage (Layer 1: recording and storage)

The stored layout of one benchmark: the registry that declares every stored table once, the metadata and bundle types, the dimension tables written at the benchmark root, and the paths, atomic writes and contract checks around them.

---

## Files

| File | Purpose |
|---|---|
| `registry/` | One `Table` class per stored table: columns, units, row keys, null reasons, the Polars and Arrow schemas and the file path. `table.py` the `Table` base class and the metric declarations, `telemetry.py` the recorded tables, `dimensions.py` the six dimension tables, the metric dictionary among them, `facts.py` the identity and recorded columns of the fact tables (the metric layer declares the metric columns), `scopes.py` the scopes and `null_reasons.py` the reasons of a NULL |
| `star_schema.py` | Dimension tables: `conform` a frame to its registry schema, `stack` the episodes, `write_benchmark_dimensions` once at the benchmark root |
| `contracts.py` | The dimension contract: registry schema, unique non-NULL keys, and a reason for every NULL. A violation raises `ContractViolationError` |
| `schemas.py` | `RunMetadata` (the metadata of one episode), `EpisodeDescriptor`, `RobotParams`, and `TopicBundle` (the recorded tables of one robot) |
| `manifest.py` | `MetadataWriter`: reads and writes the episode YAML |
| `folder_manager.py` | Paths of benchmarks, episodes and MCAP files under the data root; discovers the episodes of a benchmark |
| `data_root.py` | The data root (`ARENA_DATA_DIR`, or the installed package data) and the latest benchmark |
| `atomic.py` | Atomic replacement of files and the advisory lock of a read-modify-write |
| `planner_names.py` | Splits a planner name into its local and inter planner |
| `exceptions.py` | The package's exception types |

## Layout of a benchmark

```
<data root>/<benchmark_id>/
|-- dim_episodes.parquet          one row per episode and robot
|-- dim_robots.parquet
|-- dim_pedestrians.parquet
|-- dim_world_entities.parquet
|-- dim_world_geometry.parquet    one row per vertex of a world's geometry
|-- dim_metrics.parquet           one row per metric and scope: the metric dictionary
`-- episodes/episode_NNN/
    |-- episode_NNN.mcap          the recording
    |-- episode_NNN.yaml          RunMetadata of the episode
    |-- topics/                   the native tables, one Parquet file each
    `-- metrics/                  the fact tables and the series tables, one Parquet file each
```

## Rules

- A stored table is declared once, in `registry/`, and imported wherever it is used. No other module spells its column names. The series tables (`registry/series.py`) are the per-sample rows one robot's episode publishes, one `metrics/series_<stem>.parquet` each, keyed by run, episode, robot and native time; the writer conforms each frame to its declaration and refuses a frame that drifts from it.
- A missing value is NULL with a reason from `NullReason`. It is never a zero and never an invented default.
- The dimension tables are written once per benchmark: the episode tables from each episode's metadata and recorded telemetry, `dim_world_geometry` from the compacted level of each named world, and the metric dictionary from the metric declarations. No metric value feeds them.
- A missing required input is an error, never a zero or a silent default.
- Storage imports nothing from `processing`, `presentation`, `analytics`, `metrics`, `benchmark` or `mcp`.
