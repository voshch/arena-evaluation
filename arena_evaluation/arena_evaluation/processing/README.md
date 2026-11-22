# processing (Layer 1: extraction and dimensions)

Reads each recorded episode's MCAP once into native tables, then writes the benchmark's dimension tables from them. The metric calculators in [metrics/](metrics/README.md) are layer 2.

```
episode_NNN.mcap
  -> MCAPReader        flat tables, one per topic family
  -> topics/           the native tables of the episode, one Parquet file each
  -> dimension_rows    the dimension rows of the episode
  -> benchmark root    dim_*.parquet, stacked over the episodes, checked, written once
```

Runs recorded before the native layout are converted once by `arena evaluation migrate`; see [ingestion/README.md](../ingestion/README.md).

---

## Files

| File | Purpose |
|---|---|
| `mcap_reader.py` | `MCAPReader` streams one recording into flat tables; `load_bundles` reads the episode's tables back as one `TopicBundle` per robot |
| `telemetry/` | Shapes recorded messages into flat rows: `tracks`, `sensors`, `peds`, `events`, `context`, `messages`, and `writer`, which buffers the rows per table |
| `parquet_store.py` | `TopicParquetStore`: the extraction cache of an episode's `topics/` folder |
| `pipeline.py` | `ProcessingPipeline`: extracts an episode, a run directory or a benchmark, and writes the benchmark's dimensions |
| `dimension_rows.py` | `episode_dimensions`: the dimension rows one episode contributes |
| `world_geometry.py` | `benchmark_world_geometry`, `world_geometry_rows`: the walls, doors, elevator polygons, zones and object footprints of the benchmark's worlds, one row per vertex |
| `pose_anchor.py` | Classifies the ground-truth pose stream as dense, sparse or absent |
| `map_registry.py` | `MapRegistry`: the map image of a world, resolved the way the runtime does, and cached |
| `metrics/` | Layer 2: metric calculators, see [metrics/README.md](metrics/README.md) |
| `acoustics/`, `path/` | Layer 2: the acoustic propagation solver with its door geometry, and the reference path solver |
| `pose_segments.py` | Layer 2: teleport detection on a recorded pose track |

---

## Extraction

- `extract_benchmark` discovers the episodes of a benchmark, extracts each one (in worker processes when `workers` is above 1), then writes the dimensions once at the benchmark root.
- `extract_episode` skips an episode whose MCAP is missing, with a warning. A current `topics/` cache is reused unless `force_extract` is set; otherwise `topics/` is rebuilt from the MCAP and marked complete.
- `extract_run_dir` extracts one `episode_NNN` folder and writes no dimensions.

## Dimensions

The benchmark step reads each episode's metadata YAML and its `topics/` bundles, builds the episode dimension rows, stacks them, adds the metric dictionary and the geometry of every named world, checks the result against the registry contract (`storage/contracts.py`) and writes each table once (`storage/star_schema.py`). An episode without a `topics/` folder is skipped with a warning, so the episode dimensions have no rows for it; its world still counts for the geometry.

| Table | One row per | Key |
|---|---|---|
| `dim_episodes` | episode and robot | `run_id`, `episode_id`, `robot_id` |
| `dim_robots` | robot of an environment | `run_id`, `episode_id`, `env_id`, `robot_id` |
| `dim_pedestrians` | pedestrian of an environment | `run_id`, `episode_id`, `env_id`, `ped_id` |
| `dim_world_entities` | world entity of an environment | `run_id`, `episode_id`, `env_id`, `entity_name` |
| `dim_world_geometry` | vertex of a world's walls, doors, elevator polygons, zones or object footprints | `world`, `element_kind`, `element_name`, `part_idx`, `vertex_idx` |
| `dim_metrics` | metric and scope | `metric`, `scope` |

Pedestrians and world entities are read once per environment, not once per robot. Every NULL carries its reason in `null_reason`.

`dim_world_geometry` is the world model flattened to one row per vertex: `element_kind` is `wall`, `door`, `elevator`, `zone` or `object`; `element_name` is `wall_<index>` for a wall and the entity name otherwise; `part_idx` is the polygon index within the element (0 for a wall or a zone); `vertex_idx` is the vertex order along the segment or open ring; `x` and `y` are metres in the compacted map frame that `path/static_environment.py` rasterises. The pipeline writes it once per named world of the benchmark; a world that does not resolve is skipped with a warning and has no rows, and a plot then draws without it.

`dim_metrics` is the metric dictionary: the fact table, dtype, unit, description, direction and producing calculator of every metric at every scope it is stored at. The pipeline adds it from the metric declarations (`metrics/dictionary.py`), not from the episodes, and it holds no metric value.

## Telemetry

The native tables are declared once in `storage/registry/telemetry.py`; this folder fills them.

- robot: `robot_tracks`, `robot_odom`, `robot_cmd_vel`, `robot_scan_summary`, `robot_joint_states`, `robot_power`, `robot_collision_events`, `robot_plans`, `robot_descriptor`, `robot_params`, and the other `robot_*` tables
- pedestrian: `pedestrian_arena_peds`, `pedestrian_agent_states`, `pedestrian_gestures`, `pedestrian_animations`, `pedestrian_interactions`, `pedestrian_interaction_events`
- episode and world: `episode_lifecycle`, `episode_integrity`, `world_semantics`
- transforms: `tf`, `tf_static`

Positions in the shared `map` frame are moved into the environment's frame by the `map -> env_<n>/map` transform in effect at each row's time. A row before the first transform uses the first one, so a world swap during a recording is handled row by row. A recorded message whose bytes are corrupt, or whose schema cannot be decoded, is skipped; an undecodable schema logs one warning per topic.

## Pose stream

- `pose_anchor.classify` calls a ground-truth track dense when the median gap between its samples is at most 0.2 s, sparse when the gap is longer, and absent when it is empty. `resolve_pose_source` names the stream `tf` (dense) or `tf_sparse` (sparse). Odometry is never a pose source.
