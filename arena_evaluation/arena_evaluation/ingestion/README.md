# ingestion (Layer 1: Live Data Recording)

ROS 2 node that records live simulation runs into structured MCAP files.

---

## Responsibilities

- Subscribe to ROS 2 topics during simulation.
- Write one continuous MCAP file per step or episode.
- Embed `EpisodeRecord` messages in the MCAP stream for episode boundaries.
- Write and update `metadata.yaml` with run-level context.
- Flush and close MCAP cleanly on shutdown.

---

## Files

| File | Purpose |
|---|---|
| `recorder.py` | `DataRecorderNode` recording node |
| `episode_writer.py` | `EpisodeWriter`: episode directory, MCAP writer, flatten and metadata finalize, shared by the recorder node and offline importers (`arena asset import episode musohu <bag>` writes MuSoHu rosbag1 runs as reference episodes of the robot model `human`: scan, odom with the `/action` twist and cmd_vel on the recorder's topics, every other topic with a ROS 2 type under the robot namespace and tf on `/tf` and `/tf_static`, frames prefixed `env_0/human/`) |
| `metadata.py` | `IngestionMetadata` helper building the `RunMetadata` of an episode |
| `migrate.py` | `migrate_run`: converts a run recorded before the native layout to the current metadata and tables (see Migration) |
| `topics.py` | Topic name constants and type-string mapping |

---

## recorder.py - DataRecorderNode

### Launch

Registered as the `record` entry point in `setup.py`. Spawned automatically by the env launch file when `record.dir` is provided.

```bash
# Manual launch
arena launch sim:=gazebo task.robots:=random record.dir:=data \
    world:=map_empty robot:=jackal

# Direct node launch
ros2 run arena_evaluation record \
    --ros-args -p record_data_dir:=/opt/arena_ws/data/recordings/20260528-210000
```

### Episode Lifecycle

An episode opens on the first `RUNNING` `EpisodeRecord` of an episode id or on a START call of the
`start_episode` service, whichever arrives first, so a recording begins when the reset is over.
`QUEUED` records, which the task generator publishes during the reset, open nothing. The recorder
subscribes to `state/episode` volatile, so the records the task generator keeps for late joiners,
episodes that are already over, never reach it. A STOP call
closes the episode and moves its MCAP to `episode_XXX/episode_XXX.mcap`.

Every `EpisodeRecord` of the current episode rewrites `integrity` in `episode_XXX.yaml`, the
list of reasons the episode is compromised, empty when it is intact. Reasons accrue during the
episode, so the yaml holds the latest list.

The same rewrite stores `task_layer`, the name of the task runtime from `EpisodeRecord.task_layer` (empty in recordings made before it, which mean the default runtime and which scenario provenance looks up namespaces in), `modes`, the selection per axis decoded from `EpisodeRecord.modes` (axis to list of mode names), and `params`, the effective params of the selected modes under their full names. Episode files written before `modes` (with `tm_robots`, `tm_obstacles`, `tm_modules`, `robots_params` and `obstacles_params`) are not read by the recorder; the Migration section maps them once.

A bag directory `episode_XXX/episode_XXX/` that an earlier recorder left unclosed is kept as
`episode_XXX/episode_XXX_unclosed_N/` and the episode is recorded as usual.

### Output Path Resolution

The `record_data_dir` parameter resolution order:

1. ROS parameter `record_data_dir`.
2. CLI argument `--dir` / `-d`.
3. Default `auto:/` (generates timestamped folder).

**Run structure:**
```
<benchmark root>/
|-- dim_episodes.parquet          benchmark dimension, one row per episode and robot
|-- dim_robots.parquet
|-- dim_pedestrians.parquet
|-- dim_world_entities.parquet
|-- dim_world_geometry.parquet
|-- dim_metrics.parquet           the metric dictionary
|-- worlds/                       world snapshots, one folder per world
`-- episodes/
    `-- episode_NNN/
        |-- episode_NNN.mcap      the recording
        |-- episode_NNN.yaml      RunMetadata of the episode
        |-- topics/               one Parquet file per telemetry table
        |   |-- robot_tracks.parquet
        |   |-- robot_odom.parquet
        |   `-- tf.parquet
        `-- metrics/              one Parquet file per fact and series table
            |-- fact_episodes.parquet
            |-- fact_robots.parquet
            `-- series_path_metrics.parquet
```

### World snapshot

On the first episode of each world, the recorder writes that world as it runs
to `<run_dir>/worlds/<world name>/`: `world.yaml` (format 2, linked levels
written out, ceilings under the `world.lighting` mode) and `kg.yaml`
(stated and derived facts under `kg.derive`), and for a multi-level world
`level_origins.yaml`, the per-level shifts the runtime flattened the full
world with, which a level-filtered snapshot cannot derive from its own levels.
Env offsets and other runtime state stay out. Evaluation reads world geometry from there before the
package worlds. The recorder copies no scenario directories: the edge-case provenance that reads them moves with the metric layer.

### Recorded Topics

All topics use simulation time from `/clock`. Messages that arrive before the first `/clock` tick are held and
written at that tick. When an episode opens first, which happens while a paused simulator publishes no clock,
the held messages of non-latched topics predate it and are dropped.

| Topic | Message Type | Throttle |
|---|---|---|
| `/{ns}/cmd_vel` | `geometry_msgs/Twist` | 20 Hz |
| `/{ns}/scan`, `/{ns}/lidar` | `sensor_msgs/LaserScan` | 10 Hz |
| `/{ns}/{robot}_velocity_controller/odom` | `nav_msgs/Odometry` | 50 Hz |
| `/{ns}/joint_states` | `sensor_msgs/JointState` | 20 Hz |
| `/{ns}/plan` | `nav_msgs/Path` | unthrottled |
| `/{ns}/collision_events` | `arena_robots_msgs/CollisionEvents` | unthrottled |
| `/{parent_ns}/goal_pose` | `geometry_msgs/PoseStamped` | unthrottled |
| `/{parent_ns}/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | unthrottled, one env subscription |
| `/{parent_ns}/arena_peds` | `arena_people_msgs/Pedestrians` | 20 Hz |
| `/{parent_ns}/interactions` | `arena_humansim_msgs/InteractionFrame` | latched |
| `/{parent_ns}/state/episode` | `arena_runtime_msgs/EpisodeRecord` | unthrottled |
| `/{parent_ns}/state/robots` | `arena_runtime_msgs/RobotFleet` | latched |
| `/tf` | `tf2_msgs/TFMessage` | header-stamped, 20 Hz per frame pair |
| `/tf_static` | `tf2_msgs/TFMessage` | latched |
| `/clock` | `rosgraph_msgs/Clock` | time tracking |

Opt-in topics (`recorded: false`), recorded only when listed under `optional_topics` in `config/data_recorder_config.yaml`:

| Key | Topic | Message Type | Throttle |
|---|---|---|---|
| `planner_camera` | `/{ns}/vln_camera/image` | `sensor_msgs/Image` | 10 Hz |

`planner_camera` is the egocentric camera the VLN and frontier planners see. Raw 640x480 RGB at 10 Hz adds about
9 MB per second of episode to the bag. `arena evaluation video --benchmark-dir <run>` turns it into one
`episodes/<id>/video/<topic>.mp4` per robot and episode, in recorded time (the last image is held over gaps).

`initialpose` is subscribed once at the environment: the row is written under
`/{parent_ns}/{robot}/initialpose` when the header frame names a robot (`env_N/<robot>/<frame>`),
and stays at `/{parent_ns}/initialpose` when it names none.

`/{parent_ns}/interactions` is humansim's latched `InteractionFrame`: the live interactions (type, outcome, anchor object,
members with their participant or queued role, arrived, holding, hold duration) plus the ACTIVATED / HOLD_ONSET /
RELEASED / INTERRUPTED / CANCELED edges fired that tick, published only when its content changes or an edge fires.

Animation state is picked up by discovery (by message type and topic basename, anywhere under
the env namespace, i.e. one level above the recorder) rather than from the fixed list above:

| Topic | Message Type | Throttle |
|---|---|---|
| `/{env_ns}/animation_states` | `arena_people_msgs/AnimationStates` | 20 Hz (per-ped clip, playhead, gesture phase, blend weight) |
| `/{env_ns}/arena_peds`, `/{parent_ns}/agent_states`, `/{parent_ns}/interactions` | fallback if the fixed paths above miss them | 20 Hz, interactions latched |

The reader writes them to the `pedestrian_*` tables (interactions, interaction events, animations and gestures),
and keeps `arena_peds` (rendered, env frame) and `agent_states` (crowd-engine physics, engine frame) apart as
`pedestrian_arena_peds` and `pedestrian_agent_states`.

Simulator backends declare further topics on the latched `/{parent_ns}/state/recorded_topics` (`arena_runtime_msgs/RecordedTopics`), the acoustics backend its audio streams and heard-sound events among them. Each row names a topic template (`{tg}` is the task generator node, `{ns}` the robot namespace), a `pkg/msg/Type` string resolved by name, the throttle and default-recording flags, and its subscription QoS (reliable or best effort, transient-local or volatile, a depth where 0 keeps the recorder default of 10, or 100 when latched). Robot-scoped rows are subscribed per robot as robots join, rows published later are picked up when the latched message is republished, and a row whose type does not import is skipped with one warning. Rows with `recorded: false` are recorded only when listed under `optional_topics` in `config/data_recorder_config.yaml`.

`/tf` is written per transform at its `header.stamp`, one `TFMessage` per stamp, so a flush interval never collapses two stamps into one row. `tf_frames` in `config/data_recorder_config.yaml` is an ordered `[regex, ms]` list matched against the child frame that caps frame families; every other `(frame_id, child_frame_id)` pair is capped at the `/tf` rate of `record_rates_hz` (20 Hz by default). By default skeleton bones go out at 5 Hz.

### Shutdown

`SIGTERM` and `SIGINT` trigger `finalize()`:

1. Sets `is_shutting_down = True`.
2. Writes final `metadata.yaml`.
3. Calls `writer.close()` to flush rosbag2 buffers and write the MCAP index.

---

## metadata.py - IngestionMetadata

Builds the `RunMetadata` of one episode, written to `episode_XXX.yaml`. `modes` and `params` stay NULL until the episode's `EpisodeRecord` arrives.

```python
metadata = IngestionMetadata.create_episode_metadata(
    benchmark_id="my_benchmark",
    planner="dwa",
    stage="stage_1",
    map_name="hospital",
    episode_id=0,
    robot_model="jackal",
    suite_name="default",
    contest_name="dwa_vs_teb",
)
```

---

## topics.py - Topic Definitions

Provides topic names and expected ROS message types for rosbag2 registration.

```python
from arena_evaluation.ingestion.topics import get_topics

topics = get_topics(namespace="arena/env_0/arena_env_node/jackal")
```

`recorded_topic_definition(row, tg_namespace, robot_namespace)` turns one `RecordedTopic` row into a `TopicDefinition`, or `None` when its message type does not import.

---

## Migration: `arena evaluation migrate`

`arena evaluation migrate <benchmark dir>` converts a benchmark recorded before the native layout, in place. It recognises a benchmark by its `episodes/episode_*` layout and never mistakes `manifest.yaml` or `report_manifest.yaml` for episode metadata; a single episode folder still works. The command is idempotent: an episode whose metadata is migrated and whose `topics/` cache is current is left alone.

Supported generations, detected from the MCAP channel and schema records:

| Generation | EpisodeRecord | AgentMeta | Fleet |
|---|---|---|---|
| `native` | `arena_runtime_msgs` | carries the five `robot_*` settings | `RobotFleet` |
| `legacy_2026_09` | `task_generator_msgs` (`tm_*` fields) | native or absent | `task_generator_msgs/RobotState[]` |
| `legacy_2026_11` | `arena_runtime_msgs` | no `robot_*` fields | `RobotFleet` or absent |
| `legacy_2026_11_11` | `arena_runtime_msgs` | `robot_awareness`, `robot_fov_gain` and the two gains | `RobotFleet` or absent |

A present but unrecognised episode or fleet schema is `unknown` and is refused with its name. Genuinely missing constructs are reported, never invented: an absent `task_layer`, `modes` or `params` on the old EpisodeRecord is NULL (`not_recorded`); an old AgentMeta leaves `robot_range_offset`, `robot_fov_offset`, `robot_repulsion_gain`, `robot_reaction_gain` and `robot_blind` NULL, and a `legacy_2026_11_11` one carries only the two gains; a recording with no `/state/robots` fleet gets `robot_descriptor`/`robot_params` synthesised once from its recorded `RunMetadata.robot_model` and the installed `arena_robots` definition, and a missing model or definition is a hinted `MigrationError`, never a default radius or mass.

- Metadata: the `tm_robots`, `tm_obstacles` and `tm_modules` fields become the `robots`, `obstacles` and `modules` axes of `modes`; the nested `robots_params` and `obstacles_params` become `params` under `task.<mode>.<leaf>`. An episode with neither `modes` nor the `tm_*` fields is refused, pointing at the migration.
- Environment: taken from the topic names, then from the transform frames. A recording that names no environment, or several, is refused.
- Telemetry: the MCAP is read again into the native tables under `topics/`, and the episode YAML is rewritten.
- Old processed outputs: `metrics.parquet`, `combined_metrics.parquet` and incomplete per-episode fact files move to `legacy/` (never deleted); a fresh `process` recomputes the native tree.
- Report: every run writes `migration_report.yaml` with the generation, mapped fields, fallbacks, unsupported constructs and error of each episode, plus the retired files.
- Failure: an episode that cannot be mapped is recorded as failed and the remaining episodes still migrate. The command exits non-zero with the list of failed episodes; a failed episode keeps its topics and YAML as they were.
- Flags (exposed by the CLI): `--dry-run` detects and reports without writing, `--force` re-extracts even when the cache is current, `--report <path>` writes the report elsewhere.
- Limits: one MCAP per episode folder (continuous recordings split by an EpisodeRecord are out of scope); no back-fill of missing measurements; old processed values are quarantined and recomputed, never imported; `progress.csv` is runner state and is not migrated.
- `legacy_2026_09` records that predate `human_params`/`conditions`/`phases` are migrated with those fields NULL (`not_recorded`); a fleet of any generation without `robot_radius`/`robot_mass_kg` gets them backfilled from the installed `arena_robots` definition. After a forced `process --force-extract`, re-run `evaluation migrate --force` before processing (the re-extraction rewrites the backfilled parameter rows).

The full generation table, NULL mapping, descriptor synthesis rule, `migration_report.yaml` schema and
limits are in [docs/migrate.md](../../docs/migrate.md).

---

## Throttle Configuration

Configured in Hz in `config/data_recorder_config.yaml`:

```yaml
record_rates_hz:
  /tf: 20.0                # default per-pair transform interval
  /env_0/jackal/scan: 5.0  # exact topic name override
```

`record_rates_hz` maps the exact topic name the recorder subscribes to a rate in Hz; a name that is not an exact key keeps the rate of its `TopicDefinition` (`scan`/`lidar` 10 Hz, `odom` 50 Hz, `joint_states`/`peds`/`tf` 20 Hz, `power`/`energy`/`acoustics` 10 Hz, default 20 Hz). A rate of `0` and a topic whose `TopicDefinition` is `throttled=False` are unthrottled, as are the latched and event-driven topics. The former millisecond `record_frequencies` key with substring keys is refused by the loader with an error naming `record_rates_hz`.

