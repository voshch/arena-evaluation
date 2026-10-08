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
| `metadata.py` | `IngestionMetadata` helper building initial `RunMetadata` |
| `topics.py` | Topic name constants and type-string mapping |

---

## recorder.py - DataRecorderNode

### Launch

Registered as the `record` entry point in `setup.py`. Spawned automatically by `task_generator` when `record.dir` is provided.

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

A bag directory `episode_XXX/episode_XXX/` that an earlier recorder left unclosed is kept as
`episode_XXX/episode_XXX_unclosed_N/` and the episode is recorded as usual.

### Output Path Resolution

The `record_data_dir` parameter resolution order:

1. ROS parameter `record_data_dir`.
2. CLI argument `--dir` / `-d`.
3. Default `auto:/` (generates timestamped folder).

**Benchmark run structure:**
```
data/<benchmark_id>/recordings/<planner>/<stage>/
|-- metadata.yaml
|-- params.yaml
`-- recording/
    `-- recording_0.mcap
```

**Ad-hoc run structure:**
```
data/recordings/<YYYYMMDD-HHMMSS>/
|-- metadata.yaml
|-- params.yaml
`-- recording/
    `-- recording_0.mcap
```

### World snapshot

On the first episode of each world, the recorder writes that world as it runs
to `<run_dir>/worlds/<world name>/`: `world.yaml` (format 2, linked levels
written out, ceilings under the `world.lighting` mode) and `kg.yaml`
(stated and derived facts under `kg.derive`), and for a multi-level world
`level_origins.yaml`, the per-level shifts the runtime flattened the full
world with, which a level-filtered snapshot cannot derive from its own levels.
Env offsets and other runtime state stay out. Evaluation reads world geometry from there before the
package worlds. On the first episode of each scenario a `scenario` task mode
runs, the recorder also copies that scenario's directory (`scenario.yaml`,
`provenance.yaml`, scenario-local agent types) to
`<run_dir>/worlds/<world name>/scenarios/<file>/`, which the edge-case metrics
read before the world's own scenario.

### Recorded Topics

All topics use simulation time from `/clock`. Messages that arrive before the first `/clock` tick are held and
written at that tick. When an episode opens first, which happens while a paused simulator publishes no clock,
the held messages of non-latched topics predate it and are dropped.

| Topic | Message Type | Throttle |
|---|---|---|
| `/{ns}/cmd_vel` | `geometry_msgs/Twist` | 20 ms |
| `/{ns}/lidar` | `sensor_msgs/LaserScan` | 100 ms |
| `/{ns}/{robot}_velocity_controller/odom` | `nav_msgs/Odometry` | 20 ms |
| `/{ns}/joint_states` | `sensor_msgs/JointState` | 20 ms |
| `/{ns}/plan` | `nav_msgs/Path` | unthrottled |
| `/{ns}/collision_events` | `arena_robots_msgs/CollisionEvents` | unthrottled |
| `/{parent_ns}/goal_pose` | `geometry_msgs/PoseStamped` | unthrottled |
| `/{parent_ns}/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | unthrottled |
| `/{parent_ns}/arena_peds` | `arena_people_msgs/Pedestrians` | 20 ms |
| `/{parent_ns}/interactions` | `arena_humansim_msgs/InteractionFrame` | latched |
| `/{parent_ns}/state/episode` | `task_generator_msgs/EpisodeRecord` | unthrottled |
| `/{parent_ns}/state/robots` | `task_generator_msgs/RobotFleet` | latched |
| `/tf` | `tf2_msgs/TFMessage` | 20 ms, merged |
| `/tf_static` | `tf2_msgs/TFMessage` | latched |
| `/clock` | `rosgraph_msgs/Clock` | time tracking |

`/{parent_ns}/interactions` is humansim's latched `InteractionFrame`: the live interactions (type, outcome, anchor object,
members with their participant or queued role, arrived, holding, hold duration) plus the ACTIVATED / HOLD_ONSET /
RELEASED / INTERRUPTED / CANCELED edges fired that tick, published only when its content changes or an edge fires.

Animation state is picked up by discovery (by message type and topic basename, anywhere under
the env namespace, i.e. one level above the recorder) rather than from the fixed list above:

| Topic | Message Type | Throttle |
|---|---|---|
| `/{env_ns}/animation_states` | `arena_people_msgs/AnimationStates` | 50 ms (per-ped clip, playhead, gesture phase, blend weight) |
| `/{env_ns}/arena_peds`, `/{parent_ns}/agent_states`, `/{parent_ns}/interactions` | fallback if the fixed paths above miss them | 20 ms, interactions latched |

The reader turns them into `interactions`, `interaction_events`, `animation_states` and `ped_gestures` env tables,
and keeps `arena_peds` (rendered, env frame) and `agent_states` (crowd-engine physics, engine frame) apart as
`peds` and `peds_physics`.

Simulator backends declare further topics on the latched `/{parent_ns}/state/recorded_topics` (`task_generator_msgs/RecordedTopics`), the acoustics backend its audio streams and heard-sound events among them. Each row names a topic template (`{tg}` is the task generator node, `{ns}` the robot namespace), a `pkg/msg/Type` string resolved by name, the throttle and default-recording flags, and its subscription QoS (reliable or best effort, transient-local or volatile, a depth where 0 keeps the recorder default of 10, or 100 when latched). Robot-scoped rows are subscribed per robot as robots join, rows published later are picked up when the latched message is republished, and a row whose type does not import is skipped with one warning. Rows with `recorded: false` are recorded only when listed under `optional_topics` in `config/data_recorder_config.yaml`.

`/tf` is merged rather than throttled: the newest transform per `(frame_id, child_frame_id)` is kept and written as one `TFMessage` per 20 ms window, so no publisher can starve the others. `tf_frames` in `config/data_recorder_config.yaml` is an ordered `[regex, ms]` list matched against the child frame that caps frame families; by default skeleton bones go out at 5 Hz.

### Shutdown

`SIGTERM` and `SIGINT` trigger `finalize()`:

1. Sets `is_shutting_down = True`.
2. Writes final `metadata.yaml`.
3. Calls `writer.close()` to flush rosbag2 buffers and write the MCAP index.

---

## metadata.py - IngestionMetadata

Builds the initial `RunMetadata` written to `metadata.yaml` on node startup.

```python
metadata = IngestionMetadata.create_initial_metadata(
    benchmark_id="my_benchmark",
    planner="dwa",
    stage="stage_1",
    episodes_requested=10,
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

topics = get_topics(namespace="arena/env_0/task_generator_node/jackal")
```

`recorded_topic_definition(row, tg_namespace, robot_namespace)` turns one `RecordedTopic` row into a `TopicDefinition`, or `None` when its message type does not import.

---

## Throttle Configuration

Configured in `config/data_recorder_config.yaml`:

```yaml
record_frequencies:
  default: 20.0   # ms fallback
  lidar:  100.0   # ms (10 Hz)
  odom:    20.0   # ms (50 Hz)
```

