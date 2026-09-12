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

### Recorded Topics

All topics use simulation time from `/clock`. Messages prior to the first `/clock` tick are dropped.

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
| `/{parent_ns}/state/episode` | `task_generator_msgs/EpisodeRecord` | unthrottled |
| `/{parent_ns}/state/robots` | `task_generator_msgs/RobotFleet` | latched |
| `/tf` | `tf2_msgs/TFMessage` | 20 ms, merged |
| `/tf_static` | `tf2_msgs/TFMessage` | latched |
| `/clock` | `rosgraph_msgs/Clock` | time tracking |

Interaction and animation state is picked up by discovery (by message type and topic basename, anywhere under
the env namespace, i.e. one level above the recorder) rather than from the fixed list above:

| Topic | Message Type | Throttle |
|---|---|---|
| `/{parent_ns}/interactions` | `arena_humansim_msgs/Interactions` | unthrottled, reliable (carries edge-triggered ACTIVATED / HOLD_ONSET / RELEASED events) |
| `/{env_ns}/animation_states` | `arena_people_msgs/AnimationStates` | 50 ms (per-ped clip, playhead, gesture phase, blend weight) |
| `/{env_ns}/arena_peds`, `/{parent_ns}/agent_states` | fallback if the fixed paths above miss them | 20 ms |

The reader turns them into `interactions`, `interaction_events`, `animation_states` and `ped_gestures` env tables,
and keeps `arena_peds` (rendered, env frame) and `agent_states` (crowd-engine physics, engine frame) apart as
`peds` and `peds_physics`.

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

---

## Throttle Configuration

Configured in `config/data_recorder_config.yaml`:

```yaml
record_frequencies:
  default: 20.0   # ms fallback
  lidar:  100.0   # ms (10 Hz)
  odom:    20.0   # ms (50 Hz)
```

