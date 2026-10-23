# Metrics

Metric calculators for the Arena evaluation pipeline. Subclasses of `BaseMetricCalculator` are registered by `MetricRegistry` and executed in topological order based on `DEPENDS_ON`. Results are saved to `combined_metrics.parquet`, one row per episode.

## Categories

| Folder | Focus |
|---|---|
| `social/` | Robot and pedestrian interaction: social forces, proxemics, gaze, disturbance |
| `ecological/` | Energy, acoustics, world condition compliance |
| `performance/` | Path, motion, time, collision, efficiency, clearance |
| `naturalness/` | Trajectory naturalness against unobstructed baselines |
| `edge_case/` | Criticality of generated edge-case scenarios against the case's own agents, formation panel |

## Calculator Interface

Each calculator declares:

- `NAME`, `CATEGORY`, `REQUIRES_PEDSIM`, `DEPENDS_ON` (execution order edges), `REQUIRED_TOPICS` (topic gates; list or tuple entries indicate alternative acceptable topics), `UNITS`, `PRIMARY_OUTPUTS` (keys for default comparisons), `OUTPUT_DIRECTIONS` ("lower" or "higher" per output).
- `output_keys()`: List of produced metric keys.
- `calculate(episode, prior_results)`: Computes metric dictionary with all declared keys (filled with `None` on missing data or errors).

## Conventions

- **Multi-rate data:** `AlignedEpisodeBundle.topics` contains native-rate topic dataframes. Rate-sensitive calculators compute on their native time base; `episode.data` is aligned to the odom time axis for trajectory metrics.
- **Ground truth priority:** `resolve_native_pose` uses `tf_gt` ground truth pose when available, falling back to odom. Ground truth velocity is computed by differentiating ground truth positions.
- **Proxemic distance:** `*_zone` metrics calculate edge-to-edge distance `d_eff = d_center - (r_robot + r_ped)` based on Hall's proxemic zones (0.45 m, 1.2 m, 3.6 m). The legacy `proxemics` calculator uses center-to-center distance.
- **Energy units:** Energy is reported in watt-hours (Wh). `specific_cost_of_transport` uses total energy consumed.
- **VLN metrics:** `vln_metrics` scores against a geodesic distance field from the goal (the robot's last goto in `EpisodeRecord.phases`, read from its map-frame `map_poses`, so planners without a global plan are scored against their real goal), an 8-connected Dijkstra over the Theta* grid (same map, robot-radius dilation and start/goal clearing). `ne` is the any-angle Theta* length from the final pose, `osr` whether the pose nearest to the goal on the field is within the goal tolerance by the same measure, `success_geodesic` is `success` with `ne` within it, `spl_geodesic` weights that by the Theta* length (chained the same way) (Anderson et al. 2018), `ndtw` compares the robot path to the Theta* path, chained through the earlier gotos of the recorded phases, both resampled every 0.25 m (Ilharco et al. 2019), `sdtw` is `success_geodesic * ndtw`, and `stuck` flags a failed episode with a 10 s stretch moving under 0.2 m and turning under 15 degrees (VLN-PE). The success threshold is the `tolerance_radius` of that last goto, the one the task generator judged it by, so a VLN suite sets `task.episode.goto_pose.tolerance.radius` to 3.0 (VLN-CE). Geodesic outputs are `None` when the map does not resolve or the pose falls on an unreachable cell, every geodesic output is `None` for recordings without a goto phase, and every output but `ne` and `stuck` is `None` when that goto carries no tolerance. `goal_inputs` labels what the planner was given, joined by `+`: `pose`, `pose:oracle` (a pose inside a named target, which hands over the grounding), `instruction` (authored text or the named target), `instruction:route` (walking directions rendered from the map) or `instruction:coordinates` (the bare goal coordinates), so tables can keep native and derived inputs apart.
- **Task replay:** `task_replay` re-judges each phase of the robot's `EpisodeRecord.phases` entry over its recorded span with the predicates the node used online (`arena_simulation_setup.shared.judge`), from the robot's `task_pose` stream (the pose the online judge saw), the other robots' `task_pose`, recorded pedestrians and their gesture clips (`ped_gestures`, joined to the `peds` message held at each pose by `ped_id`), the semantic snapshot, the world's zone, door and elevator cabin polygons and its wall grid (`robot sees <ped>` is unknown without one). Gesture and reach results and the moment of a goto's signal are taken from the record. `judge_agrees` is true when every phase the record closed as met or failed is met by the replay within its span, every phase still active at the end is not, and the replay's set of violated conditions equals the recorded one. `goal_condition_rate` is phases met over phases submitted. `unsafe` counts the recorded condition violations, which never count against `success`. Without a `task_pose` stream or the world's zones the verdict columns stay empty, and without a `ped_gestures` table a `<ped> plays <clip>` atom is unknown.
- **Edge cases:** `edge_case_criticality` and `edge_case_formation` score an episode only when the scenario its `scenario` task mode ran (`file` in the episode's `obstacles_params`, else `robots_params`) has a `provenance.yaml` beside its `scenario.yaml`, every output is `None` otherwise. The world and the scenario are read from the run snapshot first (`<run_dir>/worlds/<world>/`, where the recorder copies the directory of each scenario the run ran to `scenarios/<file>/`), else through the world's scenario resolver, and a changed provenance, scenario or world file is read again. The robot pose is resampled at 10 Hz, each sample holds the pedestrians of the latest `peds` message at most 0.1 s old by scenario name, and each pedestrian counts as a disc of the `radius` its scenario `humans` entry pins in its `agent` block, else 0.3 m. Stimuli come from the `timeline` semantic entity, whose `stimulus.<agent>.<need>` fields hold the sim time the agent took that stimulus (stamps more than 0.1 s before the first robot pose belong to an earlier episode and are ignored). `*_at` times are seconds since the first robot pose.
  - `edge_case_id`, `edge_case_steps`: the provenance `id` and `steps`.
  - `edge_case_agents`, `edge_case_scope`, `edge_case_missing`: the scored agents are the ones the provenance `owned` lists under `C`, `D` and `injected` that appear in the recording, unplayed encounter agents left out (`edge_case_scope` `own`), every pedestrian when none of the case's agents appears (`all`), and none when every recorded one is unplayed (`unplayed`, the criticality outputs below stay `None`). `edge_case_missing` lists the case's agents absent from the recording.
  - `edge_case_played`, `edge_case_unplayed`, `edge_case_triggered_at`: an agent of the provenance `encounters` that took no stimulus is unplayed and scored neither for PET nor against the robot. `edge_case_played` is true when every encounter agent took one, `edge_case_unplayed` lists the others, `edge_case_triggered_at` is the first stimulus an encounter agent took. All three are `None` for a recording without the `timeline` entity or a case without encounters.
  - `edge_case_min_clearance_m`, `edge_case_min_clearance_with`, `edge_case_min_clearance_at`: the smallest edge-to-edge distance to a scored agent, the agent and when.
  - `edge_case_min_ttc_s`, `edge_case_min_ttc_with`, `edge_case_min_ttc_at`: the smallest time to contact assuming both keep their current velocities (the pedestrians' from the recorded `peds_twists`, zero without them), 0 for discs already in contact, `None` when they would never touch.
  - `edge_case_pet_s`, `edge_case_pet_with`: the smallest gap, over the played `encounters` (`agent`, `point`), between the moments the robot and that agent pass closest to the point, each within 1.5 m of it, `None` without encounters or when either never comes that close.
  - `edge_case_freeze_s`: the time the robot moves under 0.05 m/s after the first 2 s and the start hold the case was designed with (provenance `robot.hold_s`, 0 without the `robot` block).
  - `edge_case_intrusion_s`: the time a scored agent is within 0.5 m edge to edge.
  - `edge_case_collided`, `edge_case_collision_with`, `edge_case_collision_at`, `edge_case_collision_robot_closing`, `edge_case_collision_ped_closing`, `edge_case_collision_fault`: the first contact (edge-to-edge distance at or under 0), the robot's and the pedestrian's speed toward each other at that moment and who was closing at 0.05 m/s or more (`robot`, `pedestrian`, `mutual`, `static`).
  - `edge_case_formation_*` follows the provenance `D` members against the provenance `formation` (`center` in the robot's level frame like the encounter points, both moved into the map frame by the level origin, `radius`, `variant`): `edge_case_formation_variant`, `edge_case_formation_members` (the `D` members), `edge_case_formation_absent` (members never recorded, the formation is scored on the others), `edge_case_formation_seen` (any member recorded), `edge_case_formation_formed_at` (the first moment every recorded member stands within the radius of the center, or of their centroid in that sample without one, from the members' first `form` stimulus on when one was taken), `edge_case_formation_held_s` (how long they stayed), `edge_case_formation_broke_at` (the first moment after that one stood outside or left the recording), `edge_case_formation_arrived` (the most members within the radius at once), `edge_case_formation_form_fired_at` and `edge_case_formation_break_fired_at` (the first `form` and `break` stimulus a member took).
- **Reference metrics:** Reference metrics (`pfi`, `mar`, `ped_path_deflection_m`) are computed across runs in `pipeline.process_benchmark`.

## Data Sources

Per-topic Parquet files are extracted by `MCAPReader` and aligned by `TopicAligner` onto the odom time axis.

## Usage

```bash
arena evaluation run --benchmark-dir <run_id>                          # extract and metrics
arena evaluation run --benchmark-dir <run_id> --report-manifest <name> # with report
```

Listing registered calculators:

```python
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import RobotParams

reg = MetricRegistry(RobotParams())
for m in reg.list_metrics():
    print(m["name"], m["outputs"])
```


## Mechanism metrics (mechanisms)

Robot behaviour at doors, elevators and alarms, replayed from the episode's `semantic_snapshot`
on the odom time axis (each sample holds until the next). Code: `ecological/mechanism_metrics.py`
(new keys), `ecological/semantic_metrics.py` and `ecological/regime_metrics.py` (robot variants),
shared plumbing in `ecological/mechanism_common.py`.

**Naming.** The recorder names a robot `<env>_<ns>` (`env_0_jackal`, `robot` column); semantic
`members` carry sim paths (`env_0/jackal`, peds `env_0/Pedestrian_5`); entities are
`<env>/<authored name>[/<level>]`. `RobotIdentity` matches all spellings. Ped tracks record
`peds_names` (same sim paths) from this version on; older recordings key peds by array index.
Multi-env snapshots are scoped to the robot's env. When a kind never lists a member in an episode
(recordings before per-agent members), robot presence falls back to world geometry.

**None rules.** A key is None when the semantic kind it needs is absent, when it needs world
geometry and the world asset is not local, or when it needs ped tracks and none were recorded.
Counts are 0 when the robot never comes near the mechanism; conditional durations
(`elevator_wait_to_board_s`) are None when the conditioning event never happens.

Constants: stationary robot <= 0.05 m/s, waiting ped < 0.1 m/s, doorway band = robot radius + 0.3 m,
call area = within the elevator's `activation_distance` of its cabin door centre and outside the
cabin, ride arrival = cabin entry within 3 s of being in another cabin or across a teleport step.

| Key | Unit | Definition |
|---|---|---|
| `boarded_before_exit` | count | robot boardings while the cabin reports `just_arrived` and holds another agent |
| `blocked_closing_door`, `blocked_closing_door_s` | count, s | door `blocked_close` episodes / seconds with the robot in the doorway band of the door segment (members fallback without geometry) |
| `ped_mechanism_delay_seconds` | ped*s | ped-seconds spent waiting in a closed door's activation radius or a closed elevator's call area (stranded peds) |
| `elevator_wait_to_board_s` | s | sum over boardings of call-area entry -> cabin entry |
| `elevator_missed_cycles` | count | complete cabin-door open -> closed cycles while the robot waited in the call area |
| `elevator_unnecessary_calls` | count | call-area visits without boarding during which the cabin door started opening with no ped in the call area |
| `entered_full_cabin` | count | boardings with `live_occupants` (or non-robot members) >= `capacity` just before entry; `full` when capacity unknown |
| `cabin_door_blocking_s` | s | robot stationary in the cabin doorway band while the cabin door is `open` or `closing` |
| `door_yield_violations` | count | robot door crossings with a ped approaching from the far side within 2 m of the door centre in the 2 s before |
| `elevator_queue_jumps` | count | boardings overtaking a ped who was in the call area when the robot arrived and still waited there |
| `alarm_egress_blocking_s` | s | robot stationary inside a zone with predicate `egress` while any schedule is `active` |
| `door_freeze_s` | s | robot stationary within a door's activation radius while the door is steady (`open`/`closed`) |
| `door_oscillations` | count | door re-openings after the first within one robot visit of its activation radius |
| `ran_red_signal` | count | robot entries into the signal's disc while `stop`: preset params `position`/`radius`, else the same-named door |
| `time_waiting_at_doors_robot` | s | robot is a door member while the door is `triggered` and not `open` (everyone: `time_waiting_at_doors`) |
| `elevator_rides_robot` | count | `just_arrived` rises with the robot in the cabin members within 3 s (everyone: `elevator_rides`) |
| `used_elevator_during_alarm_robot` | count | alarm windows overlapped by the robot being an elevator member (everyone: `used_elevator_during_alarm`) |
| `mechanism_states` | list | per door and elevator: `progress` (0 closed, 1 open) and `locked` at every sample of `time`, the trajectory plot's animation input (`ecological/mechanism_states.py`) |

Teleports that land within 2 s of an elevator arrival are rides and no longer cut the episode
to its longest single-floor segment (`pose_segments.elevator_arrival_times`).

### `mechanism` component

`ecological/mechanism_component.py` (`mechanism_component` calculator) reports `mechanism_score`
in [0, 1] (higher is better) and `mechanism_components` (how many inputs were available). The
score is the weighted mean over the `MECHANISM_COMPONENTS` keys that are not None for the episode;
a "lower is better" value x maps to `1 / (1 + x / scale)` (0 -> 1, x == scale -> 0.5), a
"higher is better" value in [0, 1] is used as is. All weights are 1.

| Key | Direction | Scale |
|---|---|---|
| `condition_success` | higher | - |
| `boarded_before_exit`, `elevator_missed_cycles`, `elevator_unnecessary_calls`, `entered_full_cabin`, `door_yield_violations`, `elevator_queue_jumps`, `door_oscillations`, `ran_red_signal`, `used_elevator_during_alarm_robot`, `entered_over_cap_zone` | lower | 1 |
| `blocked_closing_door_s`, `cabin_door_blocking_s` | lower | 5 s |
| `alarm_egress_blocking_s`, `door_freeze_s` | lower | 10 s |
| `ped_mechanism_delay_seconds`, `elevator_wait_to_board_s`, `time_waiting_at_doors_robot` | lower | 30 s |
