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
- **Edge cases:** `edge_case_criticality` and `edge_case_formation` score an episode only when the scenario its `scenario` task mode ran (`file` in the episode's `obstacles_params`, else `robots_params`) has a `provenance.yaml` beside its `scenario.yaml`, found through the world's scenario resolver, every output is `None` otherwise. The robot pose is resampled at 10 Hz, each sample holds the latest recorded pedestrians by scenario name, and pedestrians count as 0.3 m discs. The scored agents are the ones the provenance `owned` lists under `C`, `D` and `injected` that appear in the recording (`edge_case_scope` `own`), or every pedestrian when none does (`all`), with the absent ones in `edge_case_missing`. `edge_case_min_clearance_m` is edge to edge, `edge_case_min_ttc_s` assumes constant velocities, `edge_case_pet_s` is the smallest gap, over the provenance `encounters` (`agent`, `point`), between the moments the robot and that agent pass closest to the point, each within 1.5 m of it, `None` without encounters or when either never comes that close, with the agent in `edge_case_pet_with`, `edge_case_freeze_s` the time the robot moves under 0.05 m/s after the first 2 s, `edge_case_intrusion_s` the time a scored agent is within 0.5 m, and the collision outcome names the first contact and who was closing (`robot`, `pedestrian`, `mutual`, `static`). `*_at` times are seconds since the first robot pose. `edge_case_formation` follows the `D` members against the provenance `formation` (`center` in the robot's level frame like the encounter points, both moved into the map frame, `radius`): `formed_at` is the first moment all members stand within the radius, `held_s` how long they stayed, `broke_at` the first moment one left, `arrived` the most members within the radius at once.
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

