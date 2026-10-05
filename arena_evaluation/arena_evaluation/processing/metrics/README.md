# Metrics

Metric calculators for the Arena evaluation pipeline. Subclasses of `BaseMetricCalculator` are registered by `MetricRegistry` and executed in topological order based on `DEPENDS_ON`. Results are saved to `combined_metrics.parquet`, one row per episode.

## Categories

| Folder | Focus |
|---|---|
| `social/` | Robot and pedestrian interaction: social forces, proxemics, gaze, disturbance |
| `ecological/` | Energy, acoustics, world condition compliance |
| `performance/` | Path, motion, time, collision, efficiency, clearance |
| `naturalness/` | Trajectory naturalness against unobstructed baselines |

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
- **VLN metrics:** `vln_metrics` scores against a geodesic distance field from the goal (the robot's last recorded `goal_pose` in the episode, so planners without a global plan are scored against their real goal), an 8-connected Dijkstra over the Theta* grid (same map, robot-radius dilation and start/goal clearing). `ne` is the field value at the final pose, `osr` whether any pose came within the goal tolerance, `success_geodesic` is `success` with `ne` within it, `spl_geodesic` weights that by the Theta* length (Anderson et al. 2018), `ndtw` compares the robot path to the Theta* path, both resampled every 0.25 m (Ilharco et al. 2019), `sdtw` is `success_geodesic * ndtw`, and `stuck` flags a failed episode with a 10 s stretch moving under 0.2 m and turning under 15 degrees (VLN-PE). The success threshold is the goal tolerance the task generator judged the episode by (`EpisodeRecord.goal_tolerance`), so a VLN suite sets `task.episode.goto_pose.tolerance.radius` to 3.0 (VLN-CE). Geodesic outputs are `None` when the map does not resolve or the pose falls on an unreachable cell, and every output but `ne` and `stuck` is `None` for recordings without a goal tolerance.
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

