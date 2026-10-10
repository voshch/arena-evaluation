# processing (Layer 3: Offline Metric Computation)

Computes navigational and ecological metrics from recorded episode MCAP files offline without requiring active ROS 2 nodes.

Pipeline data flow:

```
MCAPReader -> TopicParquetStore -> TopicAligner -> MetricRegistry -> ParquetStore
```

Each `episode_XXX.mcap` is aligned directly onto its odom axis.

---

## Files

| File | Purpose |
|---|---|
| `mcap_reader.py` | Decodes MCAP to `TopicBundle` (one DataFrame per topic) |
| `parquet_store.py` | Reads and writes extraction cache (`TopicParquetStore`) and output metrics (`ParquetStore`) |
| `topic_aligner.py` | Aligns multi-rate topics onto the odom time axis |
| `pipeline.py` | `ProcessingPipeline` orchestrates benchmark processing |
| `map_registry.py` | Discovers, converts (PGM to PNG), and caches map images |
| `metrics/` | Metric calculators (performance, social, ecological, naturalness) |

---

## mcap_reader.py - MCAPReader

Reads episode MCAP files using Python `mcap` libraries and produces a `TopicBundle`.

```python
from arena_evaluation.processing.mcap_reader import MCAPReader

bundle = MCAPReader(path).read()
```

Positions in the shared `map` frame (`tf_gt`, `plan`, `initialpose`, `peds`, episode start and goal) are moved into the env's world frame by the recorded `map -> env_<n>/map` anchor. A world swap moves the anchor mid-recording, so each row takes the anchor in effect at its `time_ns`: the last one at or before it, a later message winning a tie, the first one for rows before any.

### Extracted Fields per Topic

| Topic | Polars Columns |
|---|---|
| odom | `time_ns`, `stamp_ns`, `pos_x`, `pos_y`, `yaw`, `vel_linear`, `vel_angular` |
| odom_controller | same as odom, from the controller's `<*_controller>/odom`; preferred over odom when present |
| scan | `time_ns`, `scan_ranges`, `scan_min` |
| cmd_vel | `time_ns`, `linear_x`, `linear_y`, `linear_z`, `angular_x`, `angular_y`, `angular_z` |
| cmd_vel_controller | same as cmd_vel, from the controller's stamped `<*_controller>/cmd_vel`; used when the bare cmd_vel is absent |
| joint_states | `time_ns`, `name`, `position`, `velocity`, `effort` |
| power | `time_ns`, `total_power_w`, `static_power_w`, `total_mechanical_power_w`, `total_thermal_power_w`, `joint_*_power_w` |
| energy | `time_ns`, `total_energy_consumed_wh`, `battery_soc_percent` |
| acoustics | `time_ns`, `total_level_af_dba`, `total_level_zf_db`, `baseline_level_dba`, `drivetrain_level_dba`, `uncertainty_1sigma_dba`, `validity_flags` |
| characterization_phase | `time_ns`, `label` |
| peds | `time_ns`, `peds_positions`, `peds_headings`, `num_pedestrians` |
| episode_record | `time_ns`, `episode_id`, `outcome_state`, `outcome_info`, `goal_uuid`, `robots_params` |
| collision_events | `time_ns`, `collision_event` |
| collision_monitor_state | `time_ns`, `action_type`, `polygon_name` |
| tf / tf_static | `time_ns`, frame/child/trans/rot columns |
| tf_gt | `time_ns`, `stamp_ns_gt`, `pos_x_gt`, `pos_y_gt`, `yaw_gt`, `frame_id` |

---

## topic_aligner.py - TopicAligner

Aligns topics onto the odom time axis using `join_asof` with backward-looking tolerance.

```python
aligned_df = TopicAligner(tolerance_ns=100_000_000).align(bundle)
```

- Odom is the primary time axis.
- Columns are preserved without prefixes.
- Missing values in the aligned frame appear as nulls.

---

## pipeline.py - ProcessingPipeline

Orchestrates extraction and metric computation.

```python
fm = FolderManager(data_root=Path("/opt/arena_ws/data"))
ProcessingPipeline(fm, workers=-1).process_benchmark("my_benchmark")
```

For each episode:
1. Reads `episode_XXX.yaml` metadata.
2. Loads `RobotParams`.
3. Loads cached topics or extracts them via `MCAPReader`.
4. Resolves the pose stream (`pose_anchor.py`): dense `tf_gt` as is, sparse `tf_gt` anchors the odom track onto the map frame with one rigid fit, none leaves the track in the odom frame without pedestrian-relative metrics.
5. Aligns topics with `TopicAligner`.
6. Computes metrics with `MetricRegistry`.
7. Attaches run identifiers and writes `metrics.parquet`.

Every recorded episode yields a row. `status` is `evaluated`, `path_only` (no map-frame pose), `no_trajectory`, `no_recording` or `error`, with `status_reason`; rows without a trajectory carry the runtime verdict in `result` and `success` and null metrics. `pose_source` (`tf`, `anchored`, `tf_sparse`, `odom`), `pose_samples` and `pose_anchor_residual_m` say how the robot pose was obtained. The report lists every non-`evaluated` row above the summary.

---

## Open-Loop Characterization

Characterization is evaluated by `CharacterizationCalculator` (`metrics/ecological/characterization.py`). It maps recorded `characterization_phase` markers across working points and outputs per-episode timeseries columns. The presentation layer derives working point aggregates directly from `combined_metrics.parquet`.

---

## Metric Calculator Framework

### Adding a Metric

1. Create a file in `metrics/<category>/`.
2. Subclass `BaseMetricCalculator` and define class attributes.
3. Implement `output_keys()` and `calculate()`.
4. Set `UNITS` dictionary for display labels.

```python
from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

class MyMetricCalculator(BaseMetricCalculator):
    NAME = "my_metric"
    CATEGORY = "performance"
    REQUIRES_PEDSIM = False
    DEPENDS_ON = ["path_metrics"]
    REQUIRED_TOPICS = ["odom"]
    UNITS = {"my_value": "m/s"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["my_value"]

    def calculate(self, episode: AlignedEpisodeBundle, robot_params: RobotParams, prior_results: dict) -> dict:
        return {"my_value": 1.0}
```

`MetricRegistry` discovers all `BaseMetricCalculator` classes, orders them topologically by `DEPENDS_ON`, and handles errors gracefully with `None` fills.

