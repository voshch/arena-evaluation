# storage (Layer 2: Shared Schemas and Path Management)

Shared data types and path management utilities used across the evaluation pipeline.

---

## Files

| File | Purpose |
|---|---|
| `schemas.py` | Pydantic and dataclass type definitions |
| `folder_manager.py` | Path resolution and run discovery (`FolderManager`) |
| `manifest.py` | YAML reading and writing (`MetadataWriter`) |
| `exceptions.py` | Pipeline domain exceptions |

---

## schemas.py - Data Models

### `RunMetadata`

Pydantic model matching `metadata.yaml`.

```python
from arena_evaluation.storage.schemas import RunMetadata

meta = RunMetadata(
    benchmark_id="my_benchmark",
    planner="dwa",
    robot_model=["jackal"],
    map="map_empty",
    stage="stage_1",
    episodes_requested=10,
    suite_name="default",
    contest_name="dwa_vs_teb",
    inter_planner="",
    agent_name="",
    recording_started_at="2026-05-28T21:03:48+00:00",
    python_version="3.12.3",
    ros_distro="jazzy",
)
```

### `RunDescriptor`

Dataclass identifying a discovered run. Returned by `FolderManager.discover_runs()`.

```python
@dataclass(frozen=True)
class RunDescriptor:
    run_dir: str
    benchmark_id: str
    planner: str
    stage: str
```

### `TopicBundle`

Dataclass holding one Polars DataFrame per recorded topic.

```python
@dataclass
class TopicBundle:
    odom: pl.DataFrame | None
    scan: pl.DataFrame | None
    cmd_vel: pl.DataFrame | None
    joint_states: pl.DataFrame | None
    peds: pl.DataFrame | None
    peds_physics: pl.DataFrame | None
    ped_gestures: pl.DataFrame | None
    interactions: pl.DataFrame | None
    interaction_events: pl.DataFrame | None
    animation_states: pl.DataFrame | None
    episode_record: pl.DataFrame | None
    collision_events: pl.DataFrame | None
```

### `AlignedEpisodeBundle`

Dataclass produced after aligning topics onto the odom time axis.

```python
@dataclass
class AlignedEpisodeBundle:
    episode_id: int
    data: pl.DataFrame
    start_pos: list[float]
    goal_pos: list[float]
    num_pedestrians: int
```

### `RobotParams`

Dataclass loaded from `arena_robots` capabilities YAML.

```python
params = RobotParams.load("jackal")
```

### `PlotSpec`

Pydantic model for entries in report manifests.

---

## folder_manager.py - FolderManager

Resolves output paths relative to `data_root` with path traversal validation.

```python
from arena_evaluation.storage.folder_manager import FolderManager
from pathlib import Path

fm = FolderManager(data_root=Path("/opt/arena_ws/data"))

run_dir   = fm.run_dir("my_benchmark", "dwa", "stage_1")
mcap_path = fm.mcap_path(run_dir)
metrics   = fm.metrics_path(run_dir)
combined  = fm.combined_metrics_path("my_benchmark")
runs      = fm.discover_runs("my_benchmark")
```

---

## manifest.py - MetadataWriter

Helper for reading and writing `metadata.yaml` files.

```python
from arena_evaluation.storage.manifest import MetadataWriter
from pathlib import Path

dest = Path("/opt/arena_ws/data/recordings/20260528-210000/metadata.yaml")

MetadataWriter.write(metadata, dest)
metadata = MetadataWriter.read(dest)
MetadataWriter.update(dest, episodes_recorded=10, recording_ended_at="2026-05-28T22:00:00+00:00")
```

---

## exceptions.py

| Exception | Raised When |
|---|---|
| `MetricCalculationError` | Metric calculator fails |
| `CircularDependencyError` | Circular `DEPENDS_ON` chain detected |
| `SchemaViolationError` | Data violates schema requirements |
| `RobotNotFoundError` | Robot capability file is missing |
| `ManifestGenerationError` | Manifest read/write fails |

