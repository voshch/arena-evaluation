import json
import pathlib
import shutil

import pytest

import polars as pl

pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.common.caching import world_cache
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _ZoneGeometry,
)
from arena_evaluation.processing.metrics.ecological.condition_metrics import _wall_cache
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.contracts import check_benchmark
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata

S = 1_000_000_000
WORLD = "synthetic_pipeline_world"
ENV = "env_0"
ROBOT = "env_0_jackal_0"
PHASES = {
    "jackal_0": {
        "phases": [
            {
                "goto": [1.0, 1.0, 0.0],
                "tolerance_radius": 0.5,
                "tolerance_angle": 0.0,
                "hold_time": 0.0,
            }
        ],
        "conditions": [],
        "map_poses": [[1.0, 1.0, 0.0]],
    }
}
_PARAMS = {
    "robot_radius": "0.3",
    "max_angular_vel": "1.0",
    "linear_acc": "1.0",
    "linear_decel": "1.0",
    "robot_mass_kg": "10.0",
    "static_power_w": "0.0",
}


def _write_acoustic_world(base: pathlib.Path, world: str) -> None:
    """A minimal format-2 world snapshot so the acoustic barrier map resolves for a synthetic episode."""
    world_dir = base / "worlds" / world
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.yaml").write_text(
        "format: arena-world/2\nlevels:\n  - id: '0'\n    zones:\n      - name: open\n        polygon: [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]]\n"
    )


def _robot_frame(
    points: list[tuple[float, float, float]], columns: dict[str, str]
) -> dict[str, list]:
    """Flat robot rows of a track: the time, identity and the columns named by the mapping."""
    return {
        "t_ns": [int(t * S) for t, _, _ in points],
        "env_id": [ENV] * len(points),
        "robot_id": [ROBOT] * len(points),
        **{
            column: [value(x, y) for _, x, y in points]
            for column, value in columns.items()
        },
    }


def _fleet_tables(robot_id: str) -> dict[str, pl.DataFrame]:
    """The recorded descriptor and morphology of the robot, as the fleet message writes them."""
    return {
        "robot_descriptor": pl.DataFrame(
            {
                "t_ns": [0],
                "env_id": [ENV],
                "robot_id": [robot_id],
                "name": ["jackal_0"],
                "model": ["jackal"],
                "ns": ["/env_0/jackal_0"],
                "frame": ["env_0/odom"],
                "base_frame": ["env_0/jackal_0/base_link"],
            }
        ),
        "robot_params": pl.DataFrame(
            {
                "t_ns": [0] * len(_PARAMS),
                "env_id": [ENV] * len(_PARAMS),
                "robot_id": [robot_id] * len(_PARAMS),
                "param_key": list(_PARAMS),
                "param_value": list(_PARAMS.values()),
            }
        ),
    }


def _snapshot(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    n = len(rows)
    return pl.DataFrame(
        {
            "t_ns": [t for t, _, _ in rows],
            "env_id": [ENV] * n,
            "world": [WORLD] * n,
            "entity_name": ["env_0/jackal_0"] * n,
            "kind": ["robot"] * n,
            "field_name": [f for _, f, _ in rows],
            "field_kind": ["discrete"] * n,
            "value_str": [v for _, _, v in rows],
            "value_num": pl.Series([None] * n, dtype=pl.Float64),
            "value_bool": pl.Series([None] * n, dtype=pl.Boolean),
        }
    )


def _robot_rows(t: int, phase: str, met: str) -> list[tuple[int, str, str]]:
    return [
        (t, "phase", phase),
        (t, "met", met),
        (t, "failed", ""),
        (t, "dropped", ""),
        (t, "violated", ""),
    ]


def _episode(tmp_path: pathlib.Path) -> tuple[FolderManager, EpisodeDescriptor]:
    episode_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000"
    episode_dir.mkdir(parents=True)
    _write_acoustic_world(tmp_path / "run" / "episodes", WORLD)
    (episode_dir / "episode_000.mcap").touch()
    track = [(t / 10, 5.0 - 0.4 * t, 5.0 - 0.4 * t) for t in range(11)] + [
        (t / 10, 1.0, 1.0) for t in range(11, 31)
    ]
    topics = episode_dir / "topics"
    topics.mkdir()
    for name, frame in _fleet_tables(ROBOT).items():
        frame.write_parquet(topics / f"{name}.parquet")
    pl.DataFrame(
        _robot_frame(
            track,
            {
                "odom_x": lambda x, _: x,
                "odom_y": lambda _, y: y,
                "odom_yaw": lambda *_: 0.0,
                "odom_vx": lambda *_: 0.5,
                "odom_wz": lambda *_: 0.0,
                "drift_error_m": lambda *_: 0.0,
            },
        )
    ).write_parquet(topics / "robot_odom.parquet")
    pl.DataFrame(
        _robot_frame(
            track,
            {
                "x": lambda x, _: x,
                "y": lambda _, y: y,
                "z": lambda *_: 0.0,
                "yaw": lambda *_: 0.0,
                "vx": lambda *_: 0.0,
                "vy": lambda *_: 0.0,
                "vz": lambda *_: 0.0,
                "omega_z": lambda *_: 0.0,
                "speed_mps": lambda *_: 0.0,
                "pose_source": lambda *_: "tf_gt",
            },
        )
    ).write_parquet(topics / "robot_tracks.parquet")
    pl.DataFrame(
        _robot_frame(
            track,
            {
                "stamp_ns": lambda *_: 0,
                "pos_x": lambda x, _: x,
                "pos_y": lambda _, y: y,
                "yaw": lambda *_: 0.0,
            },
        )
    ).write_parquet(topics / "robot_task_pose.parquet")
    pl.DataFrame(
        {
            "t_ns": [0, 3 * S],
            "env_id": [ENV, ENV],
            "episode_id": [1, 1],
            "outcome_state": [1, 2],
            "outcome_info": ["", "finished"],
            "goal_uuid": ["", ""],
            "phases_yaml": [json.dumps(PHASES), json.dumps(PHASES)],
            "params_yaml": ["", ""],
            "conditions_yaml": ["", ""],
            "human_params": ["", ""],
            "start_t_ns": [0, 0],
            "seed": [7, 7],
        }
    ).write_parquet(topics / "episode_lifecycle.parquet")
    _snapshot(
        _robot_rows(0, "0", "") + _robot_rows(int(1.5 * S), "", "0")
    ).write_parquet(topics / "world_semantics.parquet")
    TopicParquetStore.mark_complete(topics)
    descriptor = EpisodeDescriptor(
        episode_dir=str(episode_dir),
        benchmark_id="run",
        episode_id=0,
        planner="nav2",
        stage="s0",
        map=WORLD,
    )
    return FolderManager(data_root=tmp_path), descriptor


def test_process_episode_replays_the_task_on_the_episode_world(
    tmp_path: pathlib.Path,
) -> None:
    world_cache[WORLD] = [
        _ZoneGeometry(
            name="kitchen",
            polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]),
            max_speed=None,
            quiet=False,
            restricted=False,
        )
    ]
    _wall_cache[WORLD] = None
    folders, descriptor = _episode(tmp_path)
    rows = ProcessingPipeline(folders, workers=1).process_episode(descriptor)
    assert [row["robot"] for row in rows] == [ROBOT]
    assert rows[0]["map"] == WORLD
    metrics_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000" / "metrics"
    phases = pl.read_parquet(metrics_dir / "fact_phase_outcomes.parquet")
    assert phases.get_column("outcome").to_list() == ["met"]
    assert phases.get_column("met").to_list() == [True]
    assert pl.read_parquet(metrics_dir / "fact_robots.parquet").get_column(
        "judge_agrees"
    ).to_list() == [True]
    assert pl.read_parquet(metrics_dir / "fact_judge_disagreements.parquet").height == 0
    assert not (metrics_dir / "dim_episodes.parquet").exists()
    assert (metrics_dir / "fact_episodes.parquet").exists()


def test_process_benchmark_writes_fact_and_dimension_tables_passing_contracts(
    tmp_path: pathlib.Path,
) -> None:
    world_cache[WORLD] = [
        _ZoneGeometry(
            name="kitchen",
            polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]),
            max_speed=None,
            quiet=False,
            restricted=False,
        )
    ]
    _wall_cache[WORLD] = None
    folders, descriptor = _episode(tmp_path)
    episode_dir = folders.episode_dir("run", 0)
    shutil.copytree(pathlib.Path(descriptor.episode_dir), episode_dir)
    MetadataWriter.write(
        RunMetadata(
            benchmark_id="run",
            episode_id=0,
            planner="nav2",
            local_planner="dwb",
            inter_planner="",
            modes={},
            params={},
            stage="s0",
            map=WORLD,
            outcome_state=2,
            outcome_info="finished",
            recording_started_at="2026-01-01T00:00:00Z",
            python_version="3.12",
            ros_distro="jazzy",
        ),
        episode_dir / "episode_000.yaml",
    )

    benchmark_dir = ProcessingPipeline(folders, workers=1).process_benchmark("run")
    assert benchmark_dir == tmp_path / "run"
    assert (benchmark_dir / "dim_episodes.parquet").exists()
    assert (benchmark_dir / "dim_metrics.parquet").exists()
    assert (benchmark_dir / "dim_world_geometry.parquet").exists()
    metrics_dir = episode_dir / "metrics"
    assert (metrics_dir / "fact_episodes.parquet").exists()
    assert (metrics_dir / "fact_robots.parquet").exists()
    assert check_benchmark(benchmark_dir).ok
