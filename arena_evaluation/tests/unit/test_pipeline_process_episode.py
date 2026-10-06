import json
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.ecological.condition_metrics import _world_cache
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import EpisodeDescriptor, TopicBundle

S = 1_000_000_000
WORLD = "synthetic_pipeline_world"
ROBOT = "env_0_jackal_0"
PHASES = {"jackal_0": {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": [], "map_poses": [[1.0, 1.0, 0.0]]}}


def _track(points: list[tuple[float, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [int(t * S) for t, _, _ in points],
            "stamp_ns": [int(t * S) for t, _, _ in points],
            "pos_x": [x for _, x, _ in points],
            "pos_y": [y for _, _, y in points],
            "yaw": [0.0] * len(points),
            "vel_linear": [0.5] * len(points),
            "vel_angular": [0.0] * len(points),
        }
    )


def _snapshot(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [t for t, _, _ in rows],
            "env_id": [0] * len(rows),
            "world": [WORLD] * len(rows),
            "entity": ["env_0/jackal_0"] * len(rows),
            "kind": ["robot"] * len(rows),
            "field": [f for _, f, _ in rows],
            "field_kind": ["discrete"] * len(rows),
            "value_str": [v for _, _, v in rows],
            "value_num": pl.Series([None] * len(rows), dtype=pl.Float64),
            "value_bool": pl.Series([None] * len(rows), dtype=pl.Boolean),
        }
    )


def _robot_rows(t: int, phase: str, met: str) -> list[tuple[int, str, str]]:
    return [(t, "phase", phase), (t, "met", met), (t, "failed", ""), (t, "dropped", ""), (t, "violated", "")]


def _episode(tmp_path: pathlib.Path) -> tuple[FolderManager, EpisodeDescriptor]:
    episode_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()
    track = [(t / 10, 5.0 - 0.4 * t, 5.0 - 0.4 * t) for t in range(11)] + [(t / 10, 1.0, 1.0) for t in range(11, 31)]
    bundle = TopicBundle(
        odom=_track(track),
        task_pose=_track(track),
        episode_record=pl.DataFrame(
            {
                "time_ns": [0, 3 * S],
                "episode_id": [1, 1],
                "outcome_state": [1, 2],
                "outcome_info": ["", "finished"],
                "phases": [json.dumps(PHASES), json.dumps(PHASES)],
            }
        ),
        semantic_snapshot=_snapshot(_robot_rows(0, "0", "") + _robot_rows(int(1.5 * S), "", "0")),
    )
    TopicParquetStore.write({ROBOT: bundle}, episode_dir / "topics")
    descriptor = EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="run", episode_id=0, planner="nav2", stage="s0", map=WORLD)
    return FolderManager(data_root=tmp_path), descriptor


def test_process_episode_replays_the_task_on_the_episode_world(tmp_path: pathlib.Path) -> None:
    _world_cache[WORLD] = [_ZoneGeometry(name="kitchen", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), max_speed=None, quiet=False, restricted=False)]
    folders, descriptor = _episode(tmp_path)
    rows = ProcessingPipeline(folders, workers=1).process_episode(descriptor)
    assert [row["robot"] for row in rows] == [ROBOT]
    assert rows[0]["map"] == WORLD
    assert rows[0]["phase_outcomes"] == ["met"]
    assert rows[0]["judge_agrees"] is True
    assert rows[0]["judge_disagreements"] == []
    assert (tmp_path / "run" / "episodes" / "nav2" / "episode_000" / "metrics.parquet").exists()
