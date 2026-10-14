from __future__ import annotations

import json
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.ecological.condition_metrics import _world_cache
from arena_evaluation.processing.parquet_store import ParquetStore, TopicParquetStore
from arena_evaluation.processing.pipeline import _METRIC_DTYPES, ProcessingPipeline, _record_integrity, _status_row
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata, TopicBundle

S = 1_000_000_000
WORLD = "synthetic_integrity_world"
ROBOT = "env_0_jackal_0"
PHASES = {"jackal_0": {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": [], "map_poses": [[1.0, 1.0, 0.0]]}}
MOVED = "env_0/jackal_0 moved by hand"
RETASKED = "task parameters changed mid-episode"


def _meta(*integrity: str) -> RunMetadata:
    return RunMetadata(
        benchmark_id="run",
        planner="nav2",
        map=WORLD,
        stage="s0",
        robot_model=["jackal"],
        outcome_state=2,
        integrity=list(integrity),
        recording_started_at="2026-01-01T00:00:00+00:00",
        python_version="3.12",
        ros_distro="jazzy",
    )


def _descriptor(episode_dir: pathlib.Path) -> EpisodeDescriptor:
    return EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="run", episode_id=0, planner="nav2", stage="s0", map=WORLD)


def _records(*integrity: list[str]) -> pl.DataFrame:
    n = len(integrity)
    return pl.DataFrame(
        {
            "time_ns": [i * S for i in range(n)],
            "episode_id": [1] * n,
            "outcome_state": [1] * (n - 1) + [2],
            "outcome_info": [""] * (n - 1) + ["finished"],
            "integrity": pl.Series(list(integrity), dtype=pl.List(pl.String)),
            "phases": [json.dumps(PHASES)] * n,
        }
    )


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


def test_record_integrity_takes_the_last_record_by_time() -> None:
    records = _records([], [MOVED], [MOVED, RETASKED]).reverse()
    assert _record_integrity(records, _meta()) == [MOVED, RETASKED]
    assert _record_integrity(records.lazy(), None) == [MOVED, RETASKED]


def test_record_integrity_falls_back_to_the_episode_yaml() -> None:
    without_column = pl.DataFrame({"time_ns": [0], "outcome_state": [2]})
    assert _record_integrity(None, _meta(MOVED)) == [MOVED]
    assert _record_integrity(without_column, _meta(MOVED)) == [MOVED]
    assert _record_integrity(None, None) == []


def test_status_rows_carry_integrity_as_a_string_list_column(tmp_path: pathlib.Path) -> None:
    ep = _descriptor(tmp_path)
    compromised = _status_row(ep, _meta(), "jackal", "no_trajectory", "no odom rows", _records([], [MOVED]))
    from_yaml = _status_row(ep, _meta(RETASKED), "jackal", "no_recording", "no MCAP or empty extraction")
    intact = _status_row(ep, _meta(), "jackal", "no_recording", "no MCAP or empty extraction")
    assert compromised["integrity"] == [MOVED]
    assert from_yaml["integrity"] == [RETASKED]
    assert intact["integrity"] == []

    dest = tmp_path / "m.parquet"
    ParquetStore.write_rows([intact], dest, schema_overrides=_METRIC_DTYPES)
    df, _ = ParquetStore.read(dest)
    assert df.schema["integrity"] == pl.List(pl.String)
    assert df["integrity"].to_list() == [[]]


def test_metric_rows_carry_integrity_of_the_last_record(tmp_path: pathlib.Path) -> None:
    _world_cache[WORLD] = [_ZoneGeometry(name="kitchen", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), max_speed=None, quiet=False, restricted=False)]
    episode_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_000.mcap").touch()
    MetadataWriter.write(_meta(), episode_dir / "episode_000.yaml")
    track = [(t / 10, 5.0 - 0.4 * t, 5.0 - 0.4 * t) for t in range(11)] + [(t / 10, 1.0, 1.0) for t in range(11, 31)]
    bundle = TopicBundle(odom=_track(track), task_pose=_track(track), episode_record=_records([], [MOVED], [MOVED, RETASKED]))
    TopicParquetStore.write({ROBOT: bundle}, episode_dir / "topics")

    rows = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).process_episode(_descriptor(episode_dir))

    assert [row["robot"] for row in rows] == [ROBOT]
    assert rows[0]["integrity"] == [MOVED, RETASKED]
    df, _ = ParquetStore.read(episode_dir / "metrics.parquet")
    assert df.schema["integrity"] == pl.List(pl.String)
    assert df["integrity"].to_list() == [[MOVED, RETASKED]]
