from __future__ import annotations

import json
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("arena_simulation_setup.shared.judge")

import shapely

from arena_evaluation.processing.metrics.common.caching import world_cache
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import ProcessingPipeline, _record_integrity, _status_row
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata

S = 1_000_000_000
WORLD = "synthetic_integrity_world"
ENV = "env_0"
ROBOT = "env_0_jackal_0"
PHASES = {"jackal_0": {"phases": [{"goto": [1.0, 1.0, 0.0], "tolerance_radius": 0.5, "tolerance_angle": 0.0, "hold_time": 0.0}], "conditions": [], "map_poses": [[1.0, 1.0, 0.0]]}}
MOVED = "env_0/jackal_0 moved by hand"
RETASKED = "task parameters changed mid-episode"
_PARAMS = {
    "robot_radius": "0.3",
    "max_angular_vel": "1.0",
    "linear_acc": "1.0",
    "linear_decel": "1.0",
    "robot_mass_kg": "10.0",
    "static_power_w": "0.0",
}


def _write_fleet_topics(topics: pathlib.Path) -> None:
    """The recorded descriptor and morphology of the robot, as the fleet message writes them."""
    pl.DataFrame(
        {
            "t_ns": [0],
            "env_id": [ENV],
            "robot_id": [ROBOT],
            "name": ["jackal_0"],
            "model": ["jackal"],
            "ns": ["/env_0/jackal_0"],
            "frame": ["env_0/odom"],
            "base_frame": ["env_0/jackal_0/base_link"],
        }
    ).write_parquet(topics / "robot_descriptor.parquet")
    pl.DataFrame(
        {
            "t_ns": [0] * len(_PARAMS),
            "env_id": [ENV] * len(_PARAMS),
            "robot_id": [ROBOT] * len(_PARAMS),
            "param_key": list(_PARAMS),
            "param_value": list(_PARAMS.values()),
        }
    ).write_parquet(topics / "robot_params.parquet")


def _write_acoustic_world(base: pathlib.Path, world: str) -> None:
    """A minimal format-2 world snapshot so the acoustic barrier map resolves for a synthetic episode."""
    world_dir = base / "worlds" / world
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.yaml").write_text("format: arena-world/2\nlevels:\n  - id: '0'\n    zones:\n      - name: open\n        polygon: [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]]\n")


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
        modes={},
        params={},
    )


def _descriptor(episode_dir: pathlib.Path) -> EpisodeDescriptor:
    return EpisodeDescriptor(episode_dir=str(episode_dir), benchmark_id="run", episode_id=0, planner="nav2", stage="s0", map=WORLD)


def _records(*integrity: list[str]) -> pl.DataFrame:
    """The episode's records as episode_lifecycle rows, one per message."""
    n = len(integrity)
    return pl.DataFrame(
        {
            "t_ns": [i * S for i in range(n)],
            "env_id": [ENV] * n,
            "episode_id": [1] * n,
            "outcome_state": [1] * (n - 1) + [2],
            "outcome_info": [""] * (n - 1) + ["finished"],
            "phases_yaml": [json.dumps(PHASES)] * n,
            "conditions_yaml": [""] * n,
            "human_params": [""] * n,
        }
    )


def _flags(*integrity: list[str]) -> pl.DataFrame:
    """The integrity flags each record carried, as episode_integrity rows."""
    rows = [(i * S, ENV, index, flag) for i, reasons in enumerate(integrity) for index, flag in enumerate(reasons)]
    return pl.DataFrame(rows, schema={"t_ns": pl.Int64, "env_id": pl.String, "flag_idx": pl.Int64, "flag": pl.String}, orient="row")


def _write_topics(topics: pathlib.Path, integrity: tuple[list[str], ...]) -> None:
    """A driven track of 3 s at 10 Hz with its episode records, as a complete flat topic cache."""
    topics.mkdir()
    samples = 31
    stamps = [int(i * S / 10) for i in range(samples)]
    xs = [0.1 * i for i in range(samples)]
    pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": [ENV] * samples,
            "robot_id": [ROBOT] * samples,
            "odom_x": xs,
            "odom_y": [0.0] * samples,
            "odom_yaw": [0.0] * samples,
            "odom_vx": [1.0] * samples,
            "odom_wz": [0.0] * samples,
            "drift_error_m": [0.0] * samples,
        }
    ).write_parquet(topics / "robot_odom.parquet")
    pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": [ENV] * samples,
            "robot_id": [ROBOT] * samples,
            "x": xs,
            "y": [0.0] * samples,
            "z": [0.0] * samples,
            "yaw": [0.0] * samples,
            "vx": [1.0] * samples,
            "vy": [0.0] * samples,
            "vz": [0.0] * samples,
            "omega_z": [0.0] * samples,
            "speed_mps": [1.0] * samples,
            "pose_source": ["tf_gt"] * samples,
        }
    ).write_parquet(topics / "robot_tracks.parquet")
    pl.DataFrame(
        {"t_ns": [0], "env_id": [ENV], "robot_id": [ROBOT], "name": ["jackal_0"], "model": ["jackal"],
         "ns": ["/env_0/jackal_0"], "frame": ["env_0/odom"], "base_frame": ["env_0/jackal_0/base_link"]}
    ).write_parquet(topics / "robot_descriptor.parquet")
    pl.DataFrame(
        {"t_ns": [0, 0], "env_id": [ENV] * 2, "robot_id": [ROBOT] * 2,
         "param_key": ["robot_radius", "robot_mass_kg"], "param_value": [0.25, 40.0]}
    ).write_parquet(topics / "robot_params.parquet")
    _write_fleet_topics(topics)
    _records(*integrity).write_parquet(topics / "episode_lifecycle.parquet")
    _flags(*integrity).write_parquet(topics / "episode_integrity.parquet")
    TopicParquetStore.mark_complete(topics)


def test_record_integrity_takes_the_flags_of_the_last_record_by_time() -> None:
    integrity = ([], [MOVED], [MOVED, RETASKED])
    assert _record_integrity(_records(*integrity).reverse(), _flags(*integrity), _meta()) == [MOVED, RETASKED]
    assert _record_integrity(_records(*integrity).lazy(), _flags(*integrity).lazy(), None) == [MOVED, RETASKED]


def test_record_integrity_falls_back_to_the_episode_yaml_when_no_record_carries_any() -> None:
    without_flags = pl.DataFrame({"t_ns": [0], "outcome_state": [2]})
    assert _record_integrity(None, None, _meta(MOVED)) == [MOVED]
    assert _record_integrity(without_flags, None, _meta(MOVED)) == [MOVED]
    assert _record_integrity(_records([]), _flags([]), _meta(MOVED)) == [MOVED]
    assert _record_integrity(None, None, None) == []


def test_status_rows_carry_integrity_of_the_records(tmp_path: pathlib.Path) -> None:
    ep = _descriptor(tmp_path)
    compromised = _status_row(ep, _meta(), "jackal", "no_trajectory", "no odom rows", _records([], [MOVED]), _flags([], [MOVED]))
    from_yaml = _status_row(ep, _meta(RETASKED), "jackal", "no_recording", "no MCAP or empty extraction")
    intact = _status_row(ep, _meta(), "jackal", "no_recording", "no MCAP or empty extraction")
    assert compromised["integrity"] == [MOVED]
    assert from_yaml["integrity"] == [RETASKED]
    assert intact["integrity"] == []


def test_metric_rows_carry_integrity_of_the_last_record(tmp_path: pathlib.Path) -> None:
    world_cache[WORLD] = [_ZoneGeometry(name="kitchen", polygon=shapely.Polygon([(0, 0), (0, 4), (4, 4), (4, 0)]), max_speed=None, quiet=False, restricted=False)]
    episode_dir = tmp_path / "run" / "episodes" / "nav2" / "episode_000"
    episode_dir.mkdir(parents=True)
    _write_acoustic_world(tmp_path / "run" / "episodes", WORLD)
    (episode_dir / "episode_000.mcap").touch()
    MetadataWriter.write(_meta(), episode_dir / "episode_000.yaml")
    _write_topics(episode_dir / "topics", ([], [MOVED], [MOVED, RETASKED]))

    rows = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).process_episode(_descriptor(episode_dir))

    assert [row["robot"] for row in rows] == [ROBOT]
    assert rows[0]["integrity"] == [MOVED, RETASKED]
    # Integrity is episode provenance, not a metric: the Star Schema keeps flat scalar facts and
    # the list no longer reaches Parquet (invariant 3). Dimensions are benchmark-level.
    assert (episode_dir / "metrics" / "fact_episodes.parquet").exists()
    assert not (episode_dir / "metrics" / "dim_episodes.parquet").exists()
