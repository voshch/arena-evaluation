"""Unit tests for the Parquet layer (processing/parquet_store.py and the telemetry sink it reads).

Covers TopicParquetStore (the flat telemetry cache: completion stamp, one bundle per
robot over shared global tables, unreadable caches read as misses).
"""

from __future__ import annotations


import polars as pl
import yaml

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.telemetry.writer import TableSink
from arena_evaluation.storage.manifest import MetadataWriter
from arena_evaluation.storage.schemas import RunMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DF_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "value": pl.Float64,
        "label": pl.String,
    }
)

_META_BASE = dict(
    benchmark_id="bench1",
    planner="nav2",
    map="hospital_1",
    stage="final",
    recording_started_at="2026-01-01T00:00:00Z",
    python_version="3.12",
    ros_distro="humble",
    modes={"robots": ["scenario"]},
    params={"task.scenario.file": "case_1"},
)


def _df(n: int = 5, start: int = 1_000, step: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [start + i * step for i in range(n)],
            "value": [float(i) * 0.5 for i in range(n)],
            "label": [f"row{i}" for i in range(n)],
        },
        schema=_DF_SCHEMA,
    )


def _meta(**overrides) -> RunMetadata:
    kw = dict(_META_BASE)
    kw.update(overrides)
    return RunMetadata(**kw)


def test_metadata_modes_and_params_roundtrip_through_yaml(tmp_path):
    path = tmp_path / "episode.yaml"
    path.write_text(yaml.safe_dump(_META_BASE))

    meta = MetadataWriter.read(path)

    assert meta.modes == {"robots": ["scenario"]}
    assert meta.params == {"task.scenario.file": "case_1"}
    assert meta.model_dump(exclude_none=True)["modes"] == {"robots": ["scenario"]}




def test_metadata_exclude_none():
    meta = _meta(episode_id=None, arena_git_sha=None, outcome_info="", is_reference=False)
    dumped = meta.model_dump(exclude_none=True)
    assert "episode_id" not in dumped
    assert "arena_git_sha" not in dumped
    # empty-string and False fields survive exclude_none
    assert dumped["outcome_info"] == ""
    assert dumped["is_reference"] is False






# ---------------------------------------------------------------------------
# TopicParquetStore: the flat telemetry cache
# ---------------------------------------------------------------------------


def _odometry(robot: str, stamps: list[int]) -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "env_id": ["env_0"] * n,
            "robot_id": [robot] * n,
            "odom_x": [float(i) for i in range(n)],
            "odom_y": [0.0] * n,
            "odom_yaw": [0.0] * n,
            "odom_vx": [0.0] * n,
            "odom_wz": [0.0] * n,
            "drift_error_m": [0.0] * n,
        }
    )


def _static_transform() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [0],
            "parent_frame": ["map"],
            "child_frame": ["env_0/map"],
            "tx": [0.0],
            "ty": [0.0],
            "tz": [0.0],
            "qx": [0.0],
            "qy": [0.0],
            "qz": [0.0],
            "qw": [1.0],
        }
    )


def test_each_robot_gets_its_own_rows_and_the_globals_are_shared(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    pl.concat([_odometry("env_0_r1", [1000, 2000]), _odometry("env_0_r2", [3000])]).write_parquet(dest / "robot_odom.parquet")
    _static_transform().write_parquet(dest / "tf_static.parquet")

    restored = TopicParquetStore.read(dest)

    assert set(restored) == {"env_0_r1", "env_0_r2"}
    assert restored["env_0_r1"].robot_odom.collect()["t_ns"].to_list() == [1000, 2000]
    assert restored["env_0_r2"].robot_odom.collect()["t_ns"].to_list() == [3000]
    assert restored["env_0_r1"].tf_static is restored["env_0_r2"].tf_static
    assert restored["env_0_r1"].robot_tracks is None


def test_tables_read_back_with_their_declared_types(tmp_path):
    dest = tmp_path / "telemetry"
    sink = TableSink(dest)
    sink.append("robot_odom", {"t_ns": 1000, "env_id": "env_0", "robot_id": "env_0_r1", "odom_x": 1.0, "odom_y": 2.0, "odom_yaw": 0.0, "odom_vx": 0.5, "odom_wz": 0.1, "drift_error_m": None})
    sink.close()

    frame = TopicParquetStore.read(dest)["env_0_r1"].robot_odom.collect()

    assert frame.schema["t_ns"] == pl.Int64
    assert frame.schema["robot_id"] == pl.String
    assert frame.schema["odom_x"] == pl.Float64


def test_the_sink_sorts_a_table_written_out_of_order(tmp_path):
    dest = tmp_path / "telemetry"
    sink = TableSink(dest)
    for stamp in (3000, 1000, 2000):
        sink.append("robot_odom", {"t_ns": stamp, "env_id": "env_0", "robot_id": "env_0_r1", "odom_x": float(stamp), "odom_y": 0.0, "odom_yaw": 0.0, "odom_vx": 0.0, "odom_wz": 0.0, "drift_error_m": None})
    sink.close()

    assert pl.read_parquet(dest / "robot_odom.parquet")["t_ns"].to_list() == [1000, 2000, 3000]


def test_a_table_without_rows_is_absent_from_its_bundle(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    _odometry("env_0_r1", [1000]).write_parquet(dest / "robot_odom.parquet")

    bundle = TopicParquetStore.read(dest)["env_0_r1"]

    assert bundle.robot_odom is not None
    assert bundle.robot_plans is None
    assert bundle.episode_lifecycle is None


def test_read_missing_or_file_dir(tmp_path):
    assert TopicParquetStore.read(tmp_path / "nope") is None
    f = tmp_path / "afile"
    f.write_text("x")
    assert TopicParquetStore.read(f) is None


def test_read_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert TopicParquetStore.read(empty) is None


def test_read_globals_only_returns_none(tmp_path):
    """Global tables with no robot rows produce no bundles -> None."""
    dest = tmp_path / "cache"
    dest.mkdir()
    _static_transform().write_parquet(dest / "tf_static.parquet")
    assert TopicParquetStore.read(dest) is None


def test_a_corrupt_table_reads_as_a_cache_miss(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "robot_odom.parquet").write_bytes(b"not parquet data")
    assert TopicParquetStore.read(dest) is None


def test_cache_is_current_only_once_marked_complete(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    assert not TopicParquetStore.is_current(dest)

    TopicParquetStore.mark_complete(dest)

    assert (dest / TopicParquetStore.VERSION_FILE).read_text().strip() == str(TopicParquetStore.VERSION)
    assert TopicParquetStore.is_current(dest)


def test_cache_stamped_by_another_layout_is_not_current(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / TopicParquetStore.VERSION_FILE).write_text(f"{TopicParquetStore.VERSION + 1}\n")

    assert not TopicParquetStore.is_current(dest)


