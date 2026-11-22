"""Unit tests for the Parquet layer (processing/parquet_store.py and the telemetry sink it reads)."""

from __future__ import annotations


import polars as pl

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.storage.schemas import RunMetadata

# Helpers

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


def test_metadata_exclude_none():
    meta = _meta(episode_id=None, arena_git_sha=None, outcome_info="", is_reference=False)
    dumped = meta.model_dump(exclude_none=True)
    assert "episode_id" not in dumped
    assert "arena_git_sha" not in dumped
    # empty-string and False fields survive exclude_none
    assert dumped["outcome_info"] == ""
    assert dumped["is_reference"] is False


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


def test_read_missing_or_file_dir(tmp_path):
    assert TopicParquetStore.read(tmp_path / "nope") is None
    f = tmp_path / "afile"
    f.write_text("x")
    assert TopicParquetStore.read(f) is None


def test_a_corrupt_table_reads_as_a_cache_miss(tmp_path):
    dest = tmp_path / "cache"
    dest.mkdir()
    (dest / "robot_odom.parquet").write_bytes(b"not parquet data")
    assert TopicParquetStore.read(dest) is None
