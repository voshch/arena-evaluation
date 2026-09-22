"""Unit tests for the Parquet topic cache layer (processing/parquet_store.py).

Covers ParquetStore (metric frames with Pydantic metadata in the schema footer)
and TopicParquetStore (per-topic parquet caches for a dict of TopicBundles,
global-vs-robot file layout, zstd compression, overwrite semantics, sorting).
"""

from __future__ import annotations

import json
import pathlib

import polars as pl
import pyarrow.parquet as pq
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from arena_evaluation.processing.parquet_store import ParquetStore, TopicParquetStore
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DF_SCHEMA = pl.Schema(
    {
        "time_ns": pl.Int64,
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
)


def _df(n: int = 5, start: int = 1_000, step: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [start + i * step for i in range(n)],
            "value": [float(i) * 0.5 for i in range(n)],
            "label": [f"row{i}" for i in range(n)],
        },
        schema=_DF_SCHEMA,
    )


def _meta(**overrides) -> RunMetadata:
    kw = dict(_META_BASE)
    kw.update(overrides)
    return RunMetadata(**kw)


def _write_src(tmp_path: pathlib.Path, name: str, df: pl.DataFrame) -> pathlib.Path:
    src = tmp_path / name
    ParquetStore.write(df, src)
    return src


# ---------------------------------------------------------------------------
# ParquetStore.write / read
# ---------------------------------------------------------------------------


def test_write_read_roundtrip_no_metadata(tmp_path):
    df = _df()
    dest = tmp_path / "metrics.parquet"
    ParquetStore.write(df, dest)
    assert dest.exists()

    back, metadata = ParquetStore.read(dest)
    assert metadata is None
    assert back.equals(df)
    assert back.schema == _DF_SCHEMA
    # footer carries no arena metadata key
    footer_meta = pq.read_schema(dest).metadata or {}
    assert ParquetStore.METADATA_KEY.encode() not in footer_meta


def test_write_creates_parent_directories(tmp_path):
    dest = tmp_path / "a" / "b" / "c" / "metrics.parquet"
    ParquetStore.write(_df(2), dest)
    assert dest.exists()


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ParquetStore.read(tmp_path / "nope.parquet")


def test_metadata_roundtrip_through_footer(tmp_path):
    """Pydantic metadata must round-trip through the parquet schema footer."""
    df = _df(4)
    meta = _meta(episode_id=7, outcome_state=2, obstacles_params={"wall": True}, agent_name="ada")
    dest = tmp_path / "metrics.parquet"
    ParquetStore.write(df, dest, metadata=meta)

    back, metadata = ParquetStore.read(dest)
    assert back.equals(df)
    assert back.schema == _DF_SCHEMA
    assert metadata == meta.model_dump(exclude_none=True)
    assert metadata["episode_id"] == 7
    assert metadata["obstacles_params"] == {"wall": True}

    # raw footer check: the JSON lives under the METADATA_KEY in the footer
    footer = pq.read_schema(dest)
    key = ParquetStore.METADATA_KEY.encode()
    assert footer.metadata is not None and key in footer.metadata
    assert json.loads(footer.metadata[key].decode()) == meta.model_dump(exclude_none=True)


def test_metadata_exclude_none():
    meta = _meta(episode_id=None, tm_obstacles=None, outcome_info="", is_reference=False)
    dumped = meta.model_dump(exclude_none=True)
    assert "episode_id" not in dumped
    assert "tm_obstacles" not in dumped
    # empty-string and False fields survive exclude_none
    assert dumped["outcome_info"] == ""
    assert dumped["is_reference"] is False


def test_read_metadata_none_when_key_absent(tmp_path):
    dest = _write_src(tmp_path, "m.parquet", _df(3))
    _, metadata = ParquetStore.read(dest)
    assert metadata is None


def test_write_none_metadata_keeps_footer_clean(tmp_path):
    dest = tmp_path / "m.parquet"
    ParquetStore.write(_df(2), dest, metadata=None)
    footer_meta = pq.read_schema(dest).metadata or {}
    assert ParquetStore.METADATA_KEY.encode() not in footer_meta


# ---------------------------------------------------------------------------
# ParquetStore.combine
# ---------------------------------------------------------------------------


def test_combine_empty_sources_noop(tmp_path):
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([], dest)
    assert not dest.exists()


def test_combine_single_source(tmp_path):
    src = _write_src(tmp_path, "a.parquet", _df(3))
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([src], dest)
    back, meta = ParquetStore.read(dest)
    assert meta is None
    assert back.equals(_df(3))


def test_combine_multiple_diagonal(tmp_path):
    df_a = _df(3)
    df_b = pl.DataFrame(
        {"time_ns": [10_000], "other": ["x"]},
        schema=pl.Schema({"time_ns": pl.Int64, "other": pl.String}),
    )
    src_a = _write_src(tmp_path, "a.parquet", df_a)
    src_b = _write_src(tmp_path, "b.parquet", df_b)
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([src_a, src_b], dest)

    back, _ = ParquetStore.read(dest)
    assert back.height == 4
    assert "label" in back.columns and "other" in back.columns
    assert back.filter(pl.col("label").is_not_null()).height == 3
    assert back.filter(pl.col("other").is_not_null()).height == 1


def test_combine_schema_mismatch_raises(tmp_path):
    df_a = pl.DataFrame({"a": [1, 2]}, schema=pl.Schema({"a": pl.Int64}))
    df_b = pl.DataFrame({"a": [[1], [2]]}, schema=pl.Schema({"a": pl.List(pl.Int64)}))
    src_a = _write_src(tmp_path, "a.parquet", df_a)
    src_b = _write_src(tmp_path, "b.parquet", df_b)
    dest = tmp_path / "combined.parquet"
    with pytest.raises(SchemaViolationError):
        ParquetStore.combine([src_a, src_b], dest)
    assert not dest.exists()


def test_combine_skips_unreadable_sources(tmp_path, capsys):
    # NOTE: the module logger is hijacked by the `launch` LaunchLogger
    # (propagate=False), so caplog cannot see the warning; assert behaviour
    # (and stderr output where the LaunchLogger handler writes it) instead.
    good = _write_src(tmp_path, "good.parquet", _df(2))
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([good, tmp_path / "missing.parquet"], dest)
    back, _ = ParquetStore.read(dest)
    assert back.equals(_df(2))


def test_combine_all_sources_unreadable(tmp_path):
    dest = tmp_path / "combined.parquet"
    ParquetStore.combine([tmp_path / "m1.parquet", tmp_path / "m2.parquet"], dest)
    assert not dest.exists()


@given(
    n=st.integers(min_value=0, max_value=20),
    with_str=st.booleans(),
    with_bool=st.booleans(),
)
@settings(
    max_examples=100,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_roundtrip_hypothesis(tmp_path, n, with_str, with_bool):
    cols = {
        "a": [float(i) for i in range(n)],
        "t": [i * 1000 for i in range(n)],
    }
    schema: dict = {"a": pl.Float64, "t": pl.Int64}
    if with_str:
        cols["s"] = [f"x{i}" for i in range(n)]
        schema["s"] = pl.String
    if with_bool:
        cols["b"] = [i % 2 == 0 for i in range(n)]
        schema["b"] = pl.Boolean
    df = pl.DataFrame(cols, schema=pl.Schema(schema))

    # repeated (n, flags) examples may share a file; writes are idempotent
    dest = tmp_path / f"h_{n}_{with_str}_{with_bool}.parquet"
    ParquetStore.write(df, dest, metadata=None)
    back, metadata = ParquetStore.read(dest)
    assert metadata is None
    assert back.schema == df.schema
    assert back.equals(df)


# ---------------------------------------------------------------------------
# TopicParquetStore.write / read
# ---------------------------------------------------------------------------


def _topics(tmp_path: pathlib.Path) -> tuple[dict[str, TopicBundle], pathlib.Path]:
    """Two-robot bundle set: shared globals + per-robot odom/plan."""
    odom_1 = _df(4, 1_000)
    odom_2 = _df(3, 5_000)
    plan_1 = _df(2, 1_000)
    plan_2 = _df(2, 5_000)
    peds = _df(3)
    episode_record = _df(2)
    tf_static = pl.DataFrame(
        {"time_ns": [1000], "frame_id": ["map"], "child_frame_id": ["env_0/map"]},
        schema=pl.Schema({"time_ns": pl.Int64, "frame_id": pl.String, "child_frame_id": pl.String}),
    )
    semantic_snapshot = pl.DataFrame(
        {"time_ns": [1000], "kind": ["door"], "entity": ["env_0/d1"]},
        schema=pl.Schema({"time_ns": pl.Int64, "kind": pl.String, "entity": pl.String}),
    )
    bundles = {
        "robot_1": TopicBundle(odom=odom_1, plan=plan_1, peds=peds, episode_record=episode_record),
        "robot_2": TopicBundle(
            odom=odom_2,
            plan=plan_2,
            peds=peds,
            episode_record=episode_record,
            tf_static=tf_static,
            semantic_snapshot=semantic_snapshot,
        ),
    }
    return bundles, tmp_path / "cache"


def test_topic_write_read_roundtrip(tmp_path):
    bundles, dest = _topics(tmp_path)
    TopicParquetStore.write(bundles, dest)

    # layout: globals at root, robot topics under robot dirs
    assert (dest / "peds.parquet").exists()
    assert (dest / "episode_record.parquet").exists()
    assert (dest / "tf_static.parquet").exists()
    assert (dest / "semantic_snapshot.parquet").exists()
    assert not (dest / "odom.parquet").exists()
    assert not (dest / "plan.parquet").exists()
    assert (dest / "robot_1" / "odom.parquet").exists()
    assert (dest / "robot_1" / "plan.parquet").exists()
    assert not (dest / "robot_1" / "peds.parquet").exists()
    assert (dest / "robot_2" / "odom.parquet").exists()

    restored = TopicParquetStore.read(dest)
    assert restored is not None
    assert set(restored) == {"robot_1", "robot_2"}

    r1 = restored["robot_1"]
    assert r1.odom is not None and r1.plan is not None
    assert r1.odom.collect().equals(pl.DataFrame(bundles["robot_1"].odom).sort("time_ns"))
    assert r1.odom.collect().schema == _DF_SCHEMA
    # globals are shared by reference between robots
    assert restored["robot_1"].peds is restored["robot_2"].peds
    assert restored["robot_1"].peds.collect().schema == _DF_SCHEMA
    assert restored["robot_2"].tf_static is not None
    assert restored["robot_2"].semantic_snapshot is not None


def test_topic_globals_written_once_for_multiple_robots(tmp_path):
    bundles, dest = _topics(tmp_path)
    TopicParquetStore.write(bundles, dest)
    # peds appears exactly once, at the root
    assert list(dest.glob("peds.parquet")) == [dest / "peds.parquet"]
    assert list(dest.glob("*/peds.parquet")) == []


def test_topic_empty_frames_not_written(tmp_path):
    empty = pl.DataFrame({"time_ns": pl.Series([], dtype=pl.Int64)}, schema={"time_ns": pl.Int64})
    dest = tmp_path / "cache"
    TopicParquetStore.write({"r1": TopicBundle(odom=empty)}, dest)
    assert not (dest / "r1" / "odom.parquet").exists()
    restored = TopicParquetStore.read(dest)
    assert restored is not None
    assert restored["r1"].odom is None


def test_topic_lazy_frames_collected(tmp_path):
    dest = tmp_path / "cache"
    lf = _df(3).lazy()
    TopicParquetStore.write({"r1": TopicBundle(odom=lf)}, dest)
    assert (dest / "r1" / "odom.parquet").exists()
    restored = TopicParquetStore.read(dest)
    assert isinstance(restored["r1"].odom, pl.LazyFrame)
    assert restored["r1"].odom.collect().equals(_df(3).sort("time_ns"))


def test_topic_overwrite_semantics_lazyframe(tmp_path):
    dest = tmp_path / "cache"
    original = _df(2, 1_000)
    replacement = _df(2, 9_000)
    TopicParquetStore.write({"r1": TopicBundle(odom=original)}, dest)
    assert (dest / "r1" / "odom.parquet").exists()

    # overwrite=False keeps the existing file
    TopicParquetStore.write({"r1": TopicBundle(odom=replacement.lazy())}, dest, overwrite=False)
    r1 = TopicParquetStore.read(dest)["r1"]
    assert r1.odom.collect().equals(original.sort("time_ns"))

    # overwrite=True replaces it
    TopicParquetStore.write({"r1": TopicBundle(odom=replacement.lazy())}, dest, overwrite=True)
    r1 = TopicParquetStore.read(dest)["r1"]
    assert r1.odom.collect().equals(replacement.sort("time_ns"))


def test_topic_dataframe_always_writes_regardless_of_overwrite_flag(tmp_path):
    dest = tmp_path / "cache"
    TopicParquetStore.write({"r1": TopicBundle(odom=_df(2, 1_000))}, dest)
    TopicParquetStore.write({"r1": TopicBundle(odom=_df(2, 9_000))}, dest, overwrite=False)
    r1 = TopicParquetStore.read(dest)["r1"]
    assert r1.odom.collect().equals(_df(2, 9_000).sort("time_ns"))


def test_topic_read_missing_or_file_dir(tmp_path):
    assert TopicParquetStore.read(tmp_path / "nope") is None
    f = tmp_path / "afile"
    f.write_text("x")
    assert TopicParquetStore.read(f) is None


def test_topic_read_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert TopicParquetStore.read(empty) is None


def test_topic_read_globals_only_returns_none(tmp_path):
    """Root-only topics with no robot dirs produce no bundles -> None."""
    dest = tmp_path / "cache"
    dest.mkdir()
    _df(2).write_parquet(dest / "peds.parquet")
    assert TopicParquetStore.read(dest) is None


def test_topic_read_peds_fallback(tmp_path):
    dest = tmp_path / "cache"
    peds = _df(2)
    # fallback-only layout: robot dir plus peds_fallback.parquet at root
    (dest / "robot_1").mkdir(parents=True)
    _df(3).write_parquet(dest / "robot_1" / "odom.parquet")
    peds.write_parquet(dest / "peds_fallback.parquet")
    restored = TopicParquetStore.read(dest)
    assert restored is not None
    assert restored["robot_1"].peds.collect().equals(peds.sort("time_ns"))


def test_topic_read_prefers_real_peds_over_fallback(tmp_path):
    dest = tmp_path / "cache"
    (dest / "robot_1").mkdir(parents=True)
    _df(2).write_parquet(dest / "robot_1" / "odom.parquet")
    _df(2, 1_000).write_parquet(dest / "peds.parquet")
    _df(2, 8_000).write_parquet(dest / "peds_fallback.parquet")
    restored = TopicParquetStore.read(dest)
    assert restored["robot_1"].peds.collect()["time_ns"].max() == 1_100


def test_topic_read_sorts_by_time_ns(tmp_path):
    dest = tmp_path / "cache"
    unsorted = pl.DataFrame(
        {"time_ns": [3000, 1000, 2000], "value": [3.0, 1.0, 2.0]},
        schema=pl.Schema({"time_ns": pl.Int64, "value": pl.Float64}),
    )
    TopicParquetStore.write({"r1": TopicBundle(odom=unsorted)}, dest)
    restored = TopicParquetStore.read(dest)
    assert restored["r1"].odom.collect()["time_ns"].to_list() == [1000, 2000, 3000]


def test_topic_read_keeps_frame_without_time_ns(tmp_path):
    dest = tmp_path / "cache"
    no_time = pl.DataFrame({"value": [1.0, 2.0]}, schema=pl.Schema({"value": pl.Float64}))
    TopicParquetStore.write({"r1": TopicBundle(plan=no_time)}, dest)
    restored = TopicParquetStore.read(dest)
    assert restored["r1"].plan is not None
    assert restored["r1"].plan.collect().equals(no_time)


def test_topic_read_corrupt_parquet_skipped(tmp_path):
    # NOTE: the module logger is hijacked by the `launch` LaunchLogger
    # (propagate=False), so the warning cannot be observed via caplog; assert
    # the behavioural contract instead: corrupt files leave the field None.
    dest = tmp_path / "cache"
    (dest / "robot_1").mkdir(parents=True)
    (dest / "robot_1" / "odom.parquet").write_bytes(b"not parquet data")
    restored = TopicParquetStore.read(dest)
    assert restored["robot_1"].odom is None


def test_topic_strict_schema_preserved(tmp_path):
    df = pl.DataFrame(
        {
            "time_ns": [1, 2],
            "v": [1.5, 2.5],
            "s": ["a", "b"],
            "b": [True, False],
            "l": [[1, 2], [3]],
        },
        schema=pl.Schema(
            {
                "time_ns": pl.Int64,
                "v": pl.Float64,
                "s": pl.String,
                "b": pl.Boolean,
                "l": pl.List(pl.Int64),
            }
        ),
    )
    dest = tmp_path / "cache"
    TopicParquetStore.write({"r1": TopicBundle(odom=df)}, dest)
    restored = TopicParquetStore.read(dest)
    assert restored["r1"].odom.collect().schema == df.schema
    assert restored["r1"].odom.collect().equals(df.sort("time_ns"))


def test_topic_write_empty_bundle_dict(tmp_path):
    dest = tmp_path / "cache"
    TopicParquetStore.write({}, dest)
    assert dest.is_dir()
    assert TopicParquetStore.read(dest) is None


def test_write_rows_status_rows_first_keep_metric_columns(tmp_path: pathlib.Path) -> None:
    # status rows finish first under as_completed
    status = {"episode": 0, "status": "no_trajectory", "success": False}
    metric = {"episode": 1, "status": "evaluated", "success": True, "time": [1, 2, 3], "path_length": 4.2}
    dest = tmp_path / "combined.parquet"
    ParquetStore.write_rows([dict(status, episode=i) for i in range(150)] + [metric], dest, schema_overrides={"success": pl.Boolean})
    df, _ = ParquetStore.read(dest)
    assert df.columns[-2:] == ["time", "path_length"]
    assert df["time"].null_count() == 150
    assert df["path_length"].drop_nulls().to_list() == [4.2]
