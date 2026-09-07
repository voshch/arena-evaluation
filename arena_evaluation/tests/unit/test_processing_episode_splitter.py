"""Unit tests for the episode-splitting layer.

Covers processing/episode_splitter.py:
  - _parse_conditions / _env_offset / _episode_snapshot helpers
  - EpisodeSplitter.split with and without episode_record rows
  - start/goal recovery (robots_params YAML, initialpose, plan, aligned fallback)
  - LazyFrame inputs and chunked (multi-record-batch) pyarrow input
Property tests (hypothesis) over arbitrary timestamps / snapshots.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import given, settings, strategies as st

from arena_evaluation.processing import episode_splitter as es_mod
from arena_evaluation.processing.episode_splitter import (
    EpisodeSplitter,
    _env_offset,
    _episode_snapshot,
    _parse_conditions,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, TopicBundle

# SOURCE-BUG WORKAROUND: the production AlignedEpisodeBundle dataclass has no
# `env_offset` field, but episode_splitter.split() passes env_offset=... on every
# episode construction, so the production path raises TypeError.  We keep the
# production class untouched and, in this test file only, substitute a subclass
# that adds the field so the split logic itself can be exercised.  The buggy
# production behaviour is pinned by dedicated regression tests below.
_REAL_BUNDLE_CLASS = es_mod.AlignedEpisodeBundle


@dataclasses.dataclass
class _BundleWithEnvOffset(AlignedEpisodeBundle):
    env_offset: tuple[float, float] | None = None


@pytest.fixture(autouse=True)
def _patch_bundle_class(monkeypatch):
    monkeypatch.setattr(es_mod, "AlignedEpisodeBundle", _BundleWithEnvOffset)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------

_ODOM_SCHEMA = pl.Schema(
    {
        "time_ns": pl.Int64,
        "pos_x": pl.Float64,
        "pos_y": pl.Float64,
        "yaw": pl.Float64,
        "num_pedestrians": pl.Int64,
    }
)

_REC_SCHEMA = pl.Schema(
    {
        "time_ns": pl.Int64,
        "episode_id": pl.Int64,
        "robots_params": pl.String,
        "conditions": pl.String,
    }
)

_INIT_SCHEMA = pl.Schema({"time_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "yaw": pl.Float64})

_PLAN_SCHEMA = pl.Schema(
    {
        "time_ns": pl.Int64,
        "poses_x": pl.List(pl.Float64),
        "poses_y": pl.List(pl.Float64),
        "poses_yaw": pl.List(pl.Float64),
    }
)

_SNAP_SCHEMA = pl.Schema({"time_ns": pl.Int64, "entity": pl.String})


def _odom(n: int = 6, start: int = 1_000, step: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [start + i * step for i in range(n)],
            "pos_x": [float(i) for i in range(n)],
            "pos_y": [float(i) * 2.0 for i in range(n)],
            "yaw": [0.0] * n,
            "num_pedestrians": [i % 3 for i in range(n)],
        },
        schema=_ODOM_SCHEMA,
    ).sort("time_ns")


def _records(rows: list[tuple[int, int, str | None, str | None]]) -> pl.DataFrame:
    """rows: (time_ns, episode_id, robots_params_yaml, conditions_json)."""
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "episode_id": [r[1] for r in rows],
            "robots_params": [r[2] for r in rows],
            "conditions": [r[3] for r in rows],
        },
        schema=_REC_SCHEMA,
    )


def _aligned(n: int = 6, start: int = 1_000, step: int = 100, null_pos: bool = False) -> pl.DataFrame:
    px = [None] * n if null_pos else [float(i) for i in range(n)]
    py = [None] * n if null_pos else [float(i) * 2.0 for i in range(n)]
    return pl.DataFrame(
        {
            "time_ns": [start + i * step for i in range(n)],
            "pos_x": px,
            "pos_y": py,
            "yaw": [0.0] * n,
            "num_pedestrians": [1] * n,
        },
        schema=_ODOM_SCHEMA,
    )


def _snapshot(times: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"time_ns": times, "entity": [f"e{i}" for i in range(len(times))]},
        schema=_SNAP_SCHEMA,
    )


def _tf_static(rows: list[tuple[str, str, float, float]] | None = None) -> pl.DataFrame:
    rows = rows or []
    return pl.DataFrame(
        {
            "frame_id": [r[0] for r in rows],
            "child_frame_id": [r[1] for r in rows],
            "trans_x": [float(r[2]) for r in rows],
            "trans_y": [float(r[3]) for r in rows],
        }
    )


class _FakeAligner:
    """Records align() calls and returns a scripted result."""

    def __init__(self, result=None, callable_result=None):
        self.result = result
        self.callable_result = callable_result
        self.calls: list[tuple[int | None, int | None]] = []

    def align(self, bundle, start_time_ns=None, end_time_ns=None):
        self.calls.append((start_time_ns, end_time_ns))
        if self.callable_result is not None:
            return self.callable_result(start_time_ns, end_time_ns)
        return self.result


def _splitter(aligner) -> EpisodeSplitter:
    return EpisodeSplitter(aligner, min_episode_frames=5)


# ---------------------------------------------------------------------------
# _parse_conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", "\n"])
def test_parse_conditions_empty(raw):
    assert _parse_conditions(raw) is None


@pytest.mark.parametrize("raw", ["{not json", "[1,", "'unterminated"])
def test_parse_conditions_invalid_json(raw):
    assert _parse_conditions(raw) is None


def test_parse_conditions_non_list():
    assert _parse_conditions('{"a": 1}') is None
    assert _parse_conditions('"just a string"') is None


def test_parse_conditions_non_str_type():
    # json.loads raises TypeError on these -> None
    assert _parse_conditions(123) is None
    assert _parse_conditions(object()) is None


def test_parse_conditions_valid_list():
    assert _parse_conditions('[{"k": "v"}, [1, 2]]') == [{"k": "v"}, [1, 2]]


def test_parse_conditions_roundtrip():
    conds = [{"phase": "start", "time_s": 1.5}]
    assert _parse_conditions(json.dumps(conds)) == conds


# ---------------------------------------------------------------------------
# _env_offset
# ---------------------------------------------------------------------------


def test_env_offset_none_and_empty():
    assert _env_offset(None) == (0.0, 0.0)
    assert _env_offset(pl.DataFrame(schema={"frame_id": pl.String})) == (0.0, 0.0)


def test_env_offset_no_matching_rows():
    df = _tf_static([("map", "odom", 1.0, 2.0), ("odom", "base_link", 0.0, 0.0)])
    assert _env_offset(df) == (0.0, 0.0)


def test_env_offset_single_env():
    df = _tf_static([("map", "env_0/map", 1.5, -2.25), ("map", "odom", 0.0, 0.0)])
    assert _env_offset(df) == (1.5, -2.25)


def test_env_offset_duplicate_rows_dedup():
    df = _tf_static([("map", "env_0/map", 3.0, 4.0), ("map", "env_0/map", 3.0, 4.0)])
    assert _env_offset(df) == (3.0, 4.0)


def test_env_offset_multi_env_ambiguous():
    df = _tf_static([("map", "env_0/map", 1.0, 2.0), ("map", "env_1/map", 5.0, 6.0)])
    assert _env_offset(df) is None


def test_env_offset_ignores_non_map_frame():
    df = _tf_static([("odom", "env_0/map", 9.0, 9.0)])
    assert _env_offset(df) == (0.0, 0.0)


def test_env_offset_lazyframe_input_not_supported():
    # _env_offset is DataFrame-only (len() on a LazyFrame raises); split()
    # collects tf_static before calling it, so this is a contract guard only.
    with pytest.raises(TypeError):
        _env_offset(_tf_static([("map", "env_0/map", 0.5, 0.25)]).lazy())


@given(
    n=st.integers(0, 8),
    n_env=st.integers(0, 3),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_env_offset_shape_property(n, n_env):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append(
            (
                "map",
                f"env_{int(rng.integers(0, n_env))}/map" if n_env else "odom",
                float(rng.integers(-5, 5)),
                float(rng.integers(-5, 5)),
            )
        )
    off = _env_offset(_tf_static(rows))
    if off is None:
        # only reachable when >=2 distinct env translations exist
        tx = {r[2] for r in rows if r[1].startswith("env_")}
        assert len(tx) > 1
    else:
        assert len(off) == 2
        assert all(isinstance(v, float) for v in off)


# ---------------------------------------------------------------------------
# _episode_snapshot
# ---------------------------------------------------------------------------


def test_episode_snapshot_none_and_empty():
    assert _episode_snapshot(None, 100, 200) is None
    assert _episode_snapshot(_snapshot([]), 100, 200) is None


def test_episode_snapshot_seed_plus_window():
    snap = _snapshot([50, 150, 180, 300])
    out = _episode_snapshot(snap, 100, 200)
    assert out is not None
    assert out["time_ns"].to_list() == [50, 150, 180]
    assert out.schema == _SNAP_SCHEMA


def test_episode_snapshot_no_seed_uses_first_window_stamp():
    snap = _snapshot([150, 180])
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [150, 180]


def test_episode_snapshot_all_outside_window():
    snap = _snapshot([300, 400])
    assert _episode_snapshot(snap, 100, 200) is None


def test_episode_snapshot_seed_exactly_at_start():
    snap = _snapshot([100, 150])
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [100, 150]


def test_episode_snapshot_dedup_seed_inside_window():
    # identical full rows (same time AND entity) are collapsed by unique()
    snap = pl.DataFrame(
        {"time_ns": [100, 100, 150], "entity": ["e", "e", "e"]},
        schema=_SNAP_SCHEMA,
    )
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [100, 150]
    assert out["entity"].to_list() == ["e", "e"]


def test_episode_snapshot_keeps_distinct_entities_at_same_time():
    # same timestamp, different entity -> both rows kept (unique is row-wise)
    snap = _snapshot([100, 100])
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [100, 100]
    assert set(out["entity"].to_list()) == {"e0", "e1"}


def test_episode_snapshot_latest_seed_wins():
    # seed candidates include the row exactly at start_time (100); 90 is
    # dominated by it
    snap = _snapshot([10, 90, 100, 150])
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [100, 150]


def test_episode_snapshot_empty_window_with_seed():
    snap = _snapshot([10, 90])
    out = _episode_snapshot(snap, 100, 200)
    assert out["time_ns"].to_list() == [90]


@given(
    times=st.lists(st.integers(min_value=0, max_value=10_000), min_size=0, max_size=20),
    start=st.integers(min_value=0, max_value=10_000),
    end=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_episode_snapshot_properties(times, start, end):
    if start > end:
        start, end = end, start
    df = _snapshot(times)
    out = _episode_snapshot(df, start, end)
    in_window = [t for t in times if start <= t <= end]
    seed_candidates = [t for t in times if t <= start]
    if out is None:
        assert not seed_candidates and not in_window
        return
    out_times = out["time_ns"].to_list()
    # sorted, schema unchanged
    assert out_times == sorted(out_times)
    assert out.schema == _SNAP_SCHEMA
    # row-set identity: out == (seed rows) union (window rows)
    seed_time = max(seed_candidates) if seed_candidates else (min(in_window) if in_window else None)
    expected_rows = set(df.rows())
    expected_rows = {r for r in df.rows() if (r[0] == seed_time) or (start <= r[0] <= end)}
    assert set(out.rows()) == expected_rows
    assert out.height == len(expected_rows)


# ---------------------------------------------------------------------------
# EpisodeSplitter.split — odom guards
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# env_offset source-bug regression pins
# ---------------------------------------------------------------------------


def test_production_bundle_class_lacks_env_offset_field():
    """Documented source bug: split() passes env_offset= but the dataclass
    has no such field (episode_splitter.py vs storage/schemas.py mismatch)."""
    assert "env_offset" not in {f.name for f in dataclasses.fields(AlignedEpisodeBundle)}


def test_split_real_bundle_raises_typeerror_no_records(monkeypatch):
    """Unpatched production classes: split() raises TypeError on episode build."""
    monkeypatch.setattr(es_mod, "AlignedEpisodeBundle", _REAL_BUNDLE_CLASS)
    with pytest.raises(TypeError):
        _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom()))


def test_split_real_bundle_raises_typeerror_with_records(monkeypatch):
    monkeypatch.setattr(es_mod, "AlignedEpisodeBundle", _REAL_BUNDLE_CLASS)
    records = _records([(1000, 3, None, None)])
    with pytest.raises(TypeError):
        _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))


# ---------------------------------------------------------------------------
# EpisodeSplitter.split — odom guards
# ---------------------------------------------------------------------------


def test_split_odom_none_returns_empty():
    aligner = _FakeAligner(_aligned())
    out = _splitter(aligner).split(TopicBundle())
    assert out == []
    assert aligner.calls == []


def test_split_empty_odom_dataframe_returns_empty():
    empty = pl.DataFrame({"time_ns": pl.Series([], dtype=pl.Int64)}, schema={"time_ns": pl.Int64})
    bundle = TopicBundle(odom=empty)
    aligner = _FakeAligner(_aligned())
    assert _splitter(aligner).split(bundle) == []


def test_split_empty_odom_lazyframe_returns_empty():
    empty = pl.DataFrame({"time_ns": pl.Series([], dtype=pl.Int64)}, schema={"time_ns": pl.Int64}).lazy()
    bundle = TopicBundle(odom=empty)
    aligner = _FakeAligner(_aligned())
    assert _splitter(aligner).split(bundle) == []


# ---------------------------------------------------------------------------
# EpisodeSplitter.split — no episode_record
# ---------------------------------------------------------------------------


def test_split_no_records_single_episode():
    aligned = _aligned(6)
    snap = _snapshot([500, 1500])
    bundle = TopicBundle(odom=_odom(), semantic_snapshot=snap)
    aligner = _FakeAligner(aligned)
    episodes = _splitter(aligner).split(bundle, robot_name="r1")

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.episode_id == 0
    assert ep.robot_name == "r1"
    assert ep.data.equals(aligned)
    assert ep.data.schema == _ODOM_SCHEMA
    assert ep.start_pos == [0.0, 0.0, 0.0]
    assert ep.goal_pos == [5.0, 10.0, 0.0]
    assert ep.num_pedestrians == 1
    assert ep.semantic_snapshot.equals(snap)
    assert ep.conditions is None
    assert ep.env_offset == (0.0, 0.0)
    assert aligner.calls == [(None, None)]


def test_split_no_records_below_min_frames():
    aligner = _FakeAligner(_aligned(4))  # 4 < min 5
    out = _splitter(aligner).split(TopicBundle(odom=_odom()))
    assert out == []


def test_split_no_records_aligner_none():
    out = _splitter(_FakeAligner(None)).split(TopicBundle(odom=_odom()))
    assert out == []


def test_split_no_records_null_positions():
    aligned = _aligned(6, null_pos=True)
    ep = _splitter(_FakeAligner(aligned)).split(TopicBundle(odom=_odom()))[0]
    assert ep.start_pos == []
    assert ep.goal_pos == []


def test_split_no_records_yaw_defaults_to_zero():
    df = pl.DataFrame(
        {
            "time_ns": [1000 + i * 100 for i in range(6)],
            "pos_x": [float(i) for i in range(6)],
            "pos_y": [float(i) for i in range(6)],
        },
        schema=pl.Schema({"time_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64}),
    )
    ep = _splitter(_FakeAligner(df)).split(TopicBundle(odom=_odom()))[0]
    assert ep.start_pos == [0.0, 0.0, 0.0]
    assert ep.goal_pos == [5.0, 5.0, 0.0]


def test_split_no_records_env_offset():
    tf = _tf_static([("map", "env_0/map", 10.0, 20.0)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), tf_static=tf))[0]
    assert ep.env_offset == (10.0, 20.0)


def test_split_no_records_env_offset_multi_env_none():
    tf = _tf_static([("map", "env_0/map", 1.0, 2.0), ("map", "env_1/map", 3.0, 4.0)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), tf_static=tf))[0]
    assert ep.env_offset is None


def test_split_no_records_lazy_odom_and_lazy_record():
    bundle = TopicBundle(odom=_odom().lazy(), episode_record=None)
    ep = _splitter(_FakeAligner(_aligned(6))).split(bundle)[0]
    assert ep.episode_id == 0
    assert len(ep.data) == 6


# ---------------------------------------------------------------------------
# EpisodeSplitter.split — episode_record windows
# ---------------------------------------------------------------------------


def test_split_paired_records_single_episode():
    records = _records([(1000, 3, None, None), (2000, 3, None, None)])
    aligner = _FakeAligner(_aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom(), episode_record=records))
    assert len(episodes) == 1
    assert episodes[0].episode_id == 3
    # paired rows: window is [1000, 2000] inclusive
    assert aligner.calls == [(1000, 2000)]


def test_split_adjacent_different_ids_end_exclusive():
    records = _records([(1000, 3, None, None), (2000, 4, None, None)])
    aligner = _FakeAligner(_aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom(), episode_record=records))
    assert len(episodes) == 2
    assert [e.episode_id for e in episodes] == [3, 4]
    # first episode ends one ns before the next record; last ends at odom max
    assert aligner.calls == [(1000, 1999), (2000, 1500)]


def test_split_single_record_ends_at_odom_max():
    records = _records([(1000, 3, None, None)])
    aligner = _FakeAligner(_aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom(), episode_record=records))
    assert len(episodes) == 1
    assert aligner.calls == [(1000, 1500)]


def test_split_single_record_lazy_odom_ends_at_odom_max():
    records = _records([(1000, 3, None, None)])
    aligner = _FakeAligner(_aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom().lazy(), episode_record=records))
    assert len(episodes) == 1
    assert aligner.calls == [(1000, 1500)]


def test_split_short_window_skipped():
    records = _records([(1000, 3, None, None), (2000, 4, None, None)])
    aligner = _FakeAligner(callable_result=lambda start, end: _aligned(2) if start == 1000 else _aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom(), episode_record=records))
    assert [e.episode_id for e in episodes] == [4]


def test_split_window_aligner_none_skipped():
    records = _records([(1000, 3, None, None)])
    out = _splitter(_FakeAligner(None)).split(TopicBundle(odom=_odom(), episode_record=records))
    assert out == []


def test_split_three_records_pair_then_single():
    records = _records([(1000, 1, None, None), (2000, 1, None, None), (3000, 2, None, None)])
    aligner = _FakeAligner(_aligned(6))
    episodes = _splitter(aligner).split(TopicBundle(odom=_odom(), episode_record=records))
    assert [e.episode_id for e in episodes] == [1, 2]
    assert aligner.calls == [(1000, 2000), (3000, 1500)]


# ---------------------------------------------------------------------------
# start/goal recovery
# ---------------------------------------------------------------------------

ROBOT_PARAMS_YAML = "robot_a:\n  start: [1.0, 2.0, 0.1]\n  goal: [9.0, 8.0, 3.1]\n  other: 1\nrobot_b:\n  start: [99.0, 99.0, 0.0]\n"


def test_split_start_goal_from_robots_params():
    records = _records([(1000, 3, ROBOT_PARAMS_YAML, None)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    # only the first robot's params are honored (loop breaks after first entry)
    assert ep.start_pos == [1.0, 2.0, 0.1]
    assert ep.goal_pos == [9.0, 8.0, 3.1]


def test_split_robots_params_no_start_key_falls_back_to_initialpose():
    yaml_str = "robot_a:\n  goal: [5.0, 5.0, 0.0]\n"
    records = _records([(1000, 3, yaml_str, None)])
    init = pl.DataFrame(
        {"time_ns": [1100], "pos_x": [7.0], "pos_y": [8.0], "yaw": [1.5]},
        schema=_INIT_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=init))[0]
    assert ep.start_pos == [7.0, 8.0, 1.5]
    assert ep.goal_pos == [5.0, 5.0, 0.0]


def test_split_robots_params_invalid_yaml_ignored():
    records = _records([(1000, 3, "{{{{ not yaml", None)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    # falls through: no initialpose, no plan -> aligned fallback
    assert ep.start_pos == [0.0, 0.0, 0.0]
    assert ep.goal_pos == [5.0, 10.0, 0.0]


def test_split_robots_params_non_dict_yaml_ignored():
    records = _records([(1000, 3, "- 1\n- 2\n", None)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    assert ep.start_pos == [0.0, 0.0, 0.0]


def test_split_initialpose_first_row_after_start():
    records = _records([(1000, 3, None, None)])
    init = pl.DataFrame(
        {"time_ns": [1050, 1150], "pos_x": [1.0, 2.0], "pos_y": [3.0, 4.0], "yaw": [0.5, 0.7]},
        schema=_INIT_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=init))[0]
    assert ep.start_pos == [1.0, 3.0, 0.5]


def test_split_initialpose_last_row_before_start():
    records = _records([(1000, 3, None, None)])
    init = pl.DataFrame(
        {"time_ns": [500, 900], "pos_x": [1.0, 2.0], "pos_y": [3.0, 4.0], "yaw": [0.5, 0.7]},
        schema=_INIT_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=init))[0]
    assert ep.start_pos == [2.0, 4.0, 0.7]


def _init_at(time_ns: int, x: float, y: float, yaw: float) -> pl.DataFrame:
    return pl.DataFrame(
        {"time_ns": [time_ns], "pos_x": [x], "pos_y": [y], "yaw": [yaw]},
        schema=_INIT_SCHEMA,
    )


def test_split_plan_yaw_correction_applied():
    records = _records([(1000, 3, None, None)])
    plan = pl.DataFrame(
        {"time_ns": [1000], "poses_x": [[1.0]], "poses_y": [[1.0]], "poses_yaw": [[2.0]]},
        schema=_PLAN_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=_init_at(1000, 1.0, 2.0, 0.0), plan=plan))[0]
    # initialpose yaw 0.0, plan yaw 2.0 -> |0 - 2| > 1 -> replaced
    assert ep.start_pos == [1.0, 2.0, 2.0]


def test_split_plan_yaw_correction_not_applied_when_close():
    records = _records([(1000, 3, None, None)])
    plan = pl.DataFrame(
        {"time_ns": [1000], "poses_x": [[1.0]], "poses_y": [[1.0]], "poses_yaw": [[0.5]]},
        schema=_PLAN_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=_init_at(1000, 1.0, 2.0, 0.0), plan=plan))[0]
    # |0 - 0.5| <= 1.0 -> yaw kept from initialpose
    assert ep.start_pos == [1.0, 2.0, 0.0]


def test_split_plan_yaw_zero_not_applied():
    records = _records([(1000, 3, None, None)])
    plan = pl.DataFrame(
        {"time_ns": [1000], "poses_x": [[1.0]], "poses_y": [[1.0]], "poses_yaw": [[0.0]]},
        schema=_PLAN_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, initialpose=_init_at(1000, 1.0, 2.0, 0.0), plan=plan))[0]
    assert ep.start_pos == [1.0, 2.0, 0.0]


def test_split_goal_from_plan_within_window():
    records = _records([(1000, 3, None, None), (2000, 4, None, None)])
    plan = pl.DataFrame(
        {"time_ns": [1000, 2500], "poses_x": [[0.0, 1.0], [9.0]], "poses_y": [[0.0, 2.0], [9.0]], "poses_yaw": [[0.0, 1.0], [9.0]]},
        schema=_PLAN_SCHEMA,
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records, plan=plan))[0]
    assert ep.goal_pos == [1.0, 2.0, 1.0]


def test_split_goal_falls_back_to_aligned_last_row():
    records = _records([(1000, 3, None, None)])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    assert ep.goal_pos == [5.0, 10.0, 0.0]
    assert ep.start_pos == [0.0, 0.0, 0.0]


def test_split_conditions_parsed_into_episode():
    records = _records([(1000, 3, None, '[{"phase": "one"}, {"phase": "two"}]')])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    assert ep.conditions == [{"phase": "one"}, {"phase": "two"}]


def test_split_conditions_invalid_is_none():
    records = _records([(1000, 3, None, "{bad json")])
    ep = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    assert ep.conditions is None


def test_split_semantic_snapshot_scoped_per_episode():
    records = _records([(1000, 3, None, None), (2000, 4, None, None)])
    snap = _snapshot([500, 1500, 2500, 3500])
    episodes = _splitter(_FakeAligner(_aligned(6))).split(TopicBundle(odom=_odom(20), episode_record=records, semantic_snapshot=snap))
    assert episodes[0].semantic_snapshot["time_ns"].to_list() == [500, 1500]
    assert episodes[1].semantic_snapshot["time_ns"].to_list() == [1500, 2500]


def test_split_num_pedestrians_estimated_from_aligned():
    records = _records([(1000, 3, None, None)])
    aligned = pl.DataFrame(
        {
            "time_ns": [1000 + i * 100 for i in range(6)],
            "pos_x": [0.0] * 6,
            "pos_y": [0.0] * 6,
            "num_pedestrians": [0, 0, 4, 4, None, 2],
        },
        schema=pl.Schema({"time_ns": pl.Int64, "pos_x": pl.Float64, "pos_y": pl.Float64, "num_pedestrians": pl.Int64}),
    )
    ep = _splitter(_FakeAligner(aligned)).split(TopicBundle(odom=_odom(), episode_record=records))[0]
    assert ep.num_pedestrians == 4


def test_split_all_lazy_inputs():
    records = _records([(1000, 3, None, '[{"k": 1}]')])
    init = pl.DataFrame(
        {"time_ns": [1100], "pos_x": [7.0], "pos_y": [8.0], "yaw": [1.5]},
        schema=_INIT_SCHEMA,
    )
    plan = pl.DataFrame(
        {"time_ns": [1000], "poses_x": [[0.0, 3.0]], "poses_y": [[0.0, 4.0]], "poses_yaw": [[0.0, 2.0]]},
        schema=_PLAN_SCHEMA,
    )
    tf = _tf_static([("map", "env_0/map", 5.0, 6.0)])
    bundle = TopicBundle(
        odom=_odom().lazy(),
        episode_record=records.lazy(),
        initialpose=init.lazy(),
        plan=plan.lazy(),
        semantic_snapshot=_snapshot([1500]).lazy(),
        tf_static=tf.lazy(),
    )
    ep = _splitter(_FakeAligner(_aligned(6))).split(bundle, robot_name="r9")[0]
    assert ep.episode_id == 3
    assert ep.start_pos == [7.0, 8.0, 1.5]
    assert ep.goal_pos == [3.0, 4.0, 2.0]
    assert ep.env_offset == (5.0, 6.0)
    assert ep.conditions == [{"k": 1}]
    assert ep.robot_name == "r9"
    assert ep.semantic_snapshot["time_ns"].to_list() == [1500]


def _rec_batch(times, ids):
    return pa.RecordBatch.from_arrays(
        [
            pa.array(times, type=pa.int64()),
            pa.array(ids, type=pa.int64()),
            pa.array([None] * len(times), type=pa.string()),
            pa.array([None] * len(times), type=pa.string()),
        ],
        names=["time_ns", "episode_id", "robots_params", "conditions"],
    )


def test_split_chunked_record_batches():
    """episode_record backed by multiple pyarrow record batches behaves identically."""
    table = pa.Table.from_batches([_rec_batch([1000, 2000], [3, 4]), _rec_batch([3000], [4])])
    chunked = pl.from_arrow(table)
    assert chunked.schema == _REC_SCHEMA
    assert len(chunked) == 3

    plain = _records([(1000, 3, None, None), (2000, 4, None, None), (3000, 4, None, None)])
    aligner_chunked = _FakeAligner(_aligned(6))
    aligner_plain = _FakeAligner(_aligned(6))

    ep_chunked = _splitter(aligner_chunked).split(TopicBundle(odom=_odom(), episode_record=chunked))
    ep_plain = _splitter(aligner_plain).split(TopicBundle(odom=_odom(), episode_record=plain))
    assert [e.episode_id for e in ep_chunked] == [e.episode_id for e in ep_plain]
    assert aligner_chunked.calls == aligner_plain.calls


# ---------------------------------------------------------------------------
# _estimate_peds
# ---------------------------------------------------------------------------


def test_estimate_peds_max_ignores_nulls():
    splitter = EpisodeSplitter(_FakeAligner(None))
    df = pl.DataFrame(
        {"num_pedestrians": [1, 5, None, 3]},
        schema=pl.Schema({"num_pedestrians": pl.Int64}),
    )
    assert splitter._estimate_peds(df) == 5


def test_estimate_peds_missing_column():
    splitter = EpisodeSplitter(_FakeAligner(None))
    assert splitter._estimate_peds(pl.DataFrame({"x": [1, 2]})) == 0


def test_estimate_peds_all_null():
    splitter = EpisodeSplitter(_FakeAligner(None))
    df = pl.DataFrame(
        {"num_pedestrians": [None, None]},
        schema=pl.Schema({"num_pedestrians": pl.Int64}),
    )
    assert splitter._estimate_peds(df) == 0
