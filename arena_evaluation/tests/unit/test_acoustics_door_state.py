"""Unit tests for the door-state timeline (acoustics/door_state.py).

DoorStateTimeline turns the long-format SemanticSnapshot table into a sorted
(time, open-doors) series queried with backward-asof binary search.

A door is open while its continuous 'progress' field (0 closed, 1 open) is above
OPEN_PROGRESS_THRESHOLD. Opening and closing doors follow the same rule, so a closing
door counts as open until it is closed more than 1 - OPEN_PROGRESS_THRESHOLD.
Discrete state labels do not decide openness. A door reading without a progress row
is an error, not a closed door.
"""

from __future__ import annotations

import bisect

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

from arena_evaluation.processing.acoustics.door_state import (
    OPEN_PROGRESS_THRESHOLD,
    DoorStateTimeline,
)
from arena_evaluation.storage.exceptions import MetricInputError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEM_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "world": pl.String,
        "entity_name": pl.String,
        "kind": pl.String,
        "field_name": pl.String,
        "field_kind": pl.String,
        "value_str": pl.String,
        "value_num": pl.Float64,
        "value_bool": pl.Boolean,
    }
)


def _row(
    t_ns: int,
    entity: str,
    field: str,
    kind: str = "door",
    vstr: str | None = None,
    vnum: float | None = None,
    vbool: bool | None = None,
) -> dict:
    return {
        "t_ns": t_ns,
        "env_id": "env_0",
        "world": "w",
        "entity_name": entity,
        "kind": kind,
        "field_name": field,
        "field_kind": "",
        "value_str": vstr,
        "value_num": vnum,
        "value_bool": vbool,
    }


def _progress(t_ns: int, entity: str, value: float | None) -> dict:
    return _row(t_ns, entity, "progress", vnum=value)


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=_SEM_SCHEMA)


def _frame_strict_schema(rows: list[dict]) -> pl.DataFrame:
    """Assertion helper: the constructed frame must match the declared schema."""
    df = _frame(rows)
    assert df.schema == _SEM_SCHEMA
    return df


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_from_semantic_frame_none():
    assert DoorStateTimeline.from_semantic_frame(None) is None


def test_from_semantic_frame_empty():
    assert DoorStateTimeline.from_semantic_frame(_frame([])) is None


def test_from_semantic_frame_missing_kind_column():
    df = pl.DataFrame(
        {"t_ns": [1], "entity": ["e"]},
        schema=pl.Schema({"t_ns": pl.Int64, "entity": pl.String}),
    )
    with pytest.raises(MetricInputError, match="lacks columns"):
        DoorStateTimeline.from_semantic_frame(df)


def test_from_semantic_frame_no_door_rows():
    df = _frame_strict_schema([_row(100, "r1", "progress", kind="robot", vnum=1.0)])
    assert DoorStateTimeline.from_semantic_frame(df) is None


def test_from_semantic_frame_lazy_input_collected():
    df = _frame_strict_schema([_progress(100, "env_0/d1", 1.0)])
    tl = DoorStateTimeline.from_semantic_frame(df.lazy())
    assert tl is not None
    assert tl.times_ns.tolist() == [100]
    assert tl.open_sets == [frozenset({"env_0/d1"})]


# ---------------------------------------------------------------------------
# Open determination
# ---------------------------------------------------------------------------


def test_open_above_threshold():
    rows = [
        _progress(100, "d1", 0.9),
        _progress(100, "d2", 0.15),  # exactly at threshold -> closed
        _progress(100, "d3", 0.16),
        _progress(100, "d4", 0.1),
        _progress(100, "d5", None),
    ]
    tl = DoorStateTimeline.from_semantic_frame(_frame_strict_schema(rows))
    assert tl.open_sets[0] == frozenset({"d1", "d3"})


def test_closing_door_counts_open_until_closed_past_threshold():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 0.3),
            _row(100, "d1", "state", vstr="closing"),
            _progress(100, "d2", 0.1),
            _row(100, "d2", "state", vstr="closing"),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset({"d1"})


def test_opening_door_counts_open_past_threshold():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 0.2),
            _row(100, "d1", "state", vstr="opening"),
            _progress(100, "d2", 0.05),
            _row(100, "d2", "state", vstr="opening"),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset({"d1"})


def test_state_labels_do_not_open_a_door():
    df = _frame_strict_schema(
        [_progress(100, "d1", 0.0), _row(100, "d1", "state", vstr="open")]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset()


def test_progress_nan_is_closed():
    df = _frame_strict_schema([_progress(100, "d1", float("nan"))])
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset()


def test_door_without_progress_rows_is_an_error():
    df = _frame_strict_schema([_row(100, "d1", "state", vstr="open")])
    with pytest.raises(MetricInputError, match="progress"):
        DoorStateTimeline.from_semantic_frame(df)


def test_door_without_progress_at_one_stamp_is_an_error():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 1.0),
            _progress(100, "d2", 1.0),
            _progress(200, "d1", 1.0),
            _row(200, "d2", "state", vstr="open"),
        ]
    )
    with pytest.raises(MetricInputError, match="progress"):
        DoorStateTimeline.from_semantic_frame(df)


def test_non_door_kind_ignored():
    df = _frame_strict_schema(
        [
            _progress(100, "door", 1.0),
            _row(100, "robot", "progress", kind="robot", vnum=1.0),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset({"door"})


def test_entity_names_preserved_verbatim():
    df = _frame_strict_schema([_progress(100, "env_0/main_hallway/3", 1.0)])
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets[0] == frozenset({"env_0/main_hallway/3"})


# ---------------------------------------------------------------------------
# Timeline structure
# ---------------------------------------------------------------------------


def test_multiple_timestamps_sorted():
    df = _frame_strict_schema(
        [
            _progress(300, "d1", 1.0),
            _progress(100, "d2", 1.0),
            _progress(200, "d3", 1.0),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.times_ns.dtype == np.int64
    assert tl.times_ns.tolist() == [100, 200, 300]
    assert tl.open_sets == [frozenset({"d2"}), frozenset({"d3"}), frozenset({"d1"})]


def test_state_changes_over_time():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 1.0),
            _progress(200, "d1", 0.0),
            _progress(300, "d1", 1.0),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_sets == [frozenset({"d1"}), frozenset(), frozenset({"d1"})]


def test_duplicate_stamps_merged_into_one_frame():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 1.0),
            _progress(100, "d2", 1.0),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.times_ns.tolist() == [100]
    assert tl.open_sets == [frozenset({"d1", "d2"})]


# ---------------------------------------------------------------------------
# open_doors_at (backward-asof)
# ---------------------------------------------------------------------------


def test_open_doors_at_before_first_is_empty():
    tl = DoorStateTimeline(
        np.array([100, 200], dtype=np.int64), [frozenset({"d1"}), frozenset()]
    )
    assert tl.open_doors_at(50) == frozenset()
    assert tl.open_doors_at(99) == frozenset()


def test_open_doors_at_exact_stamp():
    tl = DoorStateTimeline(
        np.array([100, 200], dtype=np.int64), [frozenset({"d1"}), frozenset({"d2"})]
    )
    assert tl.open_doors_at(100) == frozenset({"d1"})
    assert tl.open_doors_at(200) == frozenset({"d2"})


def test_open_doors_at_backward_between_stamps():
    tl = DoorStateTimeline(
        np.array([100, 200], dtype=np.int64), [frozenset({"d1"}), frozenset({"d2"})]
    )
    assert tl.open_doors_at(150) == frozenset({"d1"})
    assert tl.open_doors_at(199) == frozenset({"d1"})


def test_open_doors_at_after_last_stamp():
    tl = DoorStateTimeline(
        np.array([100, 200], dtype=np.int64), [frozenset({"d1"}), frozenset()]
    )
    assert tl.open_doors_at(10_000) == frozenset()


def test_open_doors_at_returns_fresh_frozenset():
    tl = DoorStateTimeline(np.array([100], dtype=np.int64), [frozenset({"d1"})])
    s1 = tl.open_doors_at(100)
    assert s1 == frozenset({"d1"})
    assert isinstance(s1, frozenset)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(
    timestamps=st.lists(
        st.integers(min_value=0, max_value=100_000), min_size=1, max_size=8, unique=True
    ),
    door_names=st.lists(
        st.text(min_size=1, max_size=6), min_size=1, max_size=4, unique=True
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_from_semantic_frame_consistent_with_input(timestamps, door_names, seed):
    rng = np.random.default_rng(seed)
    rows = []
    expected: dict[int, frozenset] = {}
    for ts in timestamps:
        open_now = set()
        for d in door_names:
            is_open = bool(rng.integers(0, 2))
            rows.append(_progress(ts, d, 1.0 if is_open else 0.0))
            if is_open:
                open_now.add(d)
        expected[ts] = frozenset(open_now)

    tl = DoorStateTimeline.from_semantic_frame(_frame_strict_schema(rows))
    assert tl is not None
    assert tl.times_ns.dtype == np.int64
    assert (tl.times_ns[1:] >= tl.times_ns[:-1]).all()
    for ts in timestamps:
        assert tl.open_doors_at(ts) == expected[ts]


@given(
    times=st.lists(
        st.integers(min_value=0, max_value=1000), min_size=1, max_size=12, unique=True
    ),
    door_names=st.lists(
        st.text(min_size=1, max_size=4), min_size=0, max_size=3, unique=True
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_open_doors_at_asof_semantics(times, door_names, seed):
    rng = np.random.default_rng(seed)
    sorted_times = np.sort(np.asarray(times, dtype=np.int64))
    open_sets = []
    for _ in sorted_times:
        if not door_names:
            open_sets.append(frozenset())
        else:
            k = int(rng.integers(0, len(door_names) + 1))
            chosen = rng.choice(door_names, size=k, replace=False)
            open_sets.append(frozenset(chosen))

    tl = DoorStateTimeline(sorted_times, open_sets)
    # probe a dense set of query times spanning everything
    probes = np.arange(-50, 1100, 37, dtype=np.int64)
    for t in probes:
        idx = bisect.bisect_right(sorted_times, t) - 1
        if idx < 0:
            assert tl.open_doors_at(t) == frozenset()
        else:
            assert tl.open_doors_at(t) == open_sets[idx]


def test_progress_is_the_only_open_measure():
    df = _frame_strict_schema(
        [
            _progress(100, "d1", 0.9),
            _progress(100, "d2", 0.6),
            _row(100, "d2", "state", vstr="closing"),
            _progress(100, "d3", 0.15),
            _progress(100, "d4", 0.0),
            _row(100, "d4", "state", vstr="open"),
        ]
    )
    tl = DoorStateTimeline.from_semantic_frame(df)
    assert tl.open_doors_at(100) == frozenset({"d1", "d2"})
    assert tl.open_doors_at(99) == frozenset()


def test_progress_threshold_constant():
    assert OPEN_PROGRESS_THRESHOLD == 0.15
