"""Unit tests for the door-state timeline (acoustics/door_state.py)."""

from __future__ import annotations

import inspect

import polars as pl
import pytest

from arena_evaluation.processing.acoustics.door_state import (
    OPEN_PROGRESS_THRESHOLD,
    DoorStateTimeline,
)
from arena_evaluation.storage.exceptions import MetricInputError

# Helpers

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


# Construction guards


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


# Open determination


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


# Timeline structure


# open_doors_at (backward-asof)


# Property tests


def test_progress_threshold_constant_is_the_shared_model_default():
    from arena_evaluation.processing.acoustics import constants

    assert OPEN_PROGRESS_THRESHOLD == constants.DOOR_OPEN_PROGRESS_THRESHOLD == 0.15
    default = (
        inspect.signature(DoorStateTimeline.from_semantic_frame)
        .parameters["open_progress_threshold"]
        .default
    )
    assert default == OPEN_PROGRESS_THRESHOLD


