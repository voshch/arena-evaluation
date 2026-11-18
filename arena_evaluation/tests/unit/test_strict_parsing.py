from __future__ import annotations

import types

import polars as pl
import pytest

from arena_evaluation.processing.metrics.common.parsing import parse_bool, parse_float
from arena_evaluation.processing.metrics.performance.barn_metrics import (
    _first_collision_s,
)
from arena_evaluation.processing.metrics.performance.sidewalk_metrics import _collisions
from arena_evaluation.storage.exceptions import ContractViolationError, MetricInputError


def _episode_without_event_count() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        topics={"collision_events": pl.DataFrame({"t_ns": [0]})}
    )


def test_parse_float_rejects_malformed_tokens() -> None:
    with pytest.raises(ContractViolationError):
        parse_float("abc")


def test_parse_bool_rejects_an_unknown_token() -> None:
    with pytest.raises(ContractViolationError):
        parse_bool("maybe")


def test_parse_bool_accepts_true_and_zero_case_insensitively() -> None:
    assert parse_bool("TRUE") is True
    assert parse_bool("0") is False


def test_first_collision_s_rejects_a_collision_frame_without_event_count() -> None:
    with pytest.raises(MetricInputError, match="event_count"):
        _first_collision_s(_episode_without_event_count())


def test_collisions_rejects_a_collision_frame_without_event_count() -> None:
    with pytest.raises(MetricInputError, match="event_count"):
        _collisions(_episode_without_event_count())
