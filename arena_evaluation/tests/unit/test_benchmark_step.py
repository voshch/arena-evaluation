"""Tests for the benchmark grid model: Step keys, StepErrorKind, StepResult.

Covers :mod:`arena_evaluation.benchmark.step`. Contestants and stages are
represented by lightweight stand-ins (only ``.name`` is ever read).
"""

from __future__ import annotations

import collections

import attrs
import pytest

from arena_evaluation.benchmark.step import Step, StepErrorKind, StepResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Named tuple stand-ins: hashable and value-equal, unlike SimpleNamespace.
_Name = collections.namedtuple("_Name", ["name"])


def _contestant(name: str) -> _Name:
    return _Name(name)


def _stage(name: str) -> _Name:
    return _Name(name)


# ---------------------------------------------------------------------------
# Step.key
# ---------------------------------------------------------------------------


def test_step_key_plain():
    step = Step(contestant=_contestant("planner_a"), stage=_stage("stage_one"))
    assert step.key == "planner_a/stage_one"


def test_step_key_uses_names_not_objects():
    step = Step(contestant=_contestant("teb"), stage=_stage("indoor_10"), episodes=5)
    assert step.key == "teb/indoor_10"


def test_step_key_reference_type_matches_name_suffix():
    """Reference whose type is a suffix of the contestant name keeps the plain key."""
    step = Step(
        contestant=_contestant("dwb"),
        stage=_stage("s1"),
        is_reference=True,
        reference_type="dwb",
    )
    assert step.key == "dwb/s1"


def test_step_key_reference_type_not_suffix():
    step = Step(
        contestant=_contestant("dwb"),
        stage=_stage("s1"),
        is_reference=True,
        reference_type="unobstructed_robot",
    )
    assert step.key == "dwb_unobstructed_robot/s1"


def test_step_key_reference_without_type_falls_back_to_plain():
    step = Step(
        contestant=_contestant("peds"),
        stage=_stage("s1"),
        is_reference=True,
        reference_type=None,
    )
    assert step.key == "peds/s1"


def test_step_key_type_without_reference_flag_ignored():
    step = Step(
        contestant=_contestant("teb"),
        stage=_stage("s1"),
        is_reference=False,
        reference_type="unobstructed_robot",
    )
    assert step.key == "teb/s1"


# ---------------------------------------------------------------------------
# Step defaults / immutability
# ---------------------------------------------------------------------------


def test_step_defaults():
    step = Step(contestant=_contestant("c"), stage=_stage("s"))
    assert step.episodes == 1
    assert step.record_dir is None
    assert step.is_reference is False
    assert step.reference_type is None


def test_step_is_frozen():
    step = Step(contestant=_contestant("c"), stage=_stage("s"))
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        step.episodes = 10


def test_step_equality_and_hash():
    a = Step(contestant=_contestant("c"), stage=_stage("s"), episodes=3)
    b = Step(contestant=_contestant("c"), stage=_stage("s"), episodes=3)
    c = Step(contestant=_contestant("c"), stage=_stage("s"), episodes=4)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


# ---------------------------------------------------------------------------
# StepErrorKind
# ---------------------------------------------------------------------------


def test_step_error_kind_members_and_values():
    assert {k.value for k in StepErrorKind} == {
        "env_setup",
        "robot_setup",
        "episode_timeout",
        "sim_dead",
        "cancelled",
        "internal",
    }


def test_step_error_kind_str_enum():
    assert StepErrorKind.ENV_SETUP == "env_setup"
    assert str(StepErrorKind.CANCELLED) == "cancelled"


def test_step_error_kind_from_value():
    assert StepErrorKind("internal") is StepErrorKind.INTERNAL


def test_step_error_kind_invalid_raises():
    with pytest.raises(ValueError):
        StepErrorKind("not_a_kind")


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


def test_step_result_defaults():
    r = StepResult(
        key="c/s",
        status="in_progress",
        env_id=None,
        started_at=0.0,
        ended_at=None,
        error_kind=None,
        error_detail=None,
    )
    assert r.episodes_run == 0
    assert r.episodes_failed == 0
    assert r.episodes_total == 0


def test_step_result_fields():
    r = StepResult(
        key="c/s",
        status="failed",
        env_id=2,
        started_at=1.0,
        ended_at=5.0,
        error_kind=StepErrorKind.ROBOT_SETUP,
        error_detail="nav failed",
        episodes_run=3,
        episodes_failed=1,
        episodes_total=5,
    )
    assert r.status == "failed"
    assert r.env_id == 2
    assert r.error_kind is StepErrorKind.ROBOT_SETUP
    assert r.episodes_total == 5


def test_step_result_is_frozen():
    r = StepResult(
        key="c/s",
        status="ok",
        env_id=0,
        started_at=0.0,
        ended_at=1.0,
        error_kind=None,
        error_detail=None,
    )
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        r.status = "failed"


def test_step_result_equality():
    kwargs = dict(
        key="c/s",
        status="ok",
        env_id=0,
        started_at=0.0,
        ended_at=1.0,
        error_kind=None,
        error_detail=None,
    )
    assert StepResult(**kwargs) == StepResult(**kwargs)
