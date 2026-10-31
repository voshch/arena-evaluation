"""Unit tests for arena_evaluation.storage.exceptions.

Verifies the domain exception hierarchy: every concrete exception derives
from ArenaEvaluationError, stays catchable as a plain Exception, and
preserves constructor args.
"""

from __future__ import annotations

import pytest

from arena_evaluation.storage.exceptions import (
    ArenaEvaluationError,
    CircularDependencyError,
    ManifestGenerationError,
    MetricCalculationError,
    RobotNotFoundError,
    SchemaViolationError,
)

_ALL_EXCEPTIONS = [
    ArenaEvaluationError,
    MetricCalculationError,
    CircularDependencyError,
    SchemaViolationError,
    RobotNotFoundError,
    ManifestGenerationError,
]


@pytest.mark.parametrize("exc_cls", _ALL_EXCEPTIONS)
def test_exception_is_raiseable_with_message(exc_cls):
    exc = exc_cls("boom")
    assert str(exc) == "boom"
    assert isinstance(exc, Exception)


def test_all_exceptions_subclass_base():
    for exc_cls in _ALL_EXCEPTIONS[1:]:
        assert issubclass(exc_cls, ArenaEvaluationError)
        assert issubclass(exc_cls, Exception)


def test_all_exceptions_caught_by_base():
    for exc_cls in _ALL_EXCEPTIONS[1:]:
        with pytest.raises(ArenaEvaluationError):
            raise exc_cls("oops")


def test_base_exception_caught_by_base():
    with pytest.raises(ArenaEvaluationError):
        raise ArenaEvaluationError("base")


def test_exception_args_preserved():
    exc = ManifestGenerationError("first", "second")
    assert exc.args == ("first", "second")


def test_exception_chaining_preserved():
    cause = ValueError("root cause")
    with pytest.raises(SchemaViolationError) as exc_info:
        try:
            raise cause
        except ValueError as e:
            raise SchemaViolationError("schema mismatch") from e
    assert exc_info.value.__cause__ is cause
