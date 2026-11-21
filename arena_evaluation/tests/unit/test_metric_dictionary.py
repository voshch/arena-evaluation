"""The metric dictionary: one row per metric and scope of the shared store, published as the dimension ``dim_metrics``."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator, Direction
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.processing.metrics.registry import (
    scope_producers,
    shipped_calculators,
)
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.registry import (
    MetricScope,
    defined_metrics,
)
from arena_evaluation.storage.registry.table import Metric


def _planted_metric(name: str) -> Metric:
    """A metric built outside the shared store, so no other test sees it."""
    return Metric(
        pl.Float64,
        unit="m",
        description=f"planted metric {name}",
        name=name,
        scopes=(MetricScope.EPISODE, MetricScope.ROBOT),
    )


class _HigherAtEpisode(BaseMetricCalculator):
    NAME = "planted_higher_at_episode"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.EPISODE,)}
    DIRECTIONS = {"planted_distance_m": Direction.HIGHER}

    def calculate(self, episode, prior_results):
        return {}


class _HigherAtRobot(BaseMetricCalculator):
    NAME = "planted_higher_at_robot"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.ROBOT,)}
    DIRECTIONS = {"planted_distance_m": Direction.HIGHER}

    def calculate(self, episode, prior_results):
        return {}


class _LowerAtRobot(BaseMetricCalculator):
    NAME = "planted_lower_at_robot"
    OUTPUTS = (_planted_metric("planted_distance_m"),)
    OUTPUT_SCOPES = {"planted_distance_m": (MetricScope.ROBOT,)}
    DIRECTIONS = {"planted_distance_m": Direction.LOWER}

    def calculate(self, episode, prior_results):
        return {}


def _store_pairs() -> set[tuple[str, str]]:
    return {(metric.name, scope.value) for metric in defined_metrics() for scope in metric.scopes}


def test_two_calculators_giving_one_metric_different_directions_is_an_error() -> None:
    with pytest.raises(SchemaViolationError, match="planted_distance_m.*two directions"):
        metric_dictionary((*shipped_calculators(), _HigherAtEpisode, _LowerAtRobot))


def test_a_stored_metric_that_no_calculator_publishes_is_an_error() -> None:
    with pytest.raises(SchemaViolationError, match="no calculator publishes"):
        metric_dictionary(())


def test_two_calculators_publishing_one_metric_at_one_scope_is_an_error() -> None:
    with pytest.raises(SchemaViolationError, match="produced by both"):
        scope_producers((_HigherAtRobot, _LowerAtRobot))
