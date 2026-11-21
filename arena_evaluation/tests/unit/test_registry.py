from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.exceptions import (
    CircularDependencyError,
    SchemaViolationError,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.registry.table import Metric
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params


def _output(name: str) -> Metric:
    """A test metric built outside the shared store, so no other test sees it."""
    return Metric(
        pl.Int64,
        description=f"test output {name}",
        name=name,
        scopes=(MetricScope.EPISODE,),
    )


def _episode() -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[]
    )


class Calc1(BaseMetricCalculator):
    NAME = "calc1"
    OUTPUTS = (_output("o1"),)
    DEPENDS_ON = []

    def calculate(self, ep, prior):
        return {"o1": 1}


class Calc2(BaseMetricCalculator):
    NAME = "calc2"
    OUTPUTS = (_output("o2"),)
    DEPENDS_ON = ["calc1"]

    def calculate(self, ep, prior):
        return {"o2": prior["o1"] + 1}


class Calc3(BaseMetricCalculator):
    NAME = "calc3"
    OUTPUTS = (_output("o3"),)
    DEPENDS_ON = ["calc2"]

    def calculate(self, ep, prior):
        return {"o3": prior["o2"] + 1}


class CycleA(BaseMetricCalculator):
    NAME = "cycle_a"
    OUTPUTS = (_output("cycle_a_out"),)
    DEPENDS_ON = ["cycle_b"]

    def calculate(self, ep, prior):
        return {"cycle_a_out": 1}


class CycleB(BaseMetricCalculator):
    NAME = "cycle_b"
    OUTPUTS = (_output("cycle_b_out"),)
    DEPENDS_ON = ["cycle_a"]

    def calculate(self, ep, prior):
        return {"cycle_b_out": 1}


class CustomNullReasonCalc(BaseMetricCalculator):
    NAME = "custom_null_reason"
    OUTPUTS = (_output("custom_null"),)

    def calculate(self, ep, prior):
        return {"custom_null": None, "custom_null_null_reason": "custom_reason"}


class LeakyCalc(BaseMetricCalculator):
    NAME = "leaky"
    OUTPUTS = (_output("leaky_out"),)

    def calculate(self, ep, prior):
        return {"leaky_out": 1, "leaky_extra": 2}


class SilentCalc(BaseMetricCalculator):
    NAME = "silent"
    OUTPUTS = (_output("silent_out"),)

    def calculate(self, ep, prior):
        return {}


def test_registry_orders_calculators_by_their_dependencies() -> None:
    registry = MetricRegistry(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0),
        calculators=(Calc3, Calc1, Calc2),
    )

    assert registry.execution_order() == [["calc1"], ["calc2"], ["calc3"]]


def test_registry_refuses_a_circular_dependency() -> None:
    with pytest.raises(CircularDependencyError):
        MetricRegistry(
            make_robot_params(robot_radius=0.25), calculators=(CycleA, CycleB)
        )


def test_registry_refuses_a_produced_key_without_a_declaration() -> None:
    registry = MetricRegistry(
        make_robot_params(robot_radius=0.25), calculators=(LeakyCalc,)
    )

    with pytest.raises(SchemaViolationError, match="leaky_extra"):
        registry.run_scope(
            MetricScope.EPISODE,
            _episode(),
            available_topics=set(),
            pedsim_available=True,
        )


def test_registry_refuses_a_declaration_without_a_producer() -> None:
    registry = MetricRegistry(
        make_robot_params(robot_radius=0.25), calculators=(SilentCalc,)
    )

    with pytest.raises(SchemaViolationError, match="silent_out"):
        registry.run_scope(
            MetricScope.EPISODE,
            _episode(),
            available_topics=set(),
            pedsim_available=True,
        )


class _AltTopicCalc(BaseMetricCalculator):
    NAME = "alt_topic"
    OUTPUTS = (_output("alt_out"),)
    REQUIRED_TOPICS = [("topic_a", "topic_b")]

    def calculate(self, ep, prior):
        return {"alt_out": 7}


class _PedsimCalc(BaseMetricCalculator):
    NAME = "pedsim_calc"
    OUTPUTS = (_output("pedsim_out"),)
    REQUIRES_PEDSIM = True

    def calculate(self, ep, prior):
        return {"pedsim_out": 1}


class _CrashingCalc(BaseMetricCalculator):
    NAME = "crashing"
    OUTPUTS = (_output("crash_out"),)

    def calculate(self, ep, prior):
        raise RuntimeError("boom")


def test_registry_run_scope_pins_topic_alternatives_pedsim_callback_and_exception() -> (
    None
):
    registry = MetricRegistry(
        make_robot_params(robot_radius=0.25),
        calculators=(_AltTopicCalc, _PedsimCalc, _CrashingCalc),
    )
    empty_topic = pl.DataFrame({"t_ns": pl.Series([], dtype=pl.Int64)})
    ep_empty = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=[],
        topics={"topic_a": empty_topic, "topic_b": empty_topic},
    )
    ep_full = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=[],
        topics={"topic_a": pl.DataFrame({"t_ns": [1]})},
    )

    seen: list[tuple[str, int, int]] = []

    def _cb(name: str, idx: int, total: int) -> None:
        seen.append((name, idx, total))
        if name == "crashing":
            raise ValueError("callback error")

    outcome_empty = registry.run_scope(
        MetricScope.EPISODE,
        ep_empty,
        available_topics=None,
        pedsim_available=False,
        progress_callback=_cb,
    )
    assert outcome_empty.results["alt_out"] is None
    assert outcome_empty.provenance["alt_out"] == "alt_out:no_samples"
    assert outcome_empty.results["pedsim_out"] is None
    assert outcome_empty.provenance["pedsim_out"] == "pedsim_out:pedsim_unavailable"
    assert outcome_empty.results["crash_out"] is None
    assert outcome_empty.provenance["crash_out"] == "crash_out:exception:RuntimeError"
    assert seen == [("crashing", 3, 3)]

    outcome_full = registry.run_scope(
        MetricScope.EPISODE, ep_full, available_topics=None, pedsim_available=True
    )
    assert outcome_full.results["alt_out"] == 7
