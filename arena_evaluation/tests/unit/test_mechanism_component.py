import math

import pytest

from arena_evaluation.processing.metrics.ecological.mechanism_component import (
    MECHANISM_COMPONENTS,
    MechanismComponent,
    MechanismComponentCalculator,
    mechanism_score,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def test_component_normalisation():
    lower = MechanismComponent("x", "lower", 5.0)
    assert lower.score(0) == 1.0
    assert lower.score(5.0) == pytest.approx(0.5)
    assert lower.score(-3.0) == 1.0
    assert lower.score(None) is None
    assert lower.score(float("nan")) is None
    assert lower.score("bad") is None

    higher = MechanismComponent("y", "higher")
    assert higher.score(1.0) == 1.0
    assert higher.score(0.0) == 0.0
    assert higher.score(1.7) == 1.0


def test_component_list_includes_condition_success_and_is_well_formed():
    keys = [c.key for c in MECHANISM_COMPONENTS]
    assert "condition_success" in keys
    assert len(keys) == len(set(keys))
    assert all(c.direction in ("lower", "higher") and c.scale > 0 and c.weight > 0 for c in MECHANISM_COMPONENTS)


def test_component_keys_are_registered_metric_outputs():
    registry = MetricRegistry(RobotParams())
    outputs = {key for calc in registry.calculators.values() for key in calc.output_keys()}
    assert {c.key for c in MECHANISM_COMPONENTS} <= outputs
    directions = {}
    for calc in registry.calculators.values():
        directions.update(calc.OUTPUT_DIRECTIONS)
    for component in MECHANISM_COMPONENTS:
        if component.key in directions:
            assert directions[component.key] == component.direction, component.key


def test_mechanism_score_mean_of_available_components():
    score, used = mechanism_score({"condition_success": 0.0, "ran_red_signal": 0, "door_freeze_s": None})
    assert used == 2
    assert score == pytest.approx(0.5)

    score, used = mechanism_score({"condition_success": 1.0, "boarded_before_exit": 1})
    assert score == pytest.approx(0.75)


def test_mechanism_score_none_without_components():
    assert mechanism_score({}) == (None, 0)
    assert mechanism_score({"condition_success": None, "unrelated": 3}) == (None, 0)


def test_mechanism_score_weights():
    comps = (MechanismComponent("a", "higher", weight=3.0), MechanismComponent("b", "higher"))
    score, used = mechanism_score({"a": 1.0, "b": 0.0}, comps)
    assert used == 2 and score == pytest.approx(0.75)


def test_calculator_reads_prior_results():
    calc = MechanismComponentCalculator(RobotParams())
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    out = calc.calculate(episode, {"condition_success": 1.0, "door_oscillations": 0})
    assert out == {"mechanism_score": 1.0, "mechanism_components": 2}
    empty = calc.calculate(episode, {})
    assert empty["mechanism_score"] is None and empty["mechanism_components"] == 0
    assert not math.isnan(out["mechanism_score"])


def test_calculator_runs_after_its_inputs():
    registry = MetricRegistry(RobotParams())
    stage_of = {name: i for i, stage in enumerate(registry.execution_order()) for name in stage}
    for dep in MechanismComponentCalculator.DEPENDS_ON:
        assert stage_of[dep] < stage_of["mechanism_component"]
