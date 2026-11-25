import math

import pytest

from arena_evaluation.processing.metrics.ecological.mechanism_component import (
    MECHANISM_COMPONENTS,
    MechanismComponent,
    MechanismComponentCalculator,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params


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


def test_calculator_reads_prior_results():
    calc = MechanismComponentCalculator(make_robot_params())
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    out = calc.calculate(episode, {"condition_success": 1.0, "door_oscillations": 0})
    assert out == {"mechanism_score": 1.0, "mechanism_components": 2}
    empty = calc.calculate(episode, {})
    assert empty["mechanism_score"] is None and empty["mechanism_components"] == 0
    assert not math.isnan(out["mechanism_score"])


def test_calculator_runs_after_its_inputs():
    registry = MetricRegistry(make_robot_params())
    stage_of = {name: i for i, stage in enumerate(registry.get_execution_order_for_scope(MetricScope.ROBOT)) for name in stage}
    for dep in MechanismComponentCalculator.DEPENDS_ON:
        assert stage_of[dep] < stage_of["mechanism_component"]
