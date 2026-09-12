from __future__ import annotations

import numpy as np
import pytest

from arena_evaluation.experiments import blend_quality as B


def test_transition_times_fire_on_any_change() -> None:
    a = frozenset({("body", "clip", "hold")})
    b = frozenset({("body", "clip", "release")})
    states = [(0.0, frozenset(), "walk"), (0.1, frozenset(), "walk"), (0.2, a, "idle"), (0.3, a, "idle"), (0.4, b, "idle"), (0.5, b, "walk")]
    assert B.transition_times(states) == [0.2, 0.4, 0.5]


def test_link_steps_are_rescaled_to_a_50ms_tick() -> None:
    def fk(angles: dict, _body: object):
        x = angles["x"]
        return {k: np.array([x, 0.0, 0.0]) for k in B.LINKS}, {}

    samples = [(0.0, {"x": 0.0}), (0.025, {"x": 0.1}), (0.075, {"x": 0.2})]
    t, step = B.link_steps(samples, fk, None)
    assert t.tolist() == pytest.approx([0.025, 0.075])
    assert step.tolist() == pytest.approx([0.2, 0.1])  # 0.1 m in 25 ms is 0.2 m per tick


def test_summary_counts_gate_violations() -> None:
    res = [{"events": 4, "events_over_gate": 1, "trans": np.array([0.1, 0.3]), "steady": np.array([0.01, 0.02])}]
    s = B.summarize(res)
    assert s["transitions"] == 4 and s["transition_pass_rate"] == pytest.approx(0.75)
    assert s["transition_ticks"]["over_gate"] == 1 and s["steady_ticks"]["over_gate"] == 0
