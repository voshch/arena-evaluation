from __future__ import annotations

import dataclasses
import math
import typing

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


@dataclasses.dataclass(frozen=True)
class MechanismComponent:
    """One input of the `mechanism` component: key, direction ("lower" or "higher"), scale, weight."""

    key: str
    direction: typing.Literal["lower", "higher"]
    scale: float = 1.0
    weight: float = 1.0

    def score(self, value: typing.SupportsFloat | str | None) -> float | None:
        """Normalised [0, 1] score (1 = best), None for a missing or non-finite value."""
        if value is None:
            return None
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        if self.direction == "higher":
            return min(max(x, 0.0), 1.0)
        return 1.0 / (1.0 + max(x, 0.0) / self.scale)


MECHANISM_COMPONENTS: tuple[MechanismComponent, ...] = (
    MechanismComponent("condition_success", "higher"),
    MechanismComponent("boarded_before_exit", "lower", 1.0),
    MechanismComponent("blocked_closing_door_s", "lower", 5.0),
    MechanismComponent("ped_mechanism_delay_seconds", "lower", 30.0),
    MechanismComponent("elevator_wait_to_board_s", "lower", 30.0),
    MechanismComponent("elevator_missed_cycles", "lower", 1.0),
    MechanismComponent("elevator_unnecessary_calls", "lower", 1.0),
    MechanismComponent("entered_full_cabin", "lower", 1.0),
    MechanismComponent("cabin_door_blocking_s", "lower", 5.0),
    MechanismComponent("door_yield_violations", "lower", 1.0),
    MechanismComponent("elevator_queue_jumps", "lower", 1.0),
    MechanismComponent("alarm_egress_blocking_s", "lower", 10.0),
    MechanismComponent("door_freeze_s", "lower", 10.0),
    MechanismComponent("door_oscillations", "lower", 1.0),
    MechanismComponent("ran_red_signal", "lower", 1.0),
    MechanismComponent("used_elevator_during_alarm_robot", "lower", 1.0),
    MechanismComponent("entered_over_cap_zone", "lower", 1.0),
    MechanismComponent("time_waiting_at_doors_robot", "lower", 30.0),
)


def mechanism_score(results: dict[str, typing.Any], components: typing.Sequence[MechanismComponent] = MECHANISM_COMPONENTS) -> tuple[float | None, int]:
    """Weighted mean of the available component scores and how many were available."""
    total = 0.0
    weight = 0.0
    used = 0
    for component in components:
        score = component.score(results.get(component.key))
        if score is None:
            continue
        total += component.weight * score
        weight += component.weight
        used += 1
    if weight <= 0.0:
        return None, 0
    return total / weight, used


class MechanismComponentCalculator(BaseMetricCalculator):
    """`mechanism` component: weighted mean of the normalised scores of every available `MECHANISM_COMPONENTS` key."""

    NAME = "mechanism_component"
    CATEGORY = "ecological"
    DEPENDS_ON = ["condition_compliance", "mechanism_metrics", "regime_metrics", "semantic_interaction_metrics"]

    UNITS = {
        "mechanism_score": "",
        "mechanism_components": "",
    }

    PRIMARY_OUTPUTS = ["mechanism_score"]
    OUTPUT_DIRECTIONS = {"mechanism_score": "higher"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["mechanism_score", "mechanism_components"]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del episode
        score, used = mechanism_score(prior_results)
        return {"mechanism_score": score, "mechanism_components": used}
