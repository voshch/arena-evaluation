"""Door and elevator state sampled on the episode time grid, the trajectory plot's animation input."""

from __future__ import annotations

import typing

import numpy as np

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import SemanticView, entity_level, entity_name
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

PROGRESS_FIELD = {"door": "progress", "elevator": "cabin_door_progress"}


def _floats(values: list[str]) -> list[float]:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except ValueError:
            out.append(float("nan"))
    return out


class MechanismStatesCalculator(BaseMetricCalculator):
    """Per door and elevator: opening progress and lock state at every episode sample.

    This is not a scoped metric: `mechanism_states` is a non-scope fact table (change-point rows)
    built by the evaluator from this episode-level series, so the calculator declares no OUTPUTS
    and the registry does not run it in any scope. The evaluator calls it directly.
    """

    NAME = "mechanism_states"
    CATEGORY = "ecological"
    REQUIRED_TOPICS = [("robot_tracks", "robot_odom")]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        frame = episode.data
        if frame is None or "t_ns" not in frame.columns or len(frame) == 0:
            return {"mechanism_states": None}
        view = SemanticView.from_episode(episode)
        query = frame["t_ns"].to_numpy().astype(np.int64)
        states = []
        for kind, field in PROGRESS_FIELD.items():
            for entity in view.entities(kind):
                progress = view.get(entity, field)
                if progress is None:
                    continue
                locked = view.get(entity, "locked")
                states.append(
                    {
                        "entity": entity_name(entity),
                        "level": entity_level(entity),
                        "kind": kind,
                        "progress": _floats(progress.sample(query)),
                        "locked": [v == "true" for v in locked.sample(query)] if locked is not None else [None] * len(query),
                    }
                )
        return {"mechanism_states": states or None}
