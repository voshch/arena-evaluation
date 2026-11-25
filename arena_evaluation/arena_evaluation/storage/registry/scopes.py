"""Evaluation scopes and the fact tables that publish them."""

from __future__ import annotations

from enum import StrEnum


class MetricScope(StrEnum):
    """The five first-class evaluation scopes, each published as one fact table."""

    EPISODE = "episodes"
    ENCOUNTER = "encounters"
    CONTACT = "contacts"
    PEDESTRIAN = "pedestrians"
    ROBOT = "robots"


class FactTable(StrEnum):
    """The twelve fact tables: five scoped tables and seven child tables."""

    EPISODES = "episodes"
    ENCOUNTERS = "encounters"
    CONTACTS = "contacts"
    PEDESTRIANS = "pedestrians"
    ROBOTS = "robots"
    EDGE_CASES = "edge_cases"
    FORMATIONS = "formations"
    PHASE_OUTCOMES = "phase_outcomes"
    TASK_OUTCOMES = "task_outcomes"
    JUDGE_DISAGREEMENTS = "judge_disagreements"
    MECHANISM_STATES = "mechanism_states"
    COLLISIONS = "collisions"
