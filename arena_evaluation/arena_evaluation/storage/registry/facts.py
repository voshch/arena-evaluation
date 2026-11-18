"""The eleven fact tables of one evaluated episode: identity and recorded columns; metric columns come from the metric layer."""

from __future__ import annotations

from typing import Final

import polars as pl

from arena_evaluation.storage.registry.scopes import FactTable, MetricScope
from arena_evaluation.storage.registry.table import Column, Table, TableKind

_NULL_REASON: Final[Column] = Column(
    pl.String,
    required=False,
    description="Semicolon-separated <column>:<code> reasons of the NULL metric cells of this row, empty when there are none.",
)


class _FactRow(Table, abstract=True):
    """The base of every fact table: its columns are non-null unless a column says otherwise."""

    required_by_default = True


class _RunEpisode(_FactRow, abstract=True):
    """The run and episode identity of a fact row."""

    run_id = Column(pl.String, description="Benchmark run the row belongs to.")
    episode_id = Column(pl.Int64, description="Episode number within the run.")


class _RunEpisodeRobot(_RunEpisode, abstract=True):
    """The identity of a fact row about one robot of one episode."""

    robot_id = Column(pl.String, description="Robot of the episode the row describes.")


class _RunEpisodeEnvironment(_RunEpisode, abstract=True):
    """The identity of a fact row about one simulation environment of one episode."""

    env_id = Column(pl.String, description="Simulation environment the row describes.")


class FactEpisodes(
    _RunEpisodeRobot, name="episodes", kind=TableKind.FACT, scope=MetricScope.EPISODE
):
    """Per-episode facts: one row per episode and robot."""

    key = ("run_id", "episode_id", "robot_id")

    result = Column(
        pl.String,
        null_reason=True,
        description="Terminal result of the episode as the runner recorded it.",
    )
    null_reason = _NULL_REASON


class FactEncounters(
    _RunEpisodeRobot,
    name="encounters",
    kind=TableKind.FACT,
    scope=MetricScope.ENCOUNTER,
):
    """Per-encounter facts: one row per encounter window of one robot."""

    key = ("run_id", "episode_id", "robot_id", "encounter_id")

    encounter_id = Column(
        pl.String, description="Identity of the encounter within the episode."
    )
    actor_1 = Column(pl.String, description="First party of the encounter, the robot.")
    actor_2 = Column(
        pl.String,
        description="Second party of the encounter, a pedestrian or another robot.",
    )
    actor_2_kind = Column(pl.String, description="Kind of the second party.")
    t_start_ns = Column(
        pl.Int64, description="Start of the encounter window in nanoseconds."
    )
    t_end_ns = Column(
        pl.Int64, description="End of the encounter window in nanoseconds."
    )
    contact_id = Column(
        pl.String, description="Contact of the encounter, empty when there is none."
    )
    null_reason = _NULL_REASON


class FactContacts(
    _RunEpisodeRobot, name="contacts", kind=TableKind.FACT, scope=MetricScope.CONTACT
):
    """Per-contact facts: one row per contact of one robot."""

    key = ("run_id", "episode_id", "robot_id", "contact_id")

    contact_id = Column(
        pl.String, description="Identity of the contact within the episode."
    )
    encounter_id = Column(pl.String, description="Encounter the contact belongs to.")
    collider_kind = Column(
        pl.String, description="Kind of the body the robot made contact with."
    )
    obstacle_id = Column(
        pl.String, description="Identity of the body the robot made contact with."
    )
    t_onset_ns = Column(pl.Int64, description="Onset of the contact in nanoseconds.")
    t_end_ns = Column(pl.Int64, description="End of the contact in nanoseconds.")
    null_reason = _NULL_REASON


class FactPedestrians(
    _RunEpisodeEnvironment,
    name="pedestrians",
    kind=TableKind.FACT,
    scope=MetricScope.PEDESTRIAN,
):
    """Per-pedestrian facts: one row per pedestrian of one episode."""

    key = ("run_id", "episode_id", "env_id", "ped_id")

    ped_id = Column(
        pl.String, description="Identity of the pedestrian within the environment."
    )
    ped_name = Column(pl.String, description="Name of the pedestrian agent.")
    ped_policy = Column(
        pl.String, null_reason=True, description="Behaviour policy the pedestrian ran."
    )
    desired_speed = Column(
        pl.Float64,
        null_reason=True,
        unit="m/s",
        description="Desired walking speed of the pedestrian.",
    )
    radius_m = Column(
        pl.Float64,
        null_reason=True,
        unit="m",
        description="Body radius of the pedestrian.",
    )
    spawn_x = Column(
        pl.Float64,
        null_reason=True,
        unit="m",
        description="Spawn position of the pedestrian, x.",
    )
    spawn_y = Column(
        pl.Float64,
        null_reason=True,
        unit="m",
        description="Spawn position of the pedestrian, y.",
    )
    spawn_yaw = Column(
        pl.Float64,
        null_reason=True,
        unit="rad",
        description="Spawn heading of the pedestrian.",
    )
    null_reason = _NULL_REASON


class FactRobots(
    _RunEpisodeEnvironment, name="robots", kind=TableKind.FACT, scope=MetricScope.ROBOT
):
    """Per-robot facts: one row per robot of one episode."""

    key = ("run_id", "episode_id", "env_id", "robot_id")

    robot_id = Column(
        pl.String, description="Identity of the robot within the environment."
    )
    robot_name = Column(pl.String, description="Name of the robot as launched.")
    null_reason = _NULL_REASON


class FactEdgeCases(_RunEpisodeRobot, name="edge_cases", kind=TableKind.FACT):
    """Edge-case rows: one row per list element of an edge case, reshaped to first normal form."""

    key = ("run_id", "episode_id", "robot_id", "case_idx", "kind", "ordinal")

    case_idx = Column(
        pl.Int64, description="Index of the edge case within the episode."
    )
    kind = Column(pl.String, description="Kind of the edge case.")
    ordinal = Column(pl.Int64, description="Position of the row within its edge case.")
    null_reason = _NULL_REASON


class FactFormations(_RunEpisodeRobot, name="formations", kind=TableKind.FACT):
    """Formation rows: one row per formation member."""

    key = ("run_id", "episode_id", "robot_id")
    unique_key = False

    formation_idx = Column(
        pl.Int64, description="Index of the formation within the episode."
    )
    role = Column(pl.String, description="Role of the member in the formation.")
    null_reason = _NULL_REASON


class FactPhaseOutcomes(_RunEpisodeRobot, name="phase_outcomes", kind=TableKind.FACT):
    """Phase rows: one row per replayed phase."""

    key = ("run_id", "episode_id", "robot_id", "phase_idx")

    phase_idx = Column(pl.Int64, description="Index of the replayed phase.")
    null_reason = _NULL_REASON


class FactJudgeDisagreements(
    _RunEpisodeRobot, name="judge_disagreements", kind=TableKind.FACT
):
    """Judge rows: one row per disagreement between judges."""

    key = ("run_id", "episode_id", "robot_id", "disagreement_idx")

    disagreement_idx = Column(
        pl.Int64, description="Index of the disagreement within the episode."
    )
    null_reason = _NULL_REASON


class FactMechanismStates(
    _RunEpisodeRobot, name="mechanism_states", kind=TableKind.FACT
):
    """Mechanism rows: one row per change of a mechanism state."""

    key = ("run_id", "episode_id", "robot_id")
    unique_key = False

    entity = Column(pl.String, description="Entity whose mechanism state changed.")
    level = Column(
        pl.String,
        null_reason=True,
        description="Level of the entity the state refers to.",
    )
    kind = Column(pl.String, description="Kind of the entity.")
    field = Column(pl.String, description="Name of the state field that changed.")
    t_ns = Column(
        pl.Int64, null_reason=True, description="Time of the change in nanoseconds."
    )
    value = Column(
        pl.Float64,
        unit="per field",
        description="Value of the state field, in the unit the field declares.",
    )
    null_reason = _NULL_REASON


class FactCollisions(_RunEpisodeRobot, name="collisions", kind=TableKind.FACT):
    """Collision rows: one row per collision of the robot."""

    key = ("run_id", "episode_id", "robot_id", "collision_idx")

    collision_idx = Column(
        pl.Int64, description="Index of the collision within the episode."
    )
    obstacle_kind = Column(
        pl.String,
        null_reason=True,
        description="Kind of the obstacle the robot collided with.",
    )
    t_ns = Column(
        pl.Int64, null_reason=True, description="Time of the collision in nanoseconds."
    )
    x_m = Column(
        pl.Float64,
        null_reason=True,
        unit="m",
        description="Position of the collision, x.",
    )
    y_m = Column(
        pl.Float64,
        null_reason=True,
        unit="m",
        description="Position of the collision, y.",
    )
    null_reason = _NULL_REASON


FACT_TABLES: Final[dict[FactTable, type[Table]]] = {
    FactTable.EPISODES: FactEpisodes,
    FactTable.ENCOUNTERS: FactEncounters,
    FactTable.CONTACTS: FactContacts,
    FactTable.PEDESTRIANS: FactPedestrians,
    FactTable.ROBOTS: FactRobots,
    FactTable.EDGE_CASES: FactEdgeCases,
    FactTable.FORMATIONS: FactFormations,
    FactTable.PHASE_OUTCOMES: FactPhaseOutcomes,
    FactTable.JUDGE_DISAGREEMENTS: FactJudgeDisagreements,
    FactTable.MECHANISM_STATES: FactMechanismStates,
    FactTable.COLLISIONS: FactCollisions,
}
