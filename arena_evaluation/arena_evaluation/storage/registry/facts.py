"""The eleven fact tables of one evaluated episode: identity and recorded columns; metric columns come from the metric layer."""

from __future__ import annotations

from typing import Final

import polars as pl

from arena_evaluation.storage.registry.scopes import FactTable, MetricScope
from arena_evaluation.storage.registry.table import (
    Column,
    Table,
    TableKind,
    registered_tables,
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


class FactEpisodes(_RunEpisodeRobot, name="episodes", kind=TableKind.FACT, scope=MetricScope.EPISODE):
    """Per-episode facts: one row per episode and robot."""

    key = ("run_id", "episode_id", "robot_id")

    result = Column(
        pl.String,
        null_reason=True,
        description="Terminal result of the episode as the runner recorded it.",
    )
    invalid_reason = Column(
        pl.String,
        description="Defects that make the episode unrankable, joined by semicolons; empty when it is valid.",
    )
    pose_frame = Column(
        pl.String,
        null_reason=True,
        description="TF parent frame the robot pose was taken from inside the episode window.",
    )


class FactEncounters(
    _RunEpisodeRobot,
    name="encounters",
    kind=TableKind.FACT,
    scope=MetricScope.ENCOUNTER,
):
    """Per-encounter facts: one row per encounter window of one robot."""

    key = ("run_id", "episode_id", "robot_id", "encounter_id")

    encounter_id = Column(pl.String, description="Identity of the encounter within the episode.")
    actor_1 = Column(pl.String, description="First party of the encounter, the robot.")
    actor_2 = Column(
        pl.String,
        description="Second party of the encounter, a pedestrian or another robot.",
    )
    actor_2_kind = Column(pl.String, description="Kind of the second party.")
    t_start_ns = Column(pl.Int64, description="Start of the encounter window in nanoseconds.")
    t_end_ns = Column(pl.Int64, description="End of the encounter window in nanoseconds.")
    contact_id = Column(pl.String, description="Contact of the encounter, empty when there is none.")
    approach_type = Column(
        pl.String,
        null_reason=True,
        description="Approach of the actor to the robot: stationary, head_on, overtaking or crossing.",
    )
    min_clearance_m = Column(
        pl.Float64,
        unit="m",
        description="Smallest edge-to-edge clearance of the robot and the actor inside the window.",
    )
    duration_s = Column(
        pl.Float64,
        unit="s",
        description="Length of the window from its start to its end.",
    )
    detector_version = Column(
        pl.String,
        description="Version of the encounter detector that found the window.",
    )
    detector_params_hash = Column(
        pl.String,
        description="Fingerprint of the detector parameters that found the window.",
    )


class FactContacts(_RunEpisodeRobot, name="contacts", kind=TableKind.FACT, scope=MetricScope.CONTACT):
    """Per-contact facts: one row per contact of one robot."""

    key = ("run_id", "episode_id", "robot_id", "contact_id")

    contact_id = Column(pl.String, description="Identity of the contact within the episode.")
    encounter_id = Column(pl.String, description="Encounter the contact belongs to.")
    collider_kind = Column(pl.String, description="Kind of the body the robot made contact with.")
    obstacle_id = Column(pl.String, description="Identity of the body the robot made contact with.")
    t_onset_ns = Column(pl.Int64, description="Onset of the contact in nanoseconds.")
    t_end_ns = Column(pl.Int64, description="End of the contact in nanoseconds.")
    duration_s = Column(
        pl.Float64,
        unit="s",
        description="Length of the contact window from its onset to its end.",
    )


class FactPedestrians(
    _RunEpisodeEnvironment,
    name="pedestrians",
    kind=TableKind.FACT,
    scope=MetricScope.PEDESTRIAN,
):
    """Per-pedestrian facts: one row per pedestrian of one episode."""

    key = ("run_id", "episode_id", "env_id", "ped_id")

    ped_id = Column(pl.String, description="Identity of the pedestrian within the environment.")
    ped_name = Column(pl.String, description="Name of the pedestrian agent.")
    ped_policy = Column(pl.String, null_reason=True, description="Behaviour policy the pedestrian ran.")
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


class FactRobots(_RunEpisodeEnvironment, name="robots", kind=TableKind.FACT, scope=MetricScope.ROBOT):
    """Per-robot facts: one row per robot of one episode."""

    key = ("run_id", "episode_id", "env_id", "robot_id")

    robot_id = Column(pl.String, description="Identity of the robot within the environment.")
    robot_name = Column(pl.String, description="Name of the robot as launched.")


class FactEdgeCases(_RunEpisodeRobot, name="edge_cases", kind=TableKind.FACT):
    """Edge-case rows: one row per list element of an edge case, reshaped to first normal form."""

    key = ("run_id", "episode_id", "robot_id", "case_idx", "kind", "ordinal")

    case_idx = Column(pl.Int64, description="Index of the edge case within the episode.")
    kind = Column(
        pl.String,
        description="Edge-case list the row came from: step, agent, missing or unplayed.",
    )
    ordinal = Column(pl.Int64, description="Position of the row within its edge case.")
    element = Column(pl.String, description="The step, agent or actor the row names.")


class FactFormations(_RunEpisodeRobot, name="formations", kind=TableKind.FACT):
    """Formation rows: one row per formation member."""

    key = ("run_id", "episode_id", "robot_id")
    unique_key = False

    formation_idx = Column(pl.Int64, description="Index of the formation within the episode.")
    role = Column(pl.String, description="Role of the row in the formation: member or absent.")
    element = Column(pl.String, description="The formation agent the row names.")


class FactPhaseOutcomes(_RunEpisodeRobot, name="phase_outcomes", kind=TableKind.FACT):
    """Phase rows: one row per replayed phase."""

    key = ("run_id", "episode_id", "robot_id", "phase_idx")

    phase_idx = Column(pl.Int64, description="Index of the replayed phase.")
    outcome = Column(
        pl.String,
        description="Outcome token of the phase: met, failed, dropped or pending.",
    )
    met = Column(pl.Boolean, description="Whether the replay marks the phase met.")


class FactJudgeDisagreements(_RunEpisodeRobot, name="judge_disagreements", kind=TableKind.FACT):
    """Judge rows: one row per disagreement between judges."""

    key = ("run_id", "episode_id", "robot_id", "disagreement_idx")

    disagreement_idx = Column(pl.Int64, description="Index of the disagreement within the episode.")
    disagreement = Column(
        pl.String,
        description="Text of the disagreement between the replay judge and the record.",
    )


class FactMechanismStates(_RunEpisodeRobot, name="mechanism_states", kind=TableKind.FACT):
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
    t_ns = Column(pl.Int64, null_reason=True, description="Time of the change in nanoseconds.")
    value = Column(
        pl.Float64,
        unit="per field",
        description="Value of the state field, in the unit the field declares.",
    )


class FactCollisions(_RunEpisodeRobot, name="collisions", kind=TableKind.FACT):
    """Collision rows: one row per collision of the robot."""

    key = ("run_id", "episode_id", "robot_id", "collision_idx")

    collision_idx = Column(pl.Int64, description="Index of the collision within the episode.")
    obstacle_id = Column(
        pl.String,
        null_reason=True,
        description="Identifier of the obstacle the robot collided with.",
    )
    obstacle_kind = Column(
        pl.String,
        null_reason=True,
        description="Kind of the obstacle the robot collided with.",
    )
    t_ns = Column(pl.Int64, null_reason=True, description="Time of the collision in nanoseconds.")
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


FACT_TABLES: Final[dict[FactTable, type[Table]]] = {FactTable(table.table_name): table for table in registered_tables() if table.table_kind is TableKind.FACT}
FACT_KEY_COLUMNS: Final[dict[str, tuple[str, ...]]] = {table.table_name: table.key for table in FACT_TABLES.values()}
