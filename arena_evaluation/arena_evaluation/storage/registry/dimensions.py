"""The benchmark dimensions: one row per episode, robot, pedestrian or world entity, written once per benchmark."""

from __future__ import annotations

from typing import Final

import polars as pl

from arena_evaluation.storage.registry.table import Column, Table, TableKind, registered_tables

def _key(dtype: pl.DataType | type[pl.DataType], description: str) -> Column:
    """A row-key column: never NULL."""
    return Column(dtype, required=True, description=description)


def _optional(
    dtype: pl.DataType | type[pl.DataType], description: str, unit: str = ""
) -> Column:
    """A recorded value that may be missing; a missing value carries its reason."""
    return Column(dtype, null_reason=True, unit=unit, description=description)


_RUN: Final[Column] = _key(pl.String, "Benchmark run the row belongs to.")
_EPISODE: Final[Column] = _key(pl.Int64, "Episode number within the run.")
_ENV: Final[Column] = _key(pl.String, "Simulation environment the row describes.")
_ROBOT: Final[Column] = _key(pl.String, "Robot of the episode the row describes.")


class _Dimension(Table, abstract=True):
    """The base of every dimension: keys are never NULL, every other column carries the reason of its NULL."""

    required_by_default = False


class DimEpisodes(_Dimension, name="dim_episodes", kind=TableKind.DIMENSION):
    """Episode settings as recorded: one row per episode and robot."""

    key = ("run_id", "episode_id", "robot_id")

    run_id = _RUN
    episode_id = _EPISODE
    env_id = _ENV
    robot_id = _ROBOT
    benchmark_id = _optional(pl.String, "Benchmark the run was evaluated in.")
    suite_name = _optional(pl.String, "Suite the benchmark belongs to.")
    contest_name = _optional(pl.String, "Contest the episode was run for.")
    stage = _optional(pl.String, "Stage of the contest the episode belongs to.")
    world = _optional(pl.String, "World the episode ran in.")
    scenario = _optional(pl.String, "Scenario the episode followed.")
    planner = _optional(pl.String, "Planner the robot ran.")
    inter_planner = _optional(pl.String, "Inter-planner of the episode.")
    dispatcher = _optional(pl.String, "Dispatcher that assigned the robot.")
    robot_model = _optional(
        pl.String,
        "Robot model as launched; launch provenance, not the per-robot model.",
    )
    ped_policy_default = _optional(
        pl.String, "Default crowd policy the pedestrians were given."
    )
    crowd_size = _optional(pl.Int64, "Number of pedestrians of the crowd as launched.")
    seed = _optional(pl.Int64, "Seed the episode was run with.")
    is_reference = _optional(pl.Boolean, "Whether the episode is a reference run.")
    reference_type = _optional(pl.String, "Kind of reference run.")
    duration_s = _optional(
        pl.Float64, "Duration of the episode as the runner recorded it.", unit="s"
    )
    outcome_state = _optional(
        pl.Int64, "Outcome state of the episode as the runner recorded it."
    )
    outcome_info = _optional(
        pl.String, "Outcome information of the episode as the runner recorded it."
    )
    pose_source = _optional(
        pl.String, "Ground-truth pose stream of the episode: tf (dense), tf_sparse (raw sparse samples) or absent."
    )
    pose_gap_s = _optional(pl.Float64, "Median gap between the ground-truth samples of the episode.", unit="s")


class DimRobots(_Dimension, name="dim_robots", kind=TableKind.DIMENSION):
    """One row per robot of each episode, from the recorded fleet tables."""

    key = ("run_id", "episode_id", "env_id", "robot_id")

    run_id = _RUN
    episode_id = _EPISODE
    env_id = _ENV
    robot_id = _ROBOT
    robot_model = _optional(
        pl.String, "Model of this robot, from its recorded descriptor."
    )
    planner = _optional(pl.String, "Planner the robot ran.")
    inter_planner = _optional(pl.String, "Inter-planner of the episode.")
    dispatcher = _optional(pl.String, "Dispatcher that assigned the robot.")
    max_angular_vel = _optional(
        pl.Float64, "Maximum angular velocity of the robot's parameters.", unit="rad/s"
    )
    linear_acc = _optional(
        pl.Float64, "Linear acceleration limit of the robot's parameters.", unit="m/s^2"
    )
    linear_decel = _optional(
        pl.Float64, "Linear deceleration limit of the robot's parameters.", unit="m/s^2"
    )
    robot_radius = _optional(pl.Float64, "Radius of the robot's footprint.", unit="m")
    robot_mass_kg = _optional(pl.Float64, "Mass of the robot.", unit="kg")
    static_power_w = _optional(pl.Float64, "Static power draw of the robot.", unit="W")
    lidar_variant = _optional(pl.String, "Lidar variant of the robot's model.")
    start_x = _optional(pl.Float64, "Start position of the robot, x.", unit="m")
    start_y = _optional(pl.Float64, "Start position of the robot, y.", unit="m")
    goal_x = _optional(pl.Float64, "Goal position of the robot, x.", unit="m")
    goal_y = _optional(pl.Float64, "Goal position of the robot, y.", unit="m")


class DimPedestrians(_Dimension, name="dim_pedestrians", kind=TableKind.DIMENSION):
    """One row per pedestrian of each episode: its recorded attributes and its four crowd settings."""

    key = ("run_id", "episode_id", "env_id", "ped_id")

    run_id = _RUN
    episode_id = _EPISODE
    env_id = _ENV
    ped_id = _key(pl.String, "Pedestrian the row describes.")
    ped_name = _optional(pl.String, "Name of the pedestrian agent.")
    agent_type = _optional(
        pl.String, "Agent type of the pedestrian, from its agent metadata."
    )
    ped_policy = _optional(pl.String, "Policy name the pedestrian's agent runs.")
    desired_speed = _optional(
        pl.Float64, "Desired speed of the pedestrian.", unit="m/s"
    )
    radius_m = _optional(pl.Float64, "Radius of the pedestrian.", unit="m")
    spawn_x = _optional(pl.Float64, "Spawn position of the pedestrian, x.", unit="m")
    spawn_y = _optional(pl.Float64, "Spawn position of the pedestrian, y.", unit="m")
    spawn_yaw = _optional(pl.Float64, "Spawn heading of the pedestrian.", unit="rad")
    waypoints_count = _optional(
        pl.Int64, "Number of waypoints of the pedestrian's route."
    )
    robot_awareness = _optional(
        pl.Float32,
        unit="1",
        description="How aware the pedestrian is of robots: 0 blind, 1 fully aware.",
    )
    robot_fov_gain = _optional(
        pl.Float32,
        unit="1",
        description="Robot field-of-view multiplier of the pedestrian, 1 nominal.",
    )
    robot_repulsion_gain = _optional(
        pl.Float32, unit="1", description="Repulsion multiplier from robots, 1 nominal."
    )
    robot_reaction_gain = _optional(
        pl.Float32, unit="1", description="Robot reaction-range multiplier, 1 nominal."
    )


class DimWorldEntities(_Dimension, name="dim_world_entities", kind=TableKind.DIMENSION):
    """One row per world entity of each episode, as the world declares it."""

    key = ("run_id", "episode_id", "env_id", "entity_name")

    run_id = _RUN
    episode_id = _EPISODE
    env_id = _ENV
    entity_name = _key(pl.String, "Name of the world entity.")
    entity_kind = _optional(pl.String, "Kind of the world entity.")
    zone = _optional(pl.String, "Zone the entity belongs to.")
    activation_dist = _optional(
        pl.Float64, "Distance at which the entity activates.", unit="m"
    )
    acoustic_tl_db = _optional(
        pl.Float64, "Acoustic transmission loss of the entity.", unit="dB"
    )
    initial_state = _optional(pl.String, "State the entity starts in.")


DIMENSION_TABLES: Final[dict[str, type[Table]]] = {
    table.table_name: table for table in registered_tables() if table.table_kind is TableKind.DIMENSION
}
