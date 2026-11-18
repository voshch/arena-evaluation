"""The dimension rows an episode contributes: its metadata and the recorded telemetry of each of its robots."""

from __future__ import annotations

import typing

import polars as pl

from arena_evaluation.processing.pose_anchor import PoseSource, resolve_pose_source
from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES
from arena_evaluation.storage.registry.table import Table
from arena_evaluation.storage.star_schema import conform

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import RunMetadata, TopicBundle

_NOT_RECORDED = "not_recorded"
_ROBOT_PARAMETERS = (
    "max_angular_vel",
    "linear_acc",
    "linear_decel",
    "robot_radius",
    "robot_mass_kg",
    "static_power_w",
    "lidar_variant",
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
)
_TEXT_PARAMETERS = frozenset({"lidar_variant"})
_ENTITY_FIELDS = ("zone", "activation_dist", "acoustic_tl_db", "initial_state")
_PEDESTRIAN_FIELDS = {
    "ped_name": "ped_name",
    "agent_type": "agent_type",
    "ped_policy": "policy",
    "desired_speed": "desired_speed",
    "radius_m": "radius_m",
    "spawn_x": "x",
    "spawn_y": "y",
    "spawn_yaw": "yaw",
    "robot_awareness": "robot_awareness",
    "robot_fov_gain": "robot_fov_gain",
    "robot_repulsion_gain": "robot_repulsion_gain",
    "robot_reaction_gain": "robot_reaction_gain",
}


def episode_dimensions(*, run_id: str, episode_id: int, metadata: RunMetadata, bundles: typing.Mapping[str, TopicBundle]) -> dict[str, pl.DataFrame]:
    """The four dimension tables of one episode: one episode and robot row per robot, pedestrian and entity rows once per environment."""
    episodes: list[dict[str, typing.Any]] = []
    robots: list[dict[str, typing.Any]] = []
    entities: list[dict[str, typing.Any]] = []
    pedestrians: list[pl.DataFrame] = []
    environments: set[str] = set()
    for bundle in bundles.values():
        env_id, robot_id = _identity(bundle)
        model = _robot_model(bundle)
        _, pose = resolve_pose_source(bundle, (None, None))
        episodes.append(_episode_row(run_id, episode_id, metadata, env_id, robot_id, model, pose))
        robots.append(_robot_row(run_id, episode_id, metadata, env_id, robot_id, model, bundle))
        if env_id not in environments:
            environments.add(env_id)
            entities.extend(_entity_rows(run_id, episode_id, env_id, bundle))
            pedestrians.append(conform(_pedestrian_rows(run_id, episode_id, env_id, bundle), DIMENSION_TABLES["dim_pedestrians"]))
    frames = {
        "dim_episodes": pl.DataFrame(episodes),
        "dim_robots": pl.DataFrame(robots),
        "dim_pedestrians": pl.concat(pedestrians) if pedestrians else pl.DataFrame(),
        "dim_world_entities": pl.DataFrame(entities),
    }
    return {name: _finish(frame, DIMENSION_TABLES[name]) for name, frame in frames.items()}


def _finish(frame: pl.DataFrame, table: type[Table]) -> pl.DataFrame:
    conformed = conform(frame, table)
    if conformed.height == 0:
        return conformed
    reasons = [
        pl.when(pl.col(column).is_null()).then(pl.lit(f"{column}:{_NOT_RECORDED}"))
        for column in table.null_reason_columns()
    ]
    text = pl.concat_str(reasons, separator=";", ignore_nulls=True)
    return conformed.with_columns(null_reason=pl.when(text != "").then(text))


def _episode_row(
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    env_id: str,
    robot_id: str,
    model: str | None,
    pose: PoseSource,
) -> dict[str, typing.Any]:
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "env_id": env_id,
        "robot_id": robot_id,
        "benchmark_id": metadata.benchmark_id,
        "suite_name": _text(metadata.suite_name),
        "contest_name": _text(metadata.contest_name),
        "stage": metadata.stage,
        "world": metadata.map,
        "planner": metadata.planner,
        "inter_planner": _text(metadata.inter_planner),
        "robot_model": model,
        "is_reference": metadata.is_reference,
        "reference_type": metadata.reference_type,
        "outcome_state": metadata.outcome_state,
        "outcome_info": _text(metadata.outcome_info),
        "pose_source": pose.kind,
        "pose_gap_s": pose.median_gap_s,
    }


def _robot_row(
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    env_id: str,
    robot_id: str,
    model: str | None,
    bundle: TopicBundle,
) -> dict[str, typing.Any]:
    parameters = _parameters(bundle)
    row: dict[str, typing.Any] = {
        "run_id": run_id,
        "episode_id": episode_id,
        "env_id": env_id,
        "robot_id": robot_id,
        "robot_model": model,
        "planner": metadata.planner,
        "inter_planner": _text(metadata.inter_planner),
    }
    for column in _ROBOT_PARAMETERS:
        row[column] = parameters.get(column) if column in _TEXT_PARAMETERS else _number(parameters.get(column))
    return row


def _identity(bundle: TopicBundle) -> tuple[str, str]:
    for table in (bundle.robot_descriptor, bundle.robot_tracks, bundle.robot_odom):
        frame = _frame(table)
        if frame is not None and frame.height:
            env_id, robot_id = frame.select("env_id", "robot_id").row(0)
            return str(env_id), str(robot_id)
    raise ValueError("a robot bundle has no table that names its environment and robot")


def _robot_model(bundle: TopicBundle) -> str | None:
    descriptor = _frame(bundle.robot_descriptor)
    if descriptor is None or descriptor.height == 0:
        return None
    return descriptor.get_column("model")[0]


def _parameters(bundle: TopicBundle) -> dict[str, str | None]:
    frame = _frame(bundle.robot_params)
    if frame is None or frame.height == 0:
        return {}
    return dict(zip(frame.get_column("param_key").to_list(), frame.get_column("param_value").to_list()))


def _entity_rows(run_id: str, episode_id: int, env_id: str, bundle: TopicBundle) -> list[dict[str, typing.Any]]:
    frame = _frame(bundle.world_semantics)
    if frame is None or frame.height == 0:
        return []
    entities: dict[str, dict[str, typing.Any]] = {}
    for row in frame.filter(pl.col("env_id") == env_id).iter_rows(named=True):
        entry = entities.setdefault(
            row["entity_name"],
            {"run_id": run_id, "episode_id": episode_id, "env_id": env_id, "entity_name": row["entity_name"], "entity_kind": row["kind"]},
        )
        if row["field_name"] in _ENTITY_FIELDS:
            entry.setdefault(row["field_name"], row["value_num"] if row["value_num"] is not None else row["value_str"])
    return list(entities.values())


def _pedestrian_rows(run_id: str, episode_id: int, env_id: str, bundle: TopicBundle) -> pl.DataFrame:
    frame = _frame(bundle.pedestrian_agent_states)
    if frame is None or frame.height == 0:
        frame = _frame(bundle.pedestrian_arena_peds)
    if frame is None or frame.height == 0:
        return pl.DataFrame()
    first_values = [
        pl.col(source).drop_nulls().first().alias(dimension)
        for dimension, source in _PEDESTRIAN_FIELDS.items()
        if source in frame.columns
    ]
    per_pedestrian = frame.sort("t_ns").group_by("ped_id", maintain_order=True).agg(first_values)
    return per_pedestrian.with_columns(run_id=pl.lit(run_id), episode_id=pl.lit(episode_id), env_id=pl.lit(env_id))


def _frame(table: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _text(value: str | None) -> str | None:
    return value or None


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
