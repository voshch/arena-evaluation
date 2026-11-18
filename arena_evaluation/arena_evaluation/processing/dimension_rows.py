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
_PEDESTRIAN_CORE_FIELDS = {
    "ped_name": "ped_name",
    "agent_type": "agent_type",
    "ped_policy": "policy",
    "desired_speed": "desired_speed",
    "radius_m": "radius_m",
    "spawn_x": "x",
    "spawn_y": "y",
    "spawn_yaw": "yaw",
}
_PEDESTRIAN_ENGINE_FIELDS = {
    "robot_awareness": "robot_awareness",
    "robot_fov_gain": "robot_fov_gain",
    "robot_repulsion_gain": "robot_repulsion_gain",
    "robot_reaction_gain": "robot_reaction_gain",
}
_PEDESTRIAN_FIELDS = {**_PEDESTRIAN_CORE_FIELDS, **_PEDESTRIAN_ENGINE_FIELDS}


def episode_dimensions(
    *,
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    bundles: typing.Mapping[str, TopicBundle],
) -> dict[str, pl.DataFrame]:
    """The four dimension tables of one episode: one episode and robot row per robot, pedestrian and entity rows once per environment."""
    episodes: list[dict[str, typing.Any]] = []
    robots: list[pl.DataFrame] = []
    entities: list[pl.DataFrame] = []
    pedestrians: list[pl.DataFrame] = []
    environments: set[str] = set()
    for bundle in bundles.values():
        env_id, robot_id = _identity(bundle)
        model = _robot_model(bundle)
        _, pose = resolve_pose_source(bundle, (None, None))
        episodes.append(
            _episode_row(run_id, episode_id, metadata, env_id, robot_id, model, pose)
        )
        robots.append(
            _robot_frame(run_id, episode_id, metadata, env_id, robot_id, model, bundle)
        )
        if env_id not in environments:
            environments.add(env_id)
            entities.append(
                conform(
                    _entity_frame(run_id, episode_id, env_id, bundle),
                    DIMENSION_TABLES["dim_world_entities"],
                )
            )
            pedestrians.append(
                conform(
                    _pedestrian_rows(run_id, episode_id, env_id, bundle),
                    DIMENSION_TABLES["dim_pedestrians"],
                )
            )
    frames = {
        "dim_episodes": pl.DataFrame(episodes),
        "dim_robots": pl.concat(robots) if robots else pl.DataFrame(),
        "dim_pedestrians": pl.concat(pedestrians) if pedestrians else pl.DataFrame(),
        "dim_world_entities": pl.concat(entities) if entities else pl.DataFrame(),
    }
    return {
        name: _finish(frame, DIMENSION_TABLES[name]) for name, frame in frames.items()
    }


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


def _robot_frame(
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    env_id: str,
    robot_id: str,
    model: str | None,
    bundle: TopicBundle,
) -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "run_id": [run_id],
            "episode_id": [episode_id],
            "env_id": [env_id],
            "robot_id": [robot_id],
            "robot_model": [model],
            "planner": [metadata.planner],
            "inter_planner": [_text(metadata.inter_planner)],
        }
    )
    params = _frame(bundle.robot_params)
    if params is None or params.height == 0:
        return conform(base, DIMENSION_TABLES["dim_robots"])
    param_exprs = [
        (
            pl.col("param_value").filter(pl.col("param_key") == col).first()
            if col in _TEXT_PARAMETERS
            else pl.col("param_value")
            .filter(pl.col("param_key") == col)
            .first()
            .cast(pl.Float64, strict=False)
        ).alias(col)
        for col in _ROBOT_PARAMETERS
    ]
    return conform(
        base.hstack(params.select(param_exprs)), DIMENSION_TABLES["dim_robots"]
    )


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


def _entity_frame(
    run_id: str, episode_id: int, env_id: str, bundle: TopicBundle
) -> pl.DataFrame:
    frame = _frame(bundle.world_semantics)
    if frame is None or frame.height == 0:
        return pl.DataFrame()
    filtered = frame.filter(pl.col("env_id") == env_id)
    if filtered.height == 0:
        return pl.DataFrame()
    value = pl.coalesce(pl.col("value_num").cast(pl.String), pl.col("value_str"))
    return (
        filtered.group_by("entity_name", maintain_order=True)
        .agg(
            pl.col("kind").first().alias("entity_kind"),
            *(
                value.filter(pl.col("field_name") == field).first().alias(field)
                for field in _ENTITY_FIELDS
            ),
        )
        .with_columns(
            run_id=pl.lit(run_id),
            episode_id=pl.lit(episode_id),
            env_id=pl.lit(env_id),
        )
    )


def _pedestrian_rows(
    run_id: str, episode_id: int, env_id: str, bundle: TopicBundle
) -> pl.DataFrame:
    frame = _frame(bundle.pedestrian_agent_states)
    fields = _PEDESTRIAN_FIELDS
    if frame is None or frame.height == 0:
        frame = _frame(bundle.pedestrian_arena_peds)
        fields = _PEDESTRIAN_CORE_FIELDS
    if frame is None or frame.height == 0:
        return pl.DataFrame()
    first_values = [
        pl.col(source).drop_nulls().first().alias(dimension)
        for dimension, source in fields.items()
    ]
    per_pedestrian = (
        frame.sort("t_ns").group_by("ped_id", maintain_order=True).agg(first_values)
    )
    return per_pedestrian.with_columns(
        run_id=pl.lit(run_id), episode_id=pl.lit(episode_id), env_id=pl.lit(env_id)
    )


def _frame(table: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table


def _text(value: str | None) -> str | None:
    return value or None
