"""The dimension rows an episode contributes: its metadata and the recorded telemetry of each of its robots."""

from __future__ import annotations

import typing

import polars as pl

from arena_evaluation.processing.pose_anchor import PoseSource, resolve_pose_source
from arena_evaluation.storage.exceptions import ContractViolationError
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
_NUMERIC_PARAMETERS = tuple(name for name in _ROBOT_PARAMETERS if name not in _TEXT_PARAMETERS)
_NS_PER_SECOND = 1_000_000_000
_RECORD_RUNNING = 1
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
    "robot_range_offset": "robot_range_offset",
    "robot_fov_offset": "robot_fov_offset",
    "robot_repulsion_gain": "robot_repulsion_gain",
    "robot_reaction_gain": "robot_reaction_gain",
    "robot_blind": "robot_blind",
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
        _, pose = resolve_pose_source(bundle, (None, None))
        episodes.append(_episode_row(run_id, episode_id, metadata, env_id, robot_id, pose, bundle))
        robots.append(_robot_frame(run_id, episode_id, metadata, env_id, robot_id, bundle))
        if env_id in environments:
            continue
        environments.add(env_id)
        entities.append(_entity_frame(run_id, episode_id, env_id, bundle))
        pedestrians.append(_pedestrian_rows(run_id, episode_id, env_id, bundle))
    frames = {
        "dim_episodes": pl.DataFrame(episodes),
        "dim_robots": pl.concat(robots) if robots else pl.DataFrame(),
        "dim_pedestrians": pl.concat(pedestrians) if pedestrians else pl.DataFrame(),
        "dim_world_entities": pl.concat(entities) if entities else pl.DataFrame(),
    }
    return {name: _finish(frame, DIMENSION_TABLES[name]) for name, frame in frames.items()}


def _finish(frame: pl.DataFrame, table: type[Table]) -> pl.DataFrame:
    conformed = conform(frame, table)
    if conformed.height == 0:
        return conformed
    reasons = [pl.when(pl.col(column).is_null()).then(pl.lit(f"{column}:{_NOT_RECORDED}")) for column in table.null_reason_columns()]
    text = pl.concat_str(reasons, separator=";", ignore_nulls=True)
    return conformed.with_columns(null_reason=pl.when(text != "").then(text))


def _episode_row(
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    env_id: str,
    robot_id: str,
    pose: PoseSource,
    bundle: TopicBundle,
) -> dict[str, typing.Any]:
    lifecycle = _frame(bundle.episode_lifecycle)
    peds = _first_frame(bundle, bundle.pedestrian_agent_states, bundle.pedestrian_arena_peds)
    seed, duration_s = _seed_and_duration(lifecycle)
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "env_id": env_id,
        "robot_id": robot_id,
        "benchmark_id": metadata.benchmark_id,
        "suite_name": metadata.suite_name or None,
        "contest_name": metadata.contest_name or None,
        "stage": metadata.stage,
        "world": metadata.map,
        "planner": metadata.planner,
        "inter_planner": metadata.inter_planner or None,
        "local_planner": metadata.local_planner or None,
        "is_reference": metadata.is_reference,
        "reference_type": metadata.reference_type,
        "outcome_state": metadata.outcome_state,
        "outcome_info": metadata.outcome_info or None,
        "pose_source": pose.kind,
        "pose_gap_s": pose.median_gap_s,
        "seed": seed,
        "crowd_size": _crowd_size(peds),
        "duration_s": duration_s,
        "ped_policy_default": _shared_policy(peds),
    }


def _robot_frame(
    run_id: str,
    episode_id: int,
    metadata: RunMetadata,
    env_id: str,
    robot_id: str,
    bundle: TopicBundle,
) -> pl.DataFrame:
    descriptor = _frame(bundle.robot_descriptor)
    base = pl.DataFrame(
        {
            "run_id": [run_id],
            "episode_id": [episode_id],
            "env_id": [env_id],
            "robot_id": [robot_id],
            "robot_model": [descriptor.get_column("model")[0] if descriptor is not None and descriptor.height else None],
            "planner": [metadata.planner],
            "inter_planner": [metadata.inter_planner or None],
        }
    )
    table = DIMENSION_TABLES["dim_robots"]
    params = _frame(bundle.robot_params)
    if params is None or params.height == 0:
        return conform(base, table)
    _refuse_non_numeric_parameters(params, robot_id)
    return conform(
        base.hstack(params.select(*(_parameter(name).alias(name) for name in _ROBOT_PARAMETERS))),
        table,
    )


def _parameter(name: str) -> pl.Expr:
    held = pl.col("param_value").filter(pl.col("param_key") == name).first()
    return (held if name in _TEXT_PARAMETERS else held.cast(pl.Float64, strict=True)).alias(name)


def _refuse_non_numeric_parameters(params: pl.DataFrame, robot_id: str) -> None:
    """Refuse a recorded numeric parameter that does not parse, instead of casting it to a silent NULL."""
    for key in _NUMERIC_PARAMETERS:
        recorded = params.filter(pl.col("param_key") == key).get_column("param_value").drop_nulls()
        invalid = recorded.filter(recorded.cast(pl.Float64, strict=False).is_null())
        if invalid.len():
            raise ContractViolationError(f"robot {robot_id!r}: recorded parameter {key!r} is not a number: {invalid[0]!r}")


def _identity(bundle: TopicBundle) -> tuple[str, str]:
    frame = _first_frame(bundle, bundle.robot_descriptor, bundle.robot_tracks, bundle.robot_odom)
    if frame is None:
        raise ValueError("a robot bundle has no table that names its environment and robot")
    env_id, robot_id = frame.select("env_id", "robot_id").row(0)
    return str(env_id), str(robot_id)


def _entity_frame(run_id: str, episode_id: int, env_id: str, bundle: TopicBundle) -> pl.DataFrame:
    frame = _first_frame(bundle, bundle.world_semantics)
    if frame is None:
        return pl.DataFrame()
    frame = frame.filter(pl.col("env_id") == env_id)
    if frame.height == 0:
        return pl.DataFrame()
    value = pl.coalesce(pl.col("value_num").cast(pl.String), pl.col("value_str"))
    return _stamped(
        frame.group_by("entity_name", maintain_order=True).agg(
            pl.col("kind").first().alias("entity_kind"),
            *(value.filter(pl.col("field_name") == field).first().alias(field) for field in _ENTITY_FIELDS),
        ),
        run_id,
        episode_id,
        env_id,
    )


def _pedestrian_rows(run_id: str, episode_id: int, env_id: str, bundle: TopicBundle) -> pl.DataFrame:
    agent_states = _first_frame(bundle, bundle.pedestrian_agent_states)
    frame = agent_states if agent_states is not None else _first_frame(bundle, bundle.pedestrian_arena_peds)
    if frame is None:
        return pl.DataFrame()
    fields = _PEDESTRIAN_FIELDS if agent_states is not None else _PEDESTRIAN_CORE_FIELDS
    first_values = [pl.col(source).drop_nulls().first().alias(dimension) for dimension, source in fields.items()]
    return _stamped(
        frame.sort("t_ns").group_by("ped_id", maintain_order=True).agg(first_values),
        run_id,
        episode_id,
        env_id,
    )


def _stamped(frame: pl.DataFrame, run_id: str, episode_id: int, env_id: str) -> pl.DataFrame:
    return frame.with_columns(run_id=pl.lit(run_id), episode_id=pl.lit(episode_id), env_id=pl.lit(env_id))


def _seed_and_duration(
    lifecycle: pl.DataFrame | None,
) -> tuple[int | None, float | None]:
    """The seed of the records the episode ran under and its window from its first start stamp to its last lifecycle stamp."""
    if lifecycle is None or lifecycle.height == 0:
        return None, None
    running = lifecycle.filter(pl.col("outcome_state") == _RECORD_RUNNING)
    seeds = (lifecycle if running.height == 0 else running).get_column("seed").drop_nulls().unique()
    seed = int(seeds.item()) if seeds.len() == 1 else None
    starts = lifecycle.get_column("start_t_ns").drop_nulls()
    if starts.len() == 0:
        return seed, None
    start_val = starts.min()
    end_val = lifecycle.get_column("t_ns").max()
    if not isinstance(start_val, (int, float)) or not isinstance(end_val, (int, float)):
        return seed, None
    start_ns = int(start_val)
    end_ns = int(end_val)
    return seed, (end_ns - start_ns) / _NS_PER_SECOND if end_ns > start_ns else None


def _crowd_size(peds: pl.DataFrame | None) -> int | None:
    return int(peds.get_column("ped_id").n_unique()) if peds is not None else None


def _shared_policy(peds: pl.DataFrame | None) -> str | None:
    """The pedestrian policy when every pedestrian runs the same one, None when they differ or it is absent."""
    if peds is None:
        return None
    policies = peds.get_column("policy").drop_nulls().unique()
    return str(policies.item()) if policies.len() == 1 else None


def _first_frame(
    bundle: TopicBundle,
    *tables: pl.DataFrame | pl.LazyFrame | None,
) -> pl.DataFrame | None:
    """The first recorded table of the bundle that carries a row."""
    for table in tables:
        frame = _frame(table)
        if frame is not None and frame.height:
            return frame
    return None


def _frame(table: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if table is None:
        return None
    return table.collect() if isinstance(table, pl.LazyFrame) else table
