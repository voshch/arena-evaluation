from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from arena_evaluation.storage.exceptions import ContractViolationError, MetricInputError

if TYPE_CHECKING:
    import pathlib


class RunMetadata(BaseModel):
    """Metadata for a single episode, serialized to episode_XXX.yaml."""

    benchmark_id: str
    planner: str
    robot_model: list[str] = Field(default_factory=list)
    map: str
    stage: str
    episode_id: int | None = None
    episodes_requested: int = 0
    suite_name: str = ""
    contest_name: str = ""
    local_planner: str = ""
    inter_planner: str = ""
    agent_name: str = ""
    task_generator_episode_id: int | None = None
    # Terminal outcome written by the runner's stop_episode service call
    # (QUEUED=0, RUNNING=1, SUCCESS=2, FAILED=3, SKIPPED=4, FATAL=5).
    outcome_state: int | None = None
    outcome_info: str = ""
    integrity: list[str] = Field(default_factory=list)
    recording_started_at: str
    arena_git_sha: str | None = None
    arena_git_dirty: bool = False
    python_version: str
    ros_distro: str

    task_layer: str = ""
    modes: dict[str, list[str]] | None = None
    params: dict[str, typing.Any] | None = None

    recording_ended_at: str | None = None
    episodes_recorded: int | None = None
    pedsim_available: bool | None = None
    recorded_topics: list[str] | None = None

    processing_completed_at: str | None = None
    episodes_valid: int | None = None
    pipeline_version: str | None = None

    rosbag2_message_count: int | None = None
    rosbag2_topics: list[typing.Any] | None = None

    env_ns_root: str | None = None
    is_reference: bool = False
    reference_type: str | None = None
    parent_episode_id: int | None = None


@dataclass(frozen=True)
class RobotParams:
    """Robot physical parameters loaded at runtime."""

    model: str = "unknown"
    robot_radius: float = 0.25
    laser_min_range: float = 0.0
    laser_max_range: float = 30.0
    base_mass: float = 0.0
    component_masses: dict[str, float] = field(default_factory=dict)

    @property
    def mass(self) -> float:
        """Total mass in kg, or 0.0 if undeclared."""
        return self.base_mass + sum(self.component_masses.values())

    @classmethod
    def load(cls, model: str) -> RobotParams:
        """Load parameters from the arena_robots share directory; raises when the package or a parameter file is missing."""
        import os

        import yaml
        from ament_index_python.packages import get_package_share_directory

        defaults = cls()
        radius = defaults.robot_radius
        base_mass = defaults.base_mass
        component_masses: dict[str, float] = {}

        try:
            robot_dir = os.path.join(
                get_package_share_directory("arena_robots"),
                "robots",
                model.partition("[")[0],
            )
        except (ImportError, OSError, KeyError) as error:
            raise FileNotFoundError(
                f"arena_robots is not installed, so the parameters of {model!r} are unknown"
            ) from error

        def _read(path: str) -> dict:
            with open(path) as file:
                return yaml.safe_load(file) or {}

        radius = float(
            _read(os.path.join(robot_dir, "caps", "mobile.yaml")).get("radius", radius)
        )
        base_mass = float(
            _read(os.path.join(robot_dir, "model_params.yaml"))
            .get("mass", {})
            .get("base_kg", base_mass)
        )
        components_root = os.path.join(
            os.path.dirname(os.path.dirname(robot_dir)), "components"
        )
        for kind, entries in (
            _read(os.path.join(robot_dir, "assembly.yaml")).get("defaults", {}).items()
        ):
            for entry in entries or []:
                variant = (entry or {}).get("variant")
                if not variant:
                    continue
                kg = (
                    _read(
                        os.path.join(components_root, kind, variant, "component.yaml")
                    )
                    .get("mass", {})
                    .get("kg")
                )
                if kg:
                    component_masses[f"{kind}/{variant}"] = float(kg)

        return cls(
            model=model,
            robot_radius=radius,
            laser_min_range=defaults.laser_min_range,
            laser_max_range=defaults.laser_max_range,
            base_mass=base_mass,
            component_masses=component_masses,
        )


#: The recorded parameter keys that fill a :class:`RobotParams` field.
_RECORDED_PARAMETER_FIELDS: typing.Final[dict[str, str]] = {
    "robot_radius": "robot_radius",
    "robot_mass_kg": "base_mass",
    "base_mass": "base_mass",
    "laser_min_range": "laser_min_range",
    "laser_max_range": "laser_max_range",
}

#: The RobotParams fields a recorded robot must declare, with the recorded key each one needs.
_REQUIRED_RECORDED_FIELDS: typing.Final[dict[str, str]] = {
    "robot_radius": "robot_radius",
    "base_mass": "robot_mass_kg",
}


def robot_params_from_recorded(rows: pl.DataFrame, robot_id: str) -> RobotParams:
    """The parameters of one robot from its recorded fleet rows.

    ``rows`` are the recorded fleet rows of one episode: the descriptor row (``model`` and the
    frame columns) plus one row per resolved parameter (``param_key``/``param_value``). The two
    may be concatenated or joined. A robot without a descriptor row or a required parameter raises
    ``MetricInputError``, which fails the benchmark; a parameter that cannot be parsed raises
    ``ContractViolationError``. Nothing falls back to a default model, radius or mass.
    """
    if "robot_id" not in rows.columns:
        raise MetricInputError(
            "the recorded fleet rows carry no robot_id; the robot has no descriptor"
        )
    robot_rows = rows.filter(pl.col("robot_id") == robot_id)
    if robot_rows.height == 0:
        raise MetricInputError(f"robot {robot_id!r} has no descriptor row")
    model = _descriptor_model(robot_rows, robot_id)
    recorded = _recorded_parameters(robot_rows, robot_id)
    missing = [
        key for field, key in _REQUIRED_RECORDED_FIELDS.items() if field not in recorded
    ]
    if missing:
        raise MetricInputError(
            f"robot {robot_id!r} has no recorded parameter(s) {missing}"
        )
    return RobotParams(model=model, **recorded)


def _descriptor_model(rows: pl.DataFrame, robot_id: str) -> str:
    """The model of the robot's descriptor row; a row that carries none is not a descriptor."""
    if "model" not in rows.columns:
        raise MetricInputError(f"robot {robot_id!r} has no descriptor row")
    models = rows.get_column("model").drop_nulls().unique().to_list()
    if not models:
        raise MetricInputError(f"robot {robot_id!r} has no descriptor row")
    if len(models) > 1:
        raise ContractViolationError(
            f"robot {robot_id!r} has conflicting recorded models {models}"
        )
    return str(models[0])


def _recorded_parameters(rows: pl.DataFrame, robot_id: str) -> dict[str, float]:
    """The recorded values of the RobotParams fields, from the long parameter rows and then the wide columns."""
    values: dict[str, float] = {}
    if {"param_key", "param_value"} <= set(rows.columns):
        for key, value in (
            rows.select("param_key", "param_value").drop_nulls("param_key").iter_rows()
        ):
            field = _RECORDED_PARAMETER_FIELDS.get(str(key))
            if field is not None:
                values[field] = _parameter_number(robot_id, str(key), value)
    for field, key in _REQUIRED_RECORDED_FIELDS.items():
        if field in values or field not in rows.columns:
            continue
        candidates = rows.get_column(field).drop_nulls().unique().to_list()
        if len(candidates) > 1:
            raise ContractViolationError(
                f"robot {robot_id!r} has conflicting values of {key!r}: {candidates}"
            )
        if candidates:
            values[field] = _parameter_number(robot_id, key, candidates[0])
    return values


def _parameter_number(robot_id: str, key: str, value: object) -> float:
    """The numeric value of one recorded parameter; a value that is not a number is refused, never defaulted."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ContractViolationError(
            f"robot {robot_id!r}: recorded parameter {key!r} is not a number: {value!r}"
        ) from error


@dataclass(frozen=True)
class EpisodeDescriptor:
    """Descriptor for an episode directory in flat storage."""

    episode_dir: str
    benchmark_id: str
    episode_id: int
    planner: str
    stage: str
    map: str
    is_reference: bool = False
    reference_type: str | None = None


@dataclass
class TopicBundle:
    """Flat telemetry tables of one robot: the episode's ingested tables filtered to its robot and environment."""

    tf: pl.DataFrame | pl.LazyFrame | None = None
    tf_static: pl.DataFrame | pl.LazyFrame | None = None
    world_semantics: pl.DataFrame | pl.LazyFrame | None = None
    robot_tracks: pl.DataFrame | pl.LazyFrame | None = None
    robot_odom: pl.DataFrame | pl.LazyFrame | None = None
    robot_cmd_vel: pl.DataFrame | pl.LazyFrame | None = None
    robot_scan_summary: pl.DataFrame | pl.LazyFrame | None = None
    robot_scan_rays: pl.DataFrame | pl.LazyFrame | None = None
    robot_joint_states: pl.DataFrame | pl.LazyFrame | None = None
    robot_power: pl.DataFrame | pl.LazyFrame | None = None
    robot_joint_power: pl.DataFrame | pl.LazyFrame | None = None
    robot_energy: pl.DataFrame | pl.LazyFrame | None = None
    robot_acoustics: pl.DataFrame | pl.LazyFrame | None = None
    robot_collision_events: pl.DataFrame | pl.LazyFrame | None = None
    robot_collision_contacts: pl.DataFrame | pl.LazyFrame | None = None
    robot_collision_monitor: pl.DataFrame | pl.LazyFrame | None = None
    robot_plans: pl.DataFrame | pl.LazyFrame | None = None
    robot_characterization_phase: pl.DataFrame | pl.LazyFrame | None = None
    robot_characterization_schedule: pl.DataFrame | pl.LazyFrame | None = None
    robot_initialpose: pl.DataFrame | pl.LazyFrame | None = None
    robot_task_pose: pl.DataFrame | pl.LazyFrame | None = None
    robot_descriptor: pl.DataFrame | pl.LazyFrame | None = None
    robot_params: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_arena_peds: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_agent_states: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_gestures: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_animations: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interactions: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interaction_members: pl.DataFrame | pl.LazyFrame | None = None
    pedestrian_interaction_events: pl.DataFrame | pl.LazyFrame | None = None
    episode_lifecycle: pl.DataFrame | pl.LazyFrame | None = None
    episode_integrity: pl.DataFrame | pl.LazyFrame | None = None

    def available(self) -> set[str]:
        """Names of the topics this bundle carries."""
        return {name for name, table in vars(self).items() if table is not None}


@dataclass
class AlignedEpisodeBundle:
    """Aligned topic data for a single episode."""

    episode_id: int
    data: pl.DataFrame
    start_pos: list[float]
    goal_pos: list[float]
    num_pedestrians: int = 0
    robot_name: str | None = None
    semantic_snapshot: pl.DataFrame | None = None
    conditions: list[dict] | None = None
    # This robot's entry of EpisodeRecord.phases ({phases, conditions, map_poses, goal_inputs, instructions}), None when the record has none.
    phases: dict | None = None
    # Judge poses of every robot in the env by bare name, this robot included.
    fleet: dict[str, pl.DataFrame] = field(default_factory=dict)
    # Final EpisodeRecord.outcome_state of the recording, None when no record was captured.
    outcome_state: int | None = None
    outcome_info: str | None = None
    run: typing.Any = None
    folder_manager: typing.Any = None
    peds: pl.DataFrame | None = None
    map: str | None = None
    topics: dict[str, pl.DataFrame] | None = None
    # 1NF series frames of this robot episode by table stem, populated before scope evaluation so
    # calculators with SERIES_DEPENDS_ON read the published tables instead of retired list outputs.
    series: dict[str, pl.DataFrame] = field(default_factory=dict)
    env_offset: tuple[float, float] | None = None
    run_dir: pathlib.Path | None = None
