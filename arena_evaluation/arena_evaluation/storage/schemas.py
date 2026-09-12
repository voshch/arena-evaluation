from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

import typing
from dataclasses import dataclass, field, fields

from pydantic import BaseModel, Field


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
    recording_started_at: str
    arena_git_sha: str | None = None
    arena_git_dirty: bool = False
    python_version: str
    ros_distro: str

    tm_obstacles: str | None = None
    tm_robots: str | None = None
    tm_modules: list[str] | None = None
    obstacles_params: dict[str, typing.Any] | None = None
    robots_params: dict[str, typing.Any] | None = None

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
    max_linear_velocity: float = 0.0

    @property
    def mass(self) -> float:
        """Total mass in kg, or 0.0 if undeclared."""
        return self.base_mass + sum(self.component_masses.values())

    @classmethod
    def load(cls, model: str) -> RobotParams:
        """Load parameters from arena_robots share directory, returning defaults on failure."""
        import os

        import yaml
        from ament_index_python.packages import get_package_share_directory

        defaults = cls()
        radius = defaults.robot_radius
        base_mass = defaults.base_mass
        component_masses: dict[str, float] = {}

        try:
            robot_dir = os.path.join(get_package_share_directory("arena_robots"), "robots", model.partition("[")[0])
        except Exception:
            return cls(model=model)

        def _read(path: str) -> dict:
            try:
                with open(path) as file:
                    return yaml.safe_load(file) or {}
            except Exception:
                return {}

        radius = float(_read(os.path.join(robot_dir, "caps", "mobile.yaml")).get("radius", radius))
        base_mass = float(_read(os.path.join(robot_dir, "model_params.yaml")).get("mass", {}).get("base_kg", base_mass))
        max_linear_velocity = 0.0
        for block in _read(os.path.join(robot_dir, "control.yaml")).values():
            params = block.get("ros__parameters", {}) if isinstance(block, dict) else {}
            if "linear.x.max_velocity" in params:
                max_linear_velocity = float(params["linear.x.max_velocity"])

        components_root = os.path.join(os.path.dirname(os.path.dirname(robot_dir)), "components")
        for kind, entries in _read(os.path.join(robot_dir, "assembly.yaml")).get("defaults", {}).items():
            for entry in entries or []:
                variant = (entry or {}).get("variant")
                if not variant:
                    continue
                kg = _read(os.path.join(components_root, kind, variant, "component.yaml")).get("mass", {}).get("kg")
                if kg:
                    component_masses[f"{kind}/{variant}"] = float(kg)

        return cls(
            model=model,
            robot_radius=radius,
            laser_min_range=defaults.laser_min_range,
            laser_max_range=defaults.laser_max_range,
            base_mass=base_mass,
            component_masses=component_masses,
            max_linear_velocity=max_linear_velocity,
        )


@dataclass(frozen=True)
class RunDescriptor:
    """Descriptor for a single run directory."""

    run_dir: str
    benchmark_id: str
    planner: str
    stage: str


@dataclass(frozen=True)
class EpisodeDescriptor:
    """Descriptor for an episode directory in flat storage."""

    episode_dir: str
    benchmark_id: str
    episode_id: int
    planner: str
    stage: str
    map: str = "unknown"
    is_reference: bool = False
    reference_type: str | None = None


@dataclass
class TopicBundle:
    """Raw topic dataframes extracted from MCAP."""

    odom: pl.DataFrame | None = None
    scan: pl.DataFrame | None = None
    cmd_vel: pl.DataFrame | None = None
    joint_states: pl.DataFrame | None = None
    peds: pl.DataFrame | None = None
    # crowd-engine physics poses (engine frame), where contact pairs are not drawn on their slot
    peds_physics: pl.DataFrame | None = None
    # one row per active attention channel of a ped (from arena_peds)
    ped_gestures: pl.DataFrame | None = None
    # humansim interaction snapshots and lifecycle edges (ACTIVATED / HOLD_ONSET / RELEASED / ...)
    interactions: pl.DataFrame | None = None
    interaction_events: pl.DataFrame | None = None
    # task_generator animation layer: one row per (ped, overlay slot)
    animation_states: pl.DataFrame | None = None
    episode_record: pl.DataFrame | None = None
    collision_events: pl.DataFrame | None = None
    collision_monitor_state: pl.DataFrame | None = None
    power: pl.DataFrame | None = None
    energy: pl.DataFrame | None = None
    acoustics: pl.DataFrame | None = None
    plan: pl.DataFrame | None = None
    characterization_phase: pl.DataFrame | None = None
    characterization_schedule: pl.DataFrame | None = None
    initialpose: pl.DataFrame | None = None
    tf: pl.DataFrame | None = None
    tf_static: pl.DataFrame | None = None
    tf_gt: pl.DataFrame | None = None
    semantic_snapshot: pl.DataFrame | None = None

    def available(self) -> set[str]:
        """Names of the topics this bundle carries."""
        return {f.name for f in fields(self) if getattr(self, f.name) is not None}


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
    # Final EpisodeRecord.outcome_state of the recording, None when no record was captured.
    outcome_state: int | None = None
    outcome_info: str | None = None
    run: typing.Any = None
    folder_manager: typing.Any = None
    peds: pl.DataFrame | None = None
    map: str | None = None
    topics: dict[str, pl.DataFrame] | None = None


class PlotSpec(BaseModel):
    """Specification for a single plot in a report manifest."""

    id: str
    type: str
    title: str
    data_key: str
    group_by: list[str] | str | None = None
    differentiate: str | None = "planner"
    auto_differentiate: bool = True
    filter: dict[str, typing.Any] | None = None
    options: dict[str, typing.Any] = Field(default_factory=dict)
    layout_group: str | None = None
    data_source: str | None = None
