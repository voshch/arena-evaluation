"""The recorded telemetry tables: one declaration per recorded topic, keyed by its native time."""

from __future__ import annotations

from typing import ClassVar, Final, cast

import polars as pl

from arena_evaluation.storage.registry.scopes import MetricScope
from arena_evaluation.storage.registry.table import Column, Table, TableKind

_TIME: Final[Column] = Column(pl.Int64, required=True)
_TEXT: Final[Column] = Column(pl.String)
_INT: Final[Column] = Column(pl.Int64)
_FLOAT: Final[Column] = Column(pl.Float64)
_FLAG: Final[Column] = Column(pl.Boolean)
_FLOAT32: Final[Column] = Column(pl.Float32)


class _RobotTelemetry(Table, abstract=True):
    """Telemetry of one robot: its native time, environment and robot."""

    key = ("env_id", "robot_id")
    row_key: ClassVar[tuple[str, ...]] = ("env_id", "robot_id", "t_ns")

    t_ns = _TIME
    env_id = _TEXT
    robot_id = _TEXT


class _EnvironmentTelemetry(Table, abstract=True):
    """Telemetry of one simulation environment: its native time and environment."""

    key = ("env_id",)
    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns")

    t_ns = _TIME
    env_id = _TEXT


class _GlobalTelemetry(Table, abstract=True):
    """Telemetry that belongs to no robot and no environment key: its native time only."""

    row_key: ClassVar[tuple[str, ...]] = ("t_ns",)

    t_ns = _TIME


class _OdomColumns(Table, abstract=True):
    """Odometry columns shared by a robot's odometry and the odometry its controller reports."""

    odom_x = _FLOAT
    odom_y = _FLOAT
    odom_yaw = _FLOAT
    odom_vx = _FLOAT
    odom_wz = _FLOAT
    drift_error_m = _FLOAT


class _CmdVelColumns(Table, abstract=True):
    """Velocity-command columns shared by a robot's commands and the commands its controller issues."""

    linear_x = _FLOAT
    linear_y = _FLOAT
    linear_z = _FLOAT
    angular_x = _FLOAT
    angular_y = _FLOAT
    angular_z = _FLOAT


class _JointColumns(Table, abstract=True):
    """One joint of a robot, the extra key of a per-joint row."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "robot_id", "t_ns", "joint_name")
    joint_name = _TEXT


class _PedestrianStateColumns(Table, abstract=True):
    """Crowd-engine state columns shared by the arena_peds and agent_states topics."""

    ped_id = _TEXT
    ped_name = _TEXT
    x = _FLOAT
    y = _FLOAT
    z = _FLOAT
    yaw = _FLOAT
    vx = _FLOAT
    vy = _FLOAT
    vz = _FLOAT
    speed_mps = _FLOAT
    radius_m = _FLOAT
    desired_speed = _FLOAT
    policy = _TEXT
    agent_type = _TEXT
    animation_state = _INT
    interaction_id = _INT
    interaction_type = _INT
    gait_phase = _FLOAT
    gait_cadence = _FLOAT


class _TransformColumns(Table, abstract=True):
    """A transform between two frames, shared by the dynamic and the static transform topics."""

    parent_frame = _TEXT
    child_frame = _TEXT
    tx = _FLOAT
    ty = _FLOAT
    tz = _FLOAT
    qx = _FLOAT
    qy = _FLOAT
    qz = _FLOAT
    qw = _FLOAT


class RobotTracks(
    _RobotTelemetry,
    name="robot_tracks",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Ground-truth or estimated pose and velocity of a robot."""

    x = _FLOAT
    y = _FLOAT
    z = _FLOAT
    yaw = _FLOAT
    vx = _FLOAT
    vy = _FLOAT
    vz = _FLOAT
    omega_z = _FLOAT
    speed_mps = _FLOAT
    pose_source = _TEXT


class RobotOdom(
    _OdomColumns,
    _RobotTelemetry,
    name="robot_odom",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Odometry of a robot."""


class RobotOdomController(
    _OdomColumns,
    _RobotTelemetry,
    name="robot_odom_controller",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Odometry as the controller reported it."""


class RobotCmdVel(
    _CmdVelColumns,
    _RobotTelemetry,
    name="robot_cmd_vel",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Velocity commands sent to a robot."""


class RobotCmdVelController(
    _CmdVelColumns,
    _RobotTelemetry,
    name="robot_cmd_vel_controller",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Velocity commands as the controller issued them."""


class RobotScanSummary(
    _RobotTelemetry,
    name="robot_scan_summary",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Summary of one laser scan of a robot."""

    scan_min_m = _FLOAT
    scan_max_m = _FLOAT
    range_min_m = _FLOAT
    range_max_m = _FLOAT
    n_valid_rays = _INT


class RobotGoalPose(
    _RobotTelemetry,
    name="robot_goal_pose",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """The goal the robot was last given, as recorded on its goal_pose topic."""

    pos_x = _FLOAT
    pos_y = _FLOAT
    pos_z = _FLOAT
    yaw = _FLOAT
    frame_id = _TEXT


class RobotJointStates(
    _JointColumns,
    _RobotTelemetry,
    name="robot_joint_states",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Joint state of one joint of a robot."""

    position = _FLOAT
    velocity = _FLOAT
    effort = _FLOAT


class RobotPower(
    _RobotTelemetry,
    name="robot_power",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Electrical and thermal power of a robot."""

    total_power_w = _FLOAT
    static_power_w = _FLOAT
    total_mech_power_w = _FLOAT
    total_thermal_power_w = _FLOAT


class RobotJointPower(
    _JointColumns,
    _RobotTelemetry,
    name="robot_joint_power",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Power of one joint of a robot."""

    mechanical_power_w = _FLOAT
    thermal_power_w = _FLOAT
    total_power_w = _FLOAT


class RobotEnergy(
    _RobotTelemetry,
    name="robot_energy",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Energy use and battery state of a robot."""

    total_energy_consumed_wh = _FLOAT
    battery_soc_percent = _FLOAT


class RobotAcoustics(
    _RobotTelemetry,
    name="robot_acoustics",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Acoustic levels and validity flags of a robot."""

    total_level_af_dba = _FLOAT
    total_level_zf_db = _FLOAT
    baseline_level_dba = _FLOAT
    drivetrain_level_dba = _FLOAT
    operating_state = _TEXT
    calibration_status = _TEXT
    uncertainty_1sigma_dba = _FLOAT
    validity_flags = _INT


class RobotCollisionEvents(
    _RobotTelemetry,
    name="robot_collision_events",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Counts of collision events of a robot, by obstacle class."""

    event_count = _INT
    wall_count = _INT
    static_count = _INT
    ped_count = _INT


class RobotCollisionContacts(
    _RobotTelemetry,
    name="robot_collision_contacts",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """One contact of a robot with an obstacle."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "robot_id", "t_ns", "obstacle_id")
    obstacle_id = _TEXT
    kind = _TEXT
    obstacle_x = _FLOAT
    obstacle_y = _FLOAT
    obstacle_z = _FLOAT
    distance = _FLOAT


class RobotCollisionMonitor(
    _RobotTelemetry,
    name="robot_collision_monitor",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Action taken by the collision monitor of a robot."""

    action_type = _INT
    polygon_name = _TEXT


class RobotPlans(
    _RobotTelemetry,
    name="robot_plans",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """One waypoint of a plan of a robot."""

    row_key: ClassVar[tuple[str, ...]] = (
        "env_id",
        "robot_id",
        "t_ns",
        "plan_id",
        "waypoint_idx",
    )
    plan_id = _INT
    waypoint_idx = _INT
    x = _FLOAT
    y = _FLOAT
    yaw = _FLOAT


class RobotCharacterizationPhase(
    _RobotTelemetry,
    name="robot_characterization_phase",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Characterization phase label of a robot."""

    label = _TEXT


class RobotCharacterizationSchedule(
    _RobotTelemetry,
    name="robot_characterization_schedule",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """One phase of the characterization schedule of a robot."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "robot_id", "t_ns", "phase_idx")
    phase_idx = _INT
    phase_label = _TEXT
    phase_kind = _TEXT
    vx_target = _FLOAT
    vy_target = _FLOAT
    wz_target = _FLOAT
    duration_s = _FLOAT
    ramp_s = _FLOAT
    turn_radius_m = _FLOAT


class RobotInitialPose(
    _RobotTelemetry,
    name="robot_initialpose",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Initial pose given to a robot."""

    pos_x = _FLOAT
    pos_y = _FLOAT
    yaw = _FLOAT


class RobotTaskPose(
    _RobotTelemetry,
    name="robot_task_pose",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Task pose given to a robot."""

    stamp_ns = _INT
    pos_x = _FLOAT
    pos_y = _FLOAT
    yaw = _FLOAT


class RobotDescriptor(
    _RobotTelemetry,
    name="robot_descriptor",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """Identity of a robot of the fleet: one row, from the latched fleet message."""

    name = _TEXT
    model = _TEXT
    ns = _TEXT
    frame = _TEXT
    base_frame = _TEXT


class RobotParamRows(
    _RobotTelemetry,
    name="robot_params",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.ROBOT,
):
    """One resolved morphology parameter of a robot, as the fleet message carries it."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "robot_id", "t_ns", "param_key")
    param_key = _TEXT
    param_value = _TEXT


class PedestrianArenaPeds(
    _PedestrianStateColumns,
    _EnvironmentTelemetry,
    name="pedestrian_arena_peds",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """Pedestrian poses and velocities as the arena_peds topic publishes them, one row per pedestrian and time."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "ped_id")


class PedestrianAgentStates(
    _PedestrianStateColumns,
    _EnvironmentTelemetry,
    name="pedestrian_agent_states",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """Crowd-engine state of each pedestrian, with the four robot-interaction settings of its agent."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "ped_id")
    robot_awareness = _FLOAT32
    robot_fov_gain = _FLOAT32
    robot_repulsion_gain = _FLOAT32
    robot_reaction_gain = _FLOAT32


class PedestrianGestures(
    _EnvironmentTelemetry,
    name="pedestrian_gestures",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """A gesture of a pedestrian."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "ped_id", "slot")
    ped_id = _TEXT
    slot = _TEXT
    clip = _TEXT
    hand = _TEXT
    at_x = _FLOAT
    at_y = _FLOAT
    at_z = _FLOAT
    render_pose_override = _FLAG


class PedestrianAnimations(
    _EnvironmentTelemetry,
    name="pedestrian_animations",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """An animation layer of a pedestrian."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "ped_id", "slot")
    ped_id = _TEXT
    ped_name = _TEXT
    base = _TEXT
    base_phase = _FLOAT
    slot = _TEXT
    clip = _TEXT
    playhead = _FLOAT
    duration = _FLOAT
    weight = _FLOAT
    loop = _FLAG


class PedestrianInteractions(
    _EnvironmentTelemetry,
    name="pedestrian_interactions",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """A pedestrian interaction with an object or another pedestrian."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "interaction_id")
    interaction_id = _INT
    interaction_type = _INT
    outcome = _INT
    object_id = _TEXT
    duration = _FLOAT
    arrived = _FLAG
    holding = _FLAG


class PedestrianInteractionMembers(
    _EnvironmentTelemetry,
    name="pedestrian_interaction_members",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """The members of a pedestrian interaction and their roles."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "interaction_id", "ped_id")
    interaction_id = _INT
    ped_id = _TEXT
    member_role = _TEXT


class PedestrianInteractionEvents(
    _EnvironmentTelemetry,
    name="pedestrian_interaction_events",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.PEDESTRIAN,
):
    """An event of a pedestrian interaction."""

    row_key: ClassVar[tuple[str, ...]] = (
        "env_id",
        "t_ns",
        "interaction_id",
        "event",
        "ped_id",
    )
    interaction_id = _INT
    interaction_type = _INT
    event = _INT
    event_name = _TEXT
    ped_id = _TEXT


class EpisodeLifecycle(
    _EnvironmentTelemetry,
    name="episode_lifecycle",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.EPISODE,
):
    """Lifecycle record of an episode as the runner wrote it."""

    episode_id = _INT
    outcome_state = _INT
    outcome_info = _TEXT
    goal_uuid = _TEXT
    task_layer = _TEXT
    modes_json = _TEXT
    robots_json = _TEXT
    goal_dist_start_m = _FLOAT
    goal_dist_min_m = _FLOAT
    path_length_m = _FLOAT
    start_t_ns = _INT
    phases_yaml = _TEXT
    params_yaml = _TEXT
    conditions_yaml = _TEXT
    world = _TEXT
    seed = _INT
    human_params = _TEXT


class EpisodeIntegrity(
    _EnvironmentTelemetry,
    name="episode_integrity",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.EPISODE,
):
    """An integrity flag of an episode."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "flag_idx")
    flag_idx = _INT
    flag = _TEXT


class Tf(
    _TransformColumns,
    _GlobalTelemetry,
    name="tf",
    kind=TableKind.TELEMETRY,
):
    """Dynamic transforms between frames."""

    row_key: ClassVar[tuple[str, ...]] = ("t_ns", "parent_frame", "child_frame")


class TfStatic(
    _TransformColumns,
    _GlobalTelemetry,
    name="tf_static",
    kind=TableKind.TELEMETRY,
):
    """Static transforms between frames."""

    row_key: ClassVar[tuple[str, ...]] = ("t_ns", "parent_frame", "child_frame")


class WorldSemantics(
    _GlobalTelemetry, name="world_semantics", kind=TableKind.TELEMETRY
):
    """Semantic state of the world's entities, one field per row."""

    row_key: ClassVar[tuple[str, ...]] = (
        "t_ns",
        "env_id",
        "entity_name",
        "kind",
        "field_name",
        "field_kind",
    )
    env_id = _TEXT
    world = _TEXT
    entity_name = _TEXT
    kind = _TEXT
    field_name = _TEXT
    field_kind = _TEXT
    value_str = _TEXT
    value_num = _FLOAT
    value_bool = _FLAG


class EnvironmentMap(
    _EnvironmentTelemetry,
    name="environment_map",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.EPISODE,
):
    """Metadata of one latched occupancy-grid snapshot recorded from an environment map or door mask."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "kind")
    kind = _TEXT
    frame_id = _TEXT
    resolution_m = _FLOAT
    width_cells = _INT
    height_cells = _INT
    origin_x = _FLOAT
    origin_y = _FLOAT
    origin_yaw = _FLOAT


class EnvironmentMapCells(
    _EnvironmentTelemetry,
    name="environment_map_cells",
    kind=TableKind.TELEMETRY,
    scope=MetricScope.EPISODE,
):
    """One non-free cell of a recorded occupancy-grid snapshot, stored sparsely."""

    row_key: ClassVar[tuple[str, ...]] = ("env_id", "t_ns", "kind", "row", "col")
    kind = _TEXT
    row = _INT
    col = _INT
    occupancy = _INT


_TelemetryTable = (
    type[_RobotTelemetry] | type[_EnvironmentTelemetry] | type[_GlobalTelemetry]
)


def duplicate_row_keys(table: type[Table], frame: pl.DataFrame) -> pl.DataFrame:
    """Rows of a telemetry frame whose row key repeats, with their key columns."""
    keys = cast(_TelemetryTable, table).row_key
    return frame.filter(pl.struct(keys).is_duplicated()).select(keys)


def null_row_keys(table: type[Table], frame: pl.DataFrame) -> pl.DataFrame:
    """Rows of a telemetry frame whose row key carries a null, with their key columns."""
    keys = cast(_TelemetryTable, table).row_key
    return frame.filter(
        pl.any_horizontal([pl.col(key).is_null() for key in keys])
    ).select(keys)
