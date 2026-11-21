"""The series tables: the per-sample rows one robot's episode publishes, one Parquet file per stem, keyed by native time."""

from __future__ import annotations

from typing import Final

import polars as pl

from arena_evaluation.storage.registry.scopes import MetricScope
from arena_evaluation.storage.registry.table import (
    SERIES_PREFIX,
    Column,
    Table,
    TableKind,
    registered_tables,
)

_RUN: Final[Column] = Column(
    pl.String, required=True, description="Benchmark run the row belongs to."
)
_EPISODE: Final[Column] = Column(
    pl.Int64, required=True, description="Episode number within the run."
)
_ROBOT: Final[Column] = Column(
    pl.String, required=True, description="Robot of the episode the row describes."
)
_TIME: Final[Column] = Column(
    pl.Int64,
    required=True,
    unit="ns",
    description="Native time of the sample in nanoseconds.",
)
_ELAPSED: Final[Column] = Column(
    pl.Float64,
    unit="s",
    description="Seconds from the first sample of the series to this one.",
)
_PEDESTRIAN: Final[Column] = Column(
    pl.String, required=True, description="Pedestrian the row describes."
)


class _Series(Table, abstract=True):
    """The base of every series table: the samples of one robot in one episode, keyed by their native time."""

    key = ("run_id", "episode_id", "robot_id", "t_ns")

    run_id = _RUN
    episode_id = _EPISODE
    robot_id = _ROBOT
    t_ns = _TIME


class _TimedSeries(_Series, abstract=True):
    """The base of the series tables whose samples carry their elapsed time since the first sample."""

    time_s = _ELAPSED


class SeriesAcousticExposure(
    _Series,
    name="series_acoustic_exposure",
    kind=TableKind.SERIES,
    scope=MetricScope.PEDESTRIAN,
):
    """Per-frame barrier-aware acoustic exposure at each reachable pedestrian."""

    key = ("run_id", "episode_id", "robot_id", "t_ns", "ped_id")

    ped_id = _PEDESTRIAN
    exposure_dba = Column(
        pl.Float64,
        unit="dBA",
        description="Barrier-aware A-weighted sound pressure level reaching the pedestrian.",
    )
    attenuation_db = Column(
        pl.Float64,
        unit="dB",
        description="Acoustic propagation attenuation from the robot to the pedestrian.",
    )
    robot_x_m = Column(
        pl.Float64,
        unit="m",
        description="Robot x position in the world frame at the sample.",
    )
    robot_y_m = Column(
        pl.Float64,
        unit="m",
        description="Robot y position in the world frame at the sample.",
    )
    source_dba = Column(
        pl.Float64,
        unit="dBA",
        description="Emitted robot A-weighted sound pressure level at the sample.",
    )
    ped_x_m = Column(
        pl.Float64,
        unit="m",
        description="Pedestrian x position in the world frame at the sample.",
    )
    ped_y_m = Column(
        pl.Float64,
        unit="m",
        description="Pedestrian y position in the world frame at the sample.",
    )
    is_episode_worst_frame = Column(
        pl.Boolean,
        description="Whether this sample is the episode-global loudest exposure frame.",
    )
    open_doors = Column(
        pl.String,
        description="Comma-separated world door keys open at the sample.",
    )


class SeriesCharacterization(
    _TimedSeries,
    name="series_characterization",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Characterization samples of the robot: power, sound level and achieved and commanded motion."""

    power_total_w = Column(
        pl.Float64,
        unit="W",
        description="Total electrical power of the robot at the sample.",
    )
    power_mech_w = Column(
        pl.Float64,
        unit="W",
        description="Mechanical power delivered to the drivetrain at the sample.",
    )
    dba = Column(
        pl.Float64,
        unit="dBA",
        description="A-weighted sound level of the drivetrain at the sample, from the recorded level or the joint-state model.",
    )
    vx_achieved_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Forward velocity the robot achieved at the sample.",
    )
    vy_achieved_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Lateral velocity the robot achieved at the sample, from its ground-truth motion.",
    )
    wz_achieved_radps = Column(
        pl.Float64,
        unit="rad/s",
        description="Yaw rate the robot achieved at the sample.",
    )
    speed_achieved_mps = Column(
        pl.Float64, unit="m/s", description="Speed the robot achieved at the sample."
    )
    accel_achieved_mps2 = Column(
        pl.Float64,
        unit="m/s²",
        description="Acceleration the robot achieved at the sample, from the change of its speed.",
    )
    accel_target_mps2 = Column(
        pl.Float64,
        unit="m/s²",
        description="Acceleration the schedule commanded at the sample.",
    )
    efficiency = Column(
        pl.Float64,
        unit="1",
        description="Mechanical to electrical power ratio at the sample, from 0 to 1; NULL below 0.1 W.",
    )
    phase_kind = Column(
        pl.String,
        description="Kind of the characterization phase at the sample, as the schedule names it or as the command classifies it; unknown when neither.",
    )
    vx_target_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Forward velocity the schedule commanded at the sample.",
    )
    vy_target_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Lateral velocity the schedule commanded at the sample.",
    )
    wz_target_radps = Column(
        pl.Float64,
        unit="rad/s",
        description="Yaw rate the schedule commanded at the sample.",
    )
    speed_target_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Magnitude of the forward velocity the schedule commanded at the sample.",
    )
    turn_radius_m = Column(
        pl.Float64, unit="m", description="Turn radius of the phase at the sample."
    )
    energy_intensity_j_m = Column(
        pl.Float64,
        unit="J/m",
        description="Electrical energy per metre travelled at the sample; NULL below 0.05 m/s.",
    )
    energy_per_rad_j_rad = Column(
        pl.Float64,
        unit="J/rad",
        description="Electrical energy per radian turned at the sample; NULL below 0.05 rad/s.",
    )
    leq_power_w = Column(
        pl.Float64,
        unit="1",
        description="Linear acoustic power proxy 10^(dBA/10) of the sample, the term the energy-equivalent level averages; not watts.",
    )


class SeriesClearance(
    _Series, name="series_clearance", kind=TableKind.SERIES, scope=MetricScope.ROBOT
):
    """Clearance samples: the edge-to-edge distance of the robot to its nearest obstacle and its nearest pedestrian."""

    obstacle_clearance_m = Column(
        pl.Float64,
        unit="m",
        description="Clearance of the robot to the nearest detected obstacle, robot radius subtracted; NULL without a valid scan.",
    )
    pedestrian_clearance_m = Column(
        pl.Float64,
        unit="m",
        description="Edge-to-edge clearance of the robot to its nearest pedestrian, both radii subtracted; NULL without a pedestrian.",
    )
    combined_clearance_m = Column(
        pl.Float64,
        unit="m",
        description="Smaller of the obstacle and pedestrian clearances at the sample, from whichever is valid.",
    )


class SeriesEnergy(
    _TimedSeries, name="series_energy", kind=TableKind.SERIES, scope=MetricScope.ROBOT
):
    """Energy samples of the robot: electrical power by component, battery charge and linear velocity."""

    power_total_w = Column(
        pl.Float64,
        unit="W",
        description="Total electrical power of the robot at the sample.",
    )
    power_static_w = Column(
        pl.Float64,
        unit="W",
        description="Static, idle electrical power of the robot at the sample.",
    )
    power_mechanical_w = Column(
        pl.Float64,
        unit="W",
        description="Mechanical power delivered to the drivetrain at the sample.",
    )
    power_thermal_w = Column(
        pl.Float64,
        unit="W",
        description="Thermal power dissipated in the drivetrain at the sample.",
    )
    battery_soc_pct = Column(
        pl.Float64, unit="%", description="Battery state of charge at the sample."
    )
    velocity_linear_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Linear velocity of the robot at the sample, as recorded.",
    )


class SeriesGaze(
    _TimedSeries, name="series_gaze", kind=TableKind.SERIES, scope=MetricScope.ROBOT
):
    """Gaze samples: how many pedestrians the robot looks at and how many look back at it."""

    looking_at_peds = Column(
        pl.Int64,
        description="Pedestrians inside the robot's 5 degree gaze cone at the sample.",
    )
    looked_at_by_peds = Column(
        pl.Int64,
        description="Pedestrians whose own 5 degree gaze cone contains the robot at the sample.",
    )


class SeriesHhiContact(
    _Series, name="series_hhi_contact", kind=TableKind.SERIES, scope=MetricScope.ROBOT
):
    """Stall samples of the robot in the human-human interaction study, one row per stall keyed by its onset."""

    onset_s = Column(
        pl.Float64,
        unit="s",
        description="Onset of the stall, in seconds from the start of the robot's time axis.",
    )
    duration_s = Column(pl.Float64, unit="s", description="Length of the stall.")
    window = Column(
        pl.String,
        description="Interaction window the stall onset falls in: approach, hold, post or rest; NULL when it falls in none.",
    )


class SeriesMotionMetrics(
    _TimedSeries,
    name="series_motion_metrics",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Motion samples of the robot: ground-truth speed, smoothed acceleration and smoothed jerk."""

    speed_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Ground-truth speed of the robot at the sample.",
    )
    acceleration_mps2 = Column(
        pl.Float64,
        unit="m/s²",
        description="Smoothed ground-truth acceleration magnitude at the sample: a Savitzky-Golay derivative of the speed, 1.0 s window, order 2.",
    )
    jerk_mps3 = Column(
        pl.Float64,
        unit="m/s³",
        description="Smoothed ground-truth jerk magnitude at the sample: a Savitzky-Golay derivative of the speed, 1.0 s window, order 2.",
    )


class SeriesPathMetrics(
    _TimedSeries,
    name="series_path_metrics",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Path samples of the robot: its ground-truth and odometry trajectory with the geometry at each sample."""

    x_m = Column(
        pl.Float64,
        unit="m",
        description="Ground-truth x position of the robot at the sample, map frame.",
    )
    y_m = Column(
        pl.Float64,
        unit="m",
        description="Ground-truth y position of the robot at the sample, map frame.",
    )
    yaw_rad = Column(
        pl.Float64,
        unit="rad",
        description="Ground-truth yaw of the robot at the sample.",
    )
    odom_x_m = Column(
        pl.Float64,
        unit="m",
        description="Wheel-odometry x position at the sample, anchored to the episode start; an estimate for drift only.",
    )
    odom_y_m = Column(
        pl.Float64,
        unit="m",
        description="Wheel-odometry y position at the sample, anchored to the episode start; an estimate for drift only.",
    )
    odom_yaw_rad = Column(
        pl.Float64,
        unit="rad",
        description="Wheel-odometry yaw at the sample, anchored to the episode start; an estimate for drift only.",
    )
    step_length_m = Column(
        pl.Float64,
        unit="m",
        description="Distance travelled since the previous sample; NULL at the first sample.",
    )
    curvature_inv_m = Column(
        pl.Float64,
        unit="1/m",
        description="Menger curvature at the sample; NULL at the ends of the track and where the position repeats.",
    )
    normalized_curvature = Column(
        pl.Float64,
        unit="1",
        description="Curvature times the sum of the two adjacent step lengths; dimensionless.",
    )
    roughness = Column(
        pl.Float64,
        unit="1",
        description="Triangle roughness at the sample: twice the triangle's area over the squared distance of its outer points; dimensionless.",
    )


class SeriesPedestrianPathMetrics(
    _TimedSeries,
    name="series_pedestrian_path_metrics",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Pedestrian path samples: the recorded position and velocity of every pedestrian, one row per pedestrian and sample."""

    key = ("run_id", "episode_id", "robot_id", "t_ns", "ped_id")

    ped_id = _PEDESTRIAN
    x_m = Column(
        pl.Float64,
        unit="m",
        description="Position of the pedestrian at the sample along x, map frame.",
    )
    y_m = Column(
        pl.Float64,
        unit="m",
        description="Position of the pedestrian at the sample along y, map frame.",
    )
    z_m = Column(
        pl.Float64,
        unit="m",
        description="Height of the pedestrian at the sample, when recorded.",
    )
    yaw_rad = Column(
        pl.Float64,
        unit="rad",
        description="Heading of the pedestrian at the sample, when recorded.",
    )
    vx_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Velocity of the pedestrian along x at the sample, when recorded.",
    )
    vy_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Velocity of the pedestrian along y at the sample, when recorded.",
    )
    speed_mps = Column(
        pl.Float64,
        unit="m/s",
        description="Speed of the pedestrian at the sample, when recorded.",
    )


class SeriesProxemics(
    _TimedSeries,
    name="series_proxemics",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Personal-space samples: how many pedestrians sit inside the robot's personal space at each sample."""

    peds_in_personal_space = Column(
        pl.Int64,
        description="Pedestrians inside 1.20 m edge clearance of the robot at the sample.",
    )


class SeriesProxemicsExtended(
    _TimedSeries,
    name="series_proxemics_extended",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Clearance samples to the nearest pedestrian, one row per frame of the robot's track."""

    min_ped_clearance_m = Column(
        pl.Float64,
        unit="m",
        description="Edge-to-edge clearance of the robot to its nearest pedestrian at the sample; NULL where it is unresolved.",
    )


class SeriesSocialForces(
    _TimedSeries,
    name="series_social_forces",
    kind=TableKind.SERIES,
    scope=MetricScope.ROBOT,
):
    """Force samples: the social force of the robot on the pedestrians and their personal-space index, one row per frame."""

    sfm_force_n = Column(
        pl.Float64,
        unit="N",
        description="Helbing social force from the robot onto the pedestrians of the frame, summed over them.",
    )
    ci = Column(
        pl.Float64,
        unit="1",
        description="Largest Gaussian personal-space index of the pedestrians of the frame.",
    )


SERIES_TABLES: Final[dict[str, type[Table]]] = {
    table.table_name.removeprefix(SERIES_PREFIX): table
    for table in registered_tables()
    if table.table_kind is TableKind.SERIES
}
