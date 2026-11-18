"""Robot sensor and power telemetry: scans, joint states, power, energy and acoustics as flat rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from arena_evaluation.processing.telemetry.messages import Envelope, robot_row, value_at
from arena_evaluation.processing.telemetry.writer import TableSink

if TYPE_CHECKING:
    from arena_robots_msgs.msg import Acoustics, Energy, Power
    from sensor_msgs.msg import JointState, LaserScan

SCAN_SUMMARY_TABLE = "robot_scan_summary"
SCAN_RAY_TABLE = "robot_scan_rays"


class ScanRecorder:
    """Splits each LaserScan into one summary row and, when requested, one row per ray."""

    def __init__(self, sink: TableSink, include_rays: bool) -> None:
        self._sink = sink
        self._include_rays = include_rays

    def record(self, envelope: Envelope[LaserScan]) -> None:
        scan = envelope.body
        out_of_range = float(scan.range_max)
        valid = [float(distance) for distance in scan.ranges if scan.range_min <= distance < scan.range_max]
        self._sink.append(
            SCAN_SUMMARY_TABLE,
            robot_row(
                envelope,
                scan_min_m=min(valid, default=out_of_range),
                scan_max_m=max(valid, default=out_of_range),
                range_min_m=float(scan.range_min),
                range_max_m=out_of_range,
                n_valid_rays=len(valid),
            ),
        )
        if self._include_rays:
            self._record_rays(envelope)

    def _record_rays(self, envelope: Envelope[LaserScan]) -> None:
        scan = envelope.body
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            self._sink.append(SCAN_RAY_TABLE, robot_row(envelope, ray_idx=index, range_m=float(distance), angle_rad=float(angle)))


def record_joint_states(sink: TableSink, envelope: Envelope[JointState]) -> None:
    state = envelope.body
    for index, name in enumerate(state.name):
        sink.append(
            "robot_joint_states",
            robot_row(
                envelope,
                joint_name=str(name),
                position=value_at(state.position, index),
                velocity=value_at(state.velocity, index),
                effort=value_at(state.effort, index),
            ),
        )


def record_power(sink: TableSink, envelope: Envelope[Power]) -> None:
    power = envelope.body
    sink.append(
        "robot_power",
        robot_row(
            envelope,
            total_power_w=float(power.total_power_w),
            static_power_w=float(power.static_power_w),
            total_mech_power_w=float(power.total_mechanical_power_w),
            total_thermal_power_w=float(power.total_thermal_power_w),
        ),
    )
    for index, name in enumerate(power.joint_names):
        sink.append(
            "robot_joint_power",
            robot_row(
                envelope,
                joint_name=str(name),
                mechanical_power_w=value_at(power.joint_mechanical_power_w, index),
                thermal_power_w=value_at(power.joint_thermal_power_w, index),
                total_power_w=value_at(power.joint_total_power_w, index),
            ),
        )


def record_energy(sink: TableSink, envelope: Envelope[Energy]) -> None:
    energy = envelope.body
    sink.append(
        "robot_energy",
        robot_row(
            envelope,
            total_energy_consumed_wh=float(energy.total_energy_consumed_wh),
            battery_soc_percent=float(energy.battery_soc_percent),
        ),
    )


def record_acoustics(sink: TableSink, envelope: Envelope[Acoustics]) -> None:
    acoustics = envelope.body
    sink.append(
        "robot_acoustics",
        robot_row(
            envelope,
            total_level_af_dba=float(acoustics.total_level_af_dba),
            total_level_zf_db=float(acoustics.total_level_zf_db),
            baseline_level_dba=float(acoustics.baseline_level_dba),
            drivetrain_level_dba=float(acoustics.drivetrain_level_dba),
            operating_state=str(acoustics.operating_state),
            calibration_status=str(acoustics.calibration_status),
            uncertainty_dba=float(acoustics.uncertainty_1sigma_dba),
            validity_flags=int(acoustics.validity_flags),
        ),
    )
