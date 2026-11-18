"""Robot kinematics: trajectories and velocities from /tf ground truth, odometry drift, and commanded velocity."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry

from arena_evaluation.processing.telemetry.messages import NS_PER_SECOND, Envelope, RobotRef, robot_row, yaw_from_orientation
from arena_evaluation.processing.telemetry.writer import TableSink

POSE_SOURCE = "tf_gt"
TWIST_STAMPED_SCHEMA = "geometry_msgs/msg/TwistStamped"
_WORLD_FRAMES = ("map", "world", "odom")
_GROUND_TRUTH_FRAMES = ("map", "world", "odom")
_BASE_FRAME_MARKERS = ("base_link", "base_footprint")


@dataclasses.dataclass(frozen=True, slots=True)
class Pose3:
    """Position and heading of a ground-truth pose."""

    x: float
    y: float
    z: float
    yaw: float


@dataclasses.dataclass(frozen=True, slots=True)
class _Odometry:
    t_ns: int
    x: float
    y: float
    yaw: float
    vx: float
    wz: float


@dataclasses.dataclass(frozen=True, slots=True)
class _Track:
    t_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    yaw: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    omega_z: np.ndarray


def _named_frame(frame: str, names: tuple[str, ...]) -> bool:
    """Whether a frame name is one of names, possibly prefixed by an environment namespace."""
    return frame in names or frame.endswith(tuple(f"/{name}" for name in names))


def is_world_frame(frame: str) -> bool:
    """Whether a frame name is a world-fixed frame such as map, world or odom, possibly namespaced."""
    return _named_frame(frame, _WORLD_FRAMES)


def is_base_frame(frame: str) -> bool:
    """Whether a frame name is a robot base frame."""
    return any(marker in frame for marker in _BASE_FRAME_MARKERS)


def is_ground_truth(parent: str, child: str) -> bool:
    """Whether a transform is a map, world, or odom to robot-base pose."""
    return _named_frame(parent.strip("/").lower(), _GROUND_TRUTH_FRAMES) and is_base_frame(child.strip("/").lower())


def _command_twist(envelope: Envelope[Twist | TwistStamped]) -> Twist:
    if envelope.schema.name == TWIST_STAMPED_SCHEMA:
        stamped: Any = envelope.body
        return stamped.twist
    raw: Any = envelope.body
    return raw


class KinematicsRecorder:
    """Collects ground-truth poses and odometry per robot, then writes the derived kinematic tables."""

    def __init__(self, sink: TableSink) -> None:
        self._sink = sink
        self._poses: dict[RobotRef, list[tuple[int, Pose3]]] = defaultdict(list)
        self._odometry: dict[tuple[str, RobotRef], list[_Odometry]] = defaultdict(list)

    def record_ground_truth(self, envelope: Envelope[TransformStamped], pose: Pose3) -> None:
        self._poses[envelope.ref].append((envelope.t_ns, pose))

    def record_odometry(self, table: str, envelope: Envelope[Odometry]) -> None:
        odometry = envelope.body
        pose = odometry.pose.pose
        self._odometry[(table, envelope.ref)].append(
            _Odometry(
                t_ns=envelope.t_ns,
                x=float(pose.position.x),
                y=float(pose.position.y),
                yaw=yaw_from_orientation(pose.orientation),
                vx=float(odometry.twist.twist.linear.x),
                wz=float(odometry.twist.twist.angular.z),
            )
        )

    def record_command(self, table: str, envelope: Envelope[Twist | TwistStamped]) -> None:
        twist = _command_twist(envelope)
        self._sink.append(
            table,
            robot_row(
                envelope,
                linear_x=float(twist.linear.x),
                linear_y=float(twist.linear.y),
                linear_z=float(twist.linear.z),
                angular_x=float(twist.angular.x),
                angular_y=float(twist.angular.y),
                angular_z=float(twist.angular.z),
            ),
        )

    def finalize(self) -> None:
        tracks = {ref: _ground_truth_track(samples) for ref, samples in self._poses.items()}
        for ref, track in tracks.items():
            self._write_track(ref, track)
        for (table, ref), samples in self._odometry.items():
            self._write_odometry(table, ref, sorted(samples, key=lambda sample: sample.t_ns), tracks.get(ref))

    def _write_track(self, ref: RobotRef, track: _Track) -> None:
        self._write_rows(
            "robot_tracks",
            ref,
            {
                "t_ns": track.t_ns.tolist(),
                "x": track.x.tolist(),
                "y": track.y.tolist(),
                "z": track.z.tolist(),
                "yaw": track.yaw.tolist(),
                "vx": track.vx.tolist(),
                "vy": track.vy.tolist(),
                "vz": track.vz.tolist(),
                "omega_z": track.omega_z.tolist(),
                "speed_mps": np.hypot(track.vx, track.vy).tolist(),
                "pose_source": [POSE_SOURCE] * len(track.t_ns),
            },
        )

    def _write_odometry(self, table: str, ref: RobotRef, samples: list[_Odometry], track: _Track | None) -> None:
        stamps = np.array([sample.t_ns for sample in samples], dtype=np.int64)
        x = np.array([sample.x for sample in samples], dtype=np.float64)
        y = np.array([sample.y for sample in samples], dtype=np.float64)
        drift = _drift_errors(stamps, x, y, track) if track is not None else [None] * len(samples)
        self._write_rows(
            table,
            ref,
            {
                "t_ns": stamps.tolist(),
                "odom_x": x.tolist(),
                "odom_y": y.tolist(),
                "odom_yaw": [sample.yaw for sample in samples],
                "odom_vx": [sample.vx for sample in samples],
                "odom_wz": [sample.wz for sample in samples],
                "drift_error_m": drift,
            },
        )

    def _write_rows(self, table: str, ref: RobotRef, columns: Mapping[str, Sequence[Any]]) -> None:
        names = tuple(columns)
        for values in zip(*columns.values(), strict=True):
            self._sink.append(table, {"env_id": ref.env_id, "robot_id": ref.robot_id, **dict(zip(names, values, strict=True))})


def _ground_truth_track(samples: list[tuple[int, Pose3]]) -> _Track:
    stamps, first = np.unique(np.array([stamp for stamp, _ in samples], dtype=np.int64), return_index=True)
    poses = [samples[index][1] for index in first]
    x = np.array([pose.x for pose in poses], dtype=np.float64)
    y = np.array([pose.y for pose in poses], dtype=np.float64)
    z = np.array([pose.z for pose in poses], dtype=np.float64)
    yaw = np.array([pose.yaw for pose in poses], dtype=np.float64)
    vx, vy, vz, omega_z = _gradients(stamps, x, y, z, np.unwrap(yaw))
    return _Track(stamps, x, y, z, yaw, vx, vy, vz, omega_z)


def _gradients(stamps: np.ndarray, *series: np.ndarray) -> list[np.ndarray]:
    """Central differences of each series over the native timestamps, zero where there is no interval."""
    if len(stamps) < 2:
        return [np.zeros(len(stamps)) for _ in series]
    seconds = (stamps - stamps[0]) / NS_PER_SECOND
    return [np.gradient(values, seconds) for values in series]


def _drift_errors(stamps: np.ndarray, x: np.ndarray, y: np.ndarray, track: _Track) -> list[float]:
    """Distance between odometry and ground-truth displacements, each measured from its first sample."""
    truth = np.clip(np.searchsorted(track.t_ns, stamps, side="right") - 1, 0, len(track.t_ns) - 1)
    odometry_dx = x - x[0]
    odometry_dy = y - y[0]
    truth_dx = track.x[truth] - track.x[0]
    truth_dy = track.y[truth] - track.y[0]
    return np.hypot(odometry_dx - truth_dx, odometry_dy - truth_dy).tolist()
