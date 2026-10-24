from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

BEAMS = 16
STEP_NS = 100_000_000
STAMP_LAG_NS = 20_000_000
CMD_OFFSET_NS = 5_000_000
RANGE_MAX = 10.0
COUNT = 40
T0_NS = 1_700_000_000_000_000_000
HEIGHT = 1.4

SCAN = "/scan"
ODOM = "/zed2/zed_node/odom"
VELOCITY = "/action"
CMD_VEL = "/cmd_vel"
IMU = "/zed2/zed_node/imu/data"
POINTS = "/velodyne_points"
PATH = "/zed2/zed_node/path_odom"
LOG = "/rosout"
CAMERA = "/zed2/zed_node/left/camera_info"
MARKERS = "/zed2/zed_node/markers"
DISTORTION = (0.1, -0.2, 0.0, 0.0, 0.05)
FOCAL = 525.0
CLOUD_POINTS = 4
ROS1_DEFINITIONS = {
    "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms\n",
    "rosgraph_msgs/msg/Log": "byte DEBUG=1\nbyte INFO=2\nbyte WARN=4\nbyte ERROR=8\nbyte FATAL=16\nHeader header\nbyte level\nstring name\nstring msg\nstring file\nstring function\nuint32 line\nstring[] topics\n",
}


def scan_ranges(i: int) -> np.ndarray:
    ranges = 1.0 + 0.1 * i + 0.05 * np.arange(BEAMS, dtype=np.float32)
    ranges[0] = np.inf
    return ranges.astype(np.float32)


def pose(i: int) -> tuple[float, float, float]:
    return (0.1 * i, 0.0, 0.05 * i)


def velocity(i: int) -> tuple[float, float, float]:
    return (1.0 + 0.001 * i, -0.1, 0.5)


def cmd_vel(i: int) -> tuple[float, float]:
    return (0.4 + 0.001 * i, 0.2)


def cloud_data(i: int) -> bytes:
    return np.arange(CLOUD_POINTS * 3, dtype=np.float32).tobytes() + bytes([i % 256])


def write_bag(path: Path, count: int = COUNT, *, unpaired: tuple[int, ...] = (), cmd: bool = False, stray: int = 0, twist_in_odom: bool = False, sensors: bool = False, markers: bool = False) -> Path:
    """A rosbag1 file in the measured MuSoHu layout: odom with pose only, the velocity on /action under the same header stamp."""
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    LaserScan = ts.types["sensor_msgs/msg/LaserScan"]
    Odometry = ts.types["nav_msgs/msg/Odometry"]
    Twist = ts.types["geometry_msgs/msg/Twist"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    Pose = ts.types["geometry_msgs/msg/Pose"]
    Point = ts.types["geometry_msgs/msg/Point"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    PoseWithCovariance = ts.types["geometry_msgs/msg/PoseWithCovariance"]
    TwistWithCovariance = ts.types["geometry_msgs/msg/TwistWithCovariance"]
    Marker = ts.types["visualization_msgs/msg/Marker"]
    MarkerArray = ts.types["visualization_msgs/msg/MarkerArray"]
    ColorRGBA = ts.types["std_msgs/msg/ColorRGBA"]
    Duration = ts.types["builtin_interfaces/msg/Duration"]

    def header(i: int, t: int, frame: str) -> object:
        stamp = t - STAMP_LAG_NS
        return Header(seq=i, stamp=Time(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000), frame_id=frame)

    def twist(vx: float, vy: float, wz: float) -> object:
        return Twist(linear=Vector3(x=vx, y=vy, z=0.0), angular=Vector3(x=0.0, y=0.0, z=wz))

    def odometry(i: int, t: int, frame: str, child: str, xyz: tuple[float, float, float], quat: tuple[float, float, float, float], vel: tuple[float, float, float]) -> object:
        return Odometry(
            header=header(i, t, frame),
            child_frame_id=child,
            pose=PoseWithCovariance(pose=Pose(position=Point(x=xyz[0], y=xyz[1], z=xyz[2]), orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])), covariance=np.zeros(36)),
            twist=TwistWithCovariance(twist=twist(*vel), covariance=np.zeros(36)),
        )

    with Writer(path) as writer:
        scan_conn = writer.add_connection(SCAN, LaserScan.__msgtype__, typestore=ts)
        odom_conn = writer.add_connection(ODOM, Odometry.__msgtype__, typestore=ts)
        velocity_conn = None if twist_in_odom else writer.add_connection(VELOCITY, Odometry.__msgtype__, typestore=ts)
        cmd_conn = writer.add_connection(CMD_VEL, Twist.__msgtype__, typestore=ts) if cmd or stray else None
        marker_conn = writer.add_connection(MARKERS, MarkerArray.__msgtype__, typestore=ts) if markers else None
        if sensors:
            for name, text in ROS1_DEFINITIONS.items():
                if name not in ts.types:
                    ts.register(get_types_from_msg(text, name))
            Imu = ts.types["sensor_msgs/msg/Imu"]
            PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
            PointField = ts.types["sensor_msgs/msg/PointField"]
            TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
            TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
            Transform = ts.types["geometry_msgs/msg/Transform"]
            PoseStamped = ts.types["geometry_msgs/msg/PoseStamped"]
            NavPath = ts.types["nav_msgs/msg/Path"]
            Log = ts.types["rosgraph_msgs/msg/Log"]
            CameraInfo = ts.types["sensor_msgs/msg/CameraInfo"]
            RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]
        sensor_conns = (
            {
                IMU: writer.add_connection(IMU, Imu.__msgtype__, typestore=ts),
                POINTS: writer.add_connection(POINTS, PointCloud2.__msgtype__, typestore=ts),
                PATH: writer.add_connection(PATH, NavPath.__msgtype__, typestore=ts),
                "/tf": writer.add_connection("/tf", TFMessage.__msgtype__, typestore=ts),
                "/tf_static": writer.add_connection("/tf_static", TFMessage.__msgtype__, typestore=ts),
                LOG: writer.add_connection(LOG, Log.__msgtype__, typestore=ts),
                CAMERA: writer.add_connection(CAMERA, CameraInfo.__msgtype__, typestore=ts),
            }
            if sensors
            else {}
        )
        if sensors:
            mount = TransformStamped(header=header(0, T0_NS, "base_link"), child_frame_id="velodyne", transform=Transform(translation=Vector3(x=0.0, y=0.0, z=0.5), rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))
            writer.write(sensor_conns["/tf_static"], T0_NS, ts.serialize_ros1(TFMessage(transforms=[mount]), TFMessage.__msgtype__))
        for i in range(count):
            t = T0_NS + i * STEP_NS
            scan = LaserScan(
                header=header(i, t, "lidar_link"),
                angle_min=-math.pi,
                angle_max=math.pi,
                angle_increment=2 * math.pi / BEAMS,
                time_increment=0.0,
                scan_time=0.0,
                range_min=0.0,
                range_max=RANGE_MAX,
                ranges=scan_ranges(i),
                intensities=np.zeros(BEAMS, dtype=np.float32),
            )
            writer.write(scan_conn, t, ts.serialize_ros1(scan, LaserScan.__msgtype__))
            x, y, yaw = pose(i)
            odom = odometry(i, t, "odom", "base_link", (x, y, HEIGHT), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)), velocity(i) if twist_in_odom else (0.0, 0.0, 0.0))
            writer.write(odom_conn, t, ts.serialize_ros1(odom, Odometry.__msgtype__))
            if velocity_conn is not None and i not in unpaired:
                action = odometry(i, t, "action", "", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), velocity(i))
                writer.write(velocity_conn, t + CMD_OFFSET_NS, ts.serialize_ros1(action, Odometry.__msgtype__))
            if cmd_conn is not None:
                vx, wz = cmd_vel(i)
                writer.write(cmd_conn, t + CMD_OFFSET_NS, ts.serialize_ros1(twist(vx, 0.0, wz), Twist.__msgtype__))
            if marker_conn is not None:
                x, y, _ = pose(i)
                marker = Marker(
                    header=header(i, t, "odom"),
                    ns="trail",
                    id=i,
                    type=2,
                    action=0,
                    pose=Pose(position=Point(x=x, y=y, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
                    scale=Vector3(x=0.1, y=0.1, z=0.1),
                    color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                    lifetime=Duration(sec=0, nanosec=0),
                    frame_locked=False,
                    points=[],
                    colors=[],
                    text="",
                    mesh_resource="",
                    mesh_use_embedded_materials=False,
                )
                writer.write(marker_conn, t + 7, ts.serialize_ros1(MarkerArray(markers=[marker]), MarkerArray.__msgtype__))
            if sensors:
                imu = Imu(
                    header=header(i, t, "zed2_imu_link"),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    orientation_covariance=np.zeros(9),
                    angular_velocity=Vector3(x=0.0, y=0.0, z=velocity(i)[2]),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=Vector3(x=0.0, y=0.0, z=9.81),
                    linear_acceleration_covariance=np.zeros(9),
                )
                writer.write(sensor_conns[IMU], t + 1, ts.serialize_ros1(imu, Imu.__msgtype__))
                fields = [PointField(name=axis, offset=4 * k, datatype=7, count=1) for k, axis in enumerate("xyz")]
                cloud = PointCloud2(header=header(i, t, "velodyne"), height=1, width=CLOUD_POINTS, fields=fields, is_bigendian=False, point_step=12, row_step=12 * CLOUD_POINTS, data=np.frombuffer(cloud_data(i), dtype=np.uint8), is_dense=True)
                writer.write(sensor_conns[POINTS], t + 2, ts.serialize_ros1(cloud, PointCloud2.__msgtype__))
                x, y, _ = pose(i)
                odom_to_base = TransformStamped(header=header(i, t, "odom"), child_frame_id="base_link", transform=Transform(translation=Vector3(x=x, y=y, z=0.0), rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))
                writer.write(sensor_conns["/tf"], t + 3, ts.serialize_ros1(TFMessage(transforms=[odom_to_base]), TFMessage.__msgtype__))
                trail = NavPath(header=header(i, t, "odom"), poses=[PoseStamped(header=header(j, T0_NS + j * STEP_NS, "odom"), pose=Pose(position=Point(x=pose(j)[0], y=0.0, z=0.0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))) for j in range(i + 1)])
                writer.write(sensor_conns[PATH], t + 4, ts.serialize_ros1(trail, NavPath.__msgtype__))
                log = Log(header=header(i, t, ""), level=2, name="/zed2", msg="frame", file="", function="", line=0, topics=[])
                writer.write(sensor_conns[LOG], t + 5, ts.serialize_ros1(log, Log.__msgtype__))
                camera = CameraInfo(
                    header=header(i, t, "zed2_left_camera_optical_frame"),
                    height=360,
                    width=640,
                    distortion_model="plumb_bob",
                    D=np.array(DISTORTION),
                    K=np.array([FOCAL, 0.0, 320.0, 0.0, FOCAL, 180.0, 0.0, 0.0, 1.0]),
                    R=np.eye(3).ravel(),
                    P=np.array([FOCAL, 0.0, 320.0, 0.0, 0.0, FOCAL, 180.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                    binning_x=0,
                    binning_y=0,
                    roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
                )
                writer.write(sensor_conns[CAMERA], t + 6, ts.serialize_ros1(camera, CameraInfo.__msgtype__))
        for k in range(stray):
            t = T0_NS + (count + 2 + k) * STEP_NS
            writer.write(cmd_conn, t, ts.serialize_ros1(twist(0.0, 0.0, 0.0), Twist.__msgtype__))
    return path


def listed(cache: Path, *items: str) -> Path:
    """A fresh Dataverse dataset record in cache naming one bag per item."""
    files = [{"dataFile": {"id": index, "filename": f"{item}.bag", "filesize": 0}} for index, item in enumerate(items)]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "dataset.json"
    path.write_text(json.dumps({"data": {"latestVersion": {"files": files}}}))
    return path


def convert(bag: Path, out: Path, **topics: str | None) -> Path:
    pytest.importorskip("rosbag2_py")
    from arena_evaluation.adapters.musohu import MuSoHu, Topics

    listed(out.parent / "cache", bag.stem)
    importer = MuSoHu(cache=out.parent / "cache", topics=Topics(**topics))
    return out / importer.convert(bag.stem, bag, out)
