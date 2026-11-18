from __future__ import annotations

import json
import math
import pathlib

import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import polars as pl
import yaml
from mcap.writer import Writer as McapWriter
from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader, anchor_at

SEC = 1_000_000_000
SEP = "\n" + "=" * 80 + "\n"

TIME = "MSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec"
HEADER = "MSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id"
POINT = "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z"
QUAT = "MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w"
POSE = "MSG: geometry_msgs/Pose\nPoint position\nQuaternion orientation"
VEC3 = "MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z"
TWIST = "MSG: geometry_msgs/Twist\nVector3 linear\nVector3 angular"
POSE_COV = "MSG: geometry_msgs/PoseWithCovariance\nPose pose"
TWIST_COV = "MSG: geometry_msgs/TwistWithCovariance\nTwist twist"
POSE_STAMPED = "MSG: geometry_msgs/PoseStamped\nstd_msgs/Header header\nPose pose"
PARAM_VALUE = "MSG: rcl_interfaces/ParameterValue\nuint8 type\nbool bool_value\nint64 integer_value\nfloat64 double_value\nstring string_value\nbyte[] byte_array_value\nbool[] bool_array_value\nint64[] integer_array_value\nfloat64[] double_array_value\nstring[] string_array_value"
KEY_VALUE = "MSG: diagnostic_msgs/KeyValue\nstring key\nstring value"
PARAM = "MSG: rcl_interfaces/Parameter\nstring name\nParameterValue value"
TRANSFORM_STAMPED = "MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform"
TRANSFORM = "MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation"
ENTITY = "MSG: arena_runtime_msgs/SemanticEntityState\nstring entity\nstring kind\nstring[] discrete_names\nstring[] discrete_values\nstring[] continuous_names\nfloat64[] continuous_values\nstring[] predicate_names\nbool[] predicate_values\nstring[] members"
PEDESTRIAN = "MSG: arena_people_msgs/Pedestrian\nstring name\nuint32 id\ngeometry_msgs/Pose pose\ngeometry_msgs/Twist twist\nuint8 animation_state\narena_people_msgs/Gesture[] gestures\nint32 interaction_id\nuint8 interaction_type\nstring agent_type\nfloat32 gait_phase\nfloat32 gait_cadence"
GESTURE = "MSG: arena_people_msgs/Gesture\nstring slot\ngeometry_msgs/Point at\nstring clip\nstring hand\nbool render_pose_override"


def _msgdef(body: str, *deps: str) -> str:
    return SEP.join([body, *deps])


ODOMETRY = ("nav_msgs/msg/Odometry", _msgdef("std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist", HEADER, TIME, POSE_COV, POSE, POINT, QUAT, TWIST_COV, TWIST, VEC3))
LASER_SCAN = ("sensor_msgs/msg/LaserScan", _msgdef("std_msgs/Header header\nfloat32 angle_min\nfloat32 angle_max\nfloat32 angle_increment\nfloat32 range_min\nfloat32 range_max\nfloat32[] ranges", HEADER, TIME))
TWIST_MSG = ("geometry_msgs/msg/Twist", _msgdef("geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular", VEC3))
TWIST_STAMPED = ("geometry_msgs/msg/TwistStamped", _msgdef("std_msgs/Header header\ngeometry_msgs/Twist twist", HEADER, TIME, TWIST, VEC3))
JOINT_STATE = ("sensor_msgs/msg/JointState", _msgdef("std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort", HEADER, TIME))
PEDESTRIANS = ("arena_people_msgs/msg/Pedestrians", _msgdef("std_msgs/Header header\nPedestrian[] pedestrians", HEADER, TIME, PEDESTRIAN, GESTURE, POSE, POINT, QUAT, TWIST, VEC3))
AGENT_STATES = (
    "arena_humansim_msgs/msg/AgentStates",
    _msgdef(
        "std_msgs/Header header\nAgentState[] agents",
        HEADER,
        TIME,
        "MSG: arena_humansim_msgs/AgentState\nint32 agent_id\nstring name\nuint8 kind\ngeometry_msgs/Pose2D pose\ngeometry_msgs/Vector3 velocity\nfloat64 desired_velocity\nfloat64 radius\nstring agent_type\nuint8 animation_state\nstring policy",
        "MSG: geometry_msgs/Pose2D\nfloat64 x\nfloat64 y\nfloat64 theta",
        VEC3,
    ),
)
ACOUSTICS = (
    "arena_robots_msgs/msg/Acoustics",
    _msgdef("uint32 FLAG_NO_EFFORT=2\nstd_msgs/Header header\nfloat32 total_level_af_dba\nfloat32 total_level_zf_db\nfloat32 baseline_level_dba\nfloat32 drivetrain_level_dba\nfloat32 uncertainty_1sigma_dba\nuint32 validity_flags\nstring operating_state\nstring calibration_status", HEADER, TIME),
)
STRING = ("std_msgs/msg/String", "string data")
EPISODE_RECORD = (
    "arena_runtime_msgs/msg/EpisodeRecord",
    _msgdef(
        "uint32 episode_id\nstring world\nint64 seed\nstring task_layer\ndiagnostic_msgs/KeyValue[] modes\nstring[] robots\nuint8 RUNNING = 1\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nfloat32 goal_dist_start\nfloat32 goal_dist_min\nfloat32 path_length\nbuiltin_interfaces/Time start_time\nstring[] integrity\nrcl_interfaces/Parameter[] params\nrcl_interfaces/Parameter[] human_params\nstring conditions\nstring phases",
        KEY_VALUE,
        PARAM,
        PARAM_VALUE,
        TIME,
    ),
)
SEMANTIC_SNAPSHOT = ("arena_runtime_msgs/msg/SemanticSnapshot", _msgdef("builtin_interfaces/Time stamp\nuint32 env_id\nstring world\nSemanticEntityState[] entities", TIME, ENTITY))
COLLISION_EVENTS = (
    "arena_robots_msgs/msg/CollisionEvents",
    _msgdef("std_msgs/Header header\narena_robots_msgs/CollisionEvent[] events", HEADER, TIME, 'MSG: arena_robots_msgs/CollisionEvent\nstring KIND_WALL="wall"\nstring kind\nstring obstacle_id\nfloat32 distance\ngeometry_msgs/Point obstacle_position', POINT),
)
COLLISION_MONITOR_STATE = ("nav2_msgs/msg/CollisionMonitorState", "uint8 DO_NOTHING=0\nuint8 action_type\nstring polygon_name")
POWER = (
    "arena_robots_msgs/msg/Power",
    _msgdef("std_msgs/Header header\nfloat32 total_power_w\nfloat32 static_power_w\nfloat32 total_mechanical_power_w\nfloat32 total_thermal_power_w\nstring[] joint_names\nfloat32[] joint_mechanical_power_w\nfloat32[] joint_thermal_power_w\nfloat32[] joint_total_power_w", HEADER, TIME),
)
ENERGY = ("arena_robots_msgs/msg/Energy", _msgdef("std_msgs/Header header\nfloat32 total_energy_consumed_wh\nfloat32 battery_soc_percent", HEADER, TIME))
PATH = ("nav_msgs/msg/Path", _msgdef("std_msgs/Header header\ngeometry_msgs/PoseStamped[] poses", HEADER, TIME, POSE_STAMPED, POSE, POINT, QUAT))
POSE_STAMPED_MSG = ("geometry_msgs/msg/PoseStamped", _msgdef("std_msgs/Header header\ngeometry_msgs/Pose pose", HEADER, TIME, POSE, POINT, QUAT))
POSE_COV_STAMPED = ("geometry_msgs/msg/PoseWithCovarianceStamped", _msgdef("std_msgs/Header header\ngeometry_msgs/PoseWithCovariance pose", HEADER, TIME, POSE_COV, POSE, POINT, QUAT))
TF_MESSAGE = ("tf2_msgs/msg/TFMessage", _msgdef("geometry_msgs/TransformStamped[] transforms", TRANSFORM_STAMPED, HEADER, TIME, TRANSFORM, VEC3, QUAT))

YAW_HALF_PI = {"x": 0.0, "y": 0.0, "z": math.sin(math.pi / 4), "w": math.cos(math.pi / 4)}


def _write_bag(path: pathlib.Path, records: list[tuple[str, tuple[str, str], dict, int]]) -> pathlib.Path:
    with path.open("wb") as fh:
        writer = Writer(fh)
        schemas = {}
        for topic, (name, text), msg, log_time in records:
            if (name, text) not in schemas:
                schemas[name, text] = writer.register_msgdef(name, text)
            writer.write_message(topic, schemas[name, text], msg, log_time=log_time)
        writer.finish()
    return path


def _header(sec: int = 0, nanosec: int = 0, frame_id: str = "") -> dict:
    return {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": frame_id}


def _param(name: str, kind: int, **value: object) -> dict:
    return {"name": name, "value": {"type": kind, **value}}


def _transform(parent: str, child: str, x: float, y: float, yaw: dict | None = None) -> dict:
    return {"header": _header(frame_id=parent), "child_frame_id": child, "transform": {"translation": {"x": x, "y": y, "z": 0.0}, "rotation": yaw or {"w": 1.0}}}


def _schedule(*phases: tuple[str, float]) -> dict:
    return {"data": json.dumps({"phases": [{"name": n, "kind": "ramp", "vx_target": vx, "vy_target": 0.0, "wz_target": 0.0, "duration_s": 2.0, "ramp_s": 0.5, "radius_m": 0.0} for n, vx in phases]})}


def _every_topic_bag(path: pathlib.Path) -> pathlib.Path:
    odom = {"header": _header(1, 5, "odom"), "pose": {"pose": {"position": {"x": 1.0, "y": 2.0}, "orientation": YAW_HALF_PI}}, "twist": {"twist": {"linear": {"x": 0.5}, "angular": {"z": 0.1}}}}
    controller_odom = {"header": _header(1, 0, "odom"), "pose": {"pose": {"position": {"x": 7.0, "y": 8.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    pedestrian = {"name": "alice", "id": 0, "pose": {"position": {"x": 3.0, "y": 4.0, "z": 0.0}, "orientation": YAW_HALF_PI}, "twist": {"linear": {"x": 0.3, "y": -0.1, "z": 0.0}}, "animation_state": 1, "gestures": [], "interaction_id": -1, "interaction_type": 0, "agent_type": "adult", "gait_phase": 0.0, "gait_cadence": 0.0}
    params = [
        _param("jackal.enabled", 1, bool_value=True),
        _param("jackal.count", 2, integer_value=3),
        _param("jackal.radius", 3, double_value=0.25),
        _param("jackal.name", 4, string_value="j"),
        _param("raw", 5, byte_array_value=[1, 2]),
        _param("flags", 6, bool_array_value=[True, False]),
        _param("ints", 7, integer_array_value=[4, 5]),
        _param("floats", 8, double_array_value=[0.5]),
        _param("names", 9, string_array_value=["a", "b"]),
        _param("unset", 0),
    ]
    events = [
        {"kind": "wall", "obstacle_id": "w1", "distance": 0.1, "obstacle_position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        {"kind": "static", "obstacle_id": "s1", "distance": 0.2, "obstacle_position": {"x": 2.0, "y": 3.0, "z": 0.0}},
        {"kind": "pedestrian", "obstacle_id": "p1", "distance": 0.3, "obstacle_position": {"x": 3.0, "y": 4.0, "z": 0.0}},
        {"kind": "robot", "obstacle_id": "w1", "distance": 0.4, "obstacle_position": {"x": 4.0, "y": 5.0, "z": 0.0}},
    ]
    records = [
        ("/env_0/jackal/odom", ODOMETRY, odom, 1 * SEC),
        ("/env_0/jackal/diff_drive_controller/odom", ODOMETRY, controller_odom, 1 * SEC),
        ("/env_0/burger/odom", ODOMETRY, controller_odom, 1 * SEC),
        ("/env_0/odom", ODOMETRY, controller_odom, 1 * SEC),
        ("/env_0/jackal/scan", LASER_SCAN, {"range_min": 0.1, "range_max": 10.0, "ranges": [0.05, 2.0, 1.5, 30.0]}, 2 * SEC),
        ("/env_0/jackal/lidar", LASER_SCAN, {"range_min": 0.1, "range_max": 10.0, "ranges": [11.0]}, 3 * SEC),
        ("/env_0/jackal/cmd_vel", TWIST_MSG, {"linear": {"x": 0.4, "y": 0.1}, "angular": {"z": -0.2}}, 2 * SEC),
        ("/env_0/jackal/diff_drive_controller/cmd_vel", TWIST_STAMPED, {"header": _header(), "twist": {"linear": {"x": 9.0}}}, 2 * SEC),
        ("/env_0/burger/diff_drive_controller/cmd_vel", TWIST_STAMPED, {"header": _header(), "twist": {"linear": {"x": 0.7}, "angular": {"z": 0.3}}}, 2 * SEC),
        ("/env_0/jackal/joint_states", JOINT_STATE, {"header": _header(), "name": ["left", "right"], "position": [0.1, 0.2], "velocity": [1.0, 1.1], "effort": [0.0, 0.5]}, 2 * SEC),
        ("/env_0/peds", PEDESTRIANS, {"header": _header(frame_id="map"), "pedestrians": [pedestrian]}, 2 * SEC),
        ("/env_0/agent_states", AGENT_STATES, {"header": _header(frame_id="map"), "agents": [{"agent_id": 4, "kind": 0, "pose": {"x": 9.0, "y": 9.0, "theta": 0.0}, "velocity": {}}]}, 2 * SEC),
        ("/env_0/jackal/acoustics", ACOUSTICS, {"header": _header(), "total_level_af_dba": 55.5, "total_level_zf_db": 60.0, "baseline_level_dba": 40.0, "drivetrain_level_dba": 50.0, "uncertainty_1sigma_dba": 2.0, "validity_flags": 2, "operating_state": "driving", "calibration_status": "nominal"}, 2 * SEC),
        ("/env_0/jackal/characterization_phase", STRING, {"data": "ramp_up"}, 2 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, _schedule(("a", 0.2), ("b", 0.4)), 2 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, {"data": "not json"}, 3 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, _schedule(("c", 0.6)), 4 * SEC),
        (
            "/env_0/state/episode",
            EPISODE_RECORD,
            {
                "episode_id": 7,
                "world": "test_world",
                "seed": 42,
                "task_layer": "standard",
                "modes": [],
                "robots": ["jackal"],
                "outcome_state": 2,
                "outcome_info": "done",
                "goal_uuid": "g",
                "goal_dist_start": 5.0,
                "goal_dist_min": 0.1,
                "path_length": 6.2,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": [],
                "params": params,
                "human_params": [],
                "conditions": "",
                "phases": '{"jackal": {"map_poses": []}}',
            },
            5 * SEC,
        ),
        (
            "/env_0/state/episode",
            EPISODE_RECORD,
            {
                "episode_id": 7,
                "world": "test_world",
                "seed": 42,
                "task_layer": "standard",
                "modes": [],
                "robots": ["jackal"],
                "outcome_state": 3,
                "outcome_info": "failed",
                "goal_uuid": "g",
                "goal_dist_start": 5.0,
                "goal_dist_min": 0.1,
                "path_length": 6.2,
                "start_time": {"sec": 1, "nanosec": 0},
                "integrity": [],
                "params": [],
                "human_params": [],
                "conditions": "",
                "phases": "",
            },
            6 * SEC,
        ),
        ("/env_1/agent_states", AGENT_STATES, {"header": _header(frame_id="map"), "agents": [{"agent_id": 5, "kind": 0, "pose": {"x": 1.0, "y": 2.0, "theta": 0.5}, "velocity": {"x": 0.1}}]}, 2 * SEC),
        (
            "/arena/state/semantics",
            SEMANTIC_SNAPSHOT,
            {"env_id": 0, "world": "w", "entities": [{"entity": "env_0/door_1", "kind": "door", "discrete_names": ["state"], "discrete_values": ["open"], "continuous_names": ["angle"], "continuous_values": [1.5], "predicate_names": ["locked"], "predicate_values": [True], "members": ["ped_1"]}]},
            1 * SEC,
        ),
        ("/env_0/jackal/collision_events", COLLISION_EVENTS, {"header": _header(), "events": events}, 3 * SEC),
        ("/env_0/jackal/collision_events", COLLISION_EVENTS, {"header": _header(), "events": []}, 4 * SEC),
        ("/env_0/burger/collision_events", COLLISION_EVENTS, {"header": _header(), "events": [{"kind": "wall", "obstacle_id": "x", "distance": 0.5, "obstacle_position": {"x": 0.0, "y": 0.0, "z": 0.0}}]}, 3 * SEC),
        ("/env_0/jackal/collision_monitor_state", COLLISION_MONITOR_STATE, {"action_type": 2, "polygon_name": "stop_zone"}, 3 * SEC),
        ("/env_0/jackal/power_publisher/power", POWER, {"header": _header(), "total_power_w": 12.0, "static_power_w": 5.0, "total_mechanical_power_w": 6.0, "total_thermal_power_w": 1.0, "joint_names": ["left"], "joint_mechanical_power_w": [6.0], "joint_thermal_power_w": [1.0], "joint_total_power_w": [7.0]}, 3 * SEC),
        ("/env_0/jackal/power_publisher/energy", ENERGY, {"header": _header(), "total_energy_consumed_wh": 0.5, "battery_soc_percent": 99.0}, 3 * SEC),
        ("/env_0/jackal/plan", PATH, {"header": _header(), "poses": [{"header": _header(), "pose": {"position": {"x": 0.0, "y": 0.0}, "orientation": {"w": 1.0}}}, {"header": _header(), "pose": {"position": {"x": 1.0, "y": 1.0}, "orientation": YAW_HALF_PI}}]}, 3 * SEC),
        ("/env_0/jackal/task_pose", POSE_STAMPED_MSG, {"header": _header(2, 0), "pose": {"position": {"x": 1.5, "y": 2.5}, "orientation": YAW_HALF_PI}}, 3 * SEC),
        ("/env_0/jackal/initialpose", POSE_COV_STAMPED, {"header": _header(), "pose": {"pose": {"position": {"x": 0.5, "y": 0.25}, "orientation": {"w": 1.0}}}}, 3 * SEC),
        ("/env_0/jackal/unrecognized", STRING, {"data": "ignored"}, 3 * SEC),
        ("/tf", TF_MESSAGE, {"transforms": [_transform("map", "env_0/jackal/base_link", 4.0, 5.0, YAW_HALF_PI), _transform("map", "env_0/map", 0.0, 0.0), _transform("env_0/jackal/odom", "env_0/jackal/base_link", 9.0, 9.0)]}, 4 * SEC),
        ("/tf", TF_MESSAGE, {"transforms": []}, 5 * SEC),
        ("/tf_static", TF_MESSAGE, {"transforms": [_transform("map", "env_0/map", 0.0, 0.0), _transform("map", "env_0/jackal/base_footprint", 0.0, 0.0), _transform("world", "camera", 1.0, 1.0)]}, 0),
    ]
    return _write_bag(path, records)


@pytest.fixture(scope="module")
def every_topic(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, dict]:
    root = tmp_path_factory.mktemp("every_topic")
    bag = _every_topic_bag(root / "run.mcap")
    out = root / "topics"
    return out, MCAPReader(bag).read(out)


def test_reader_names_robots_from_topic_namespaces(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    assert set(bundles) == {"env_0_jackal", "env_0_burger", "env_0"}


def test_odometry_prefers_the_controller_topic(every_topic: tuple[pathlib.Path, dict]):
    out, bundles = every_topic

    controller = bundles["env_0_jackal"].robot_odom.collect()
    raw = pl.read_parquet(out / "robot_odom.parquet").filter(pl.col("robot_id") == "env_0_jackal")
    assert controller["odom_x"].to_list() == [7.0]
    assert raw.row(0, named=True) == {
        "t_ns": SEC,
        "env_id": "env_0",
        "robot_id": "env_0_jackal",
        "odom_x": 1.0,
        "odom_y": 2.0,
        "odom_yaw": pytest.approx(math.pi / 2),
        "odom_vx": 0.5,
        "odom_wz": 0.1,
        "drift_error_m": 0.0,
    }


def test_scan_and_lidar_share_one_table_with_the_nearest_valid_return(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    summary = bundles["env_0_jackal"].robot_scan_summary.collect().sort("t_ns")
    assert summary["scan_min_m"].to_list() == [1.5, 10.0]
    assert summary["scan_max_m"].to_list() == [2.0, 10.0]
    assert summary["n_valid_rays"].to_list() == [2, 0]


def test_cmd_vel_reads_twist_and_falls_back_to_the_controller_twist_stamped(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    jackal = bundles["env_0_jackal"].robot_cmd_vel.collect().row(0, named=True)
    burger = bundles["env_0_burger"].robot_cmd_vel.collect().row(0, named=True)
    assert (jackal["linear_x"], jackal["linear_y"], jackal["angular_z"]) == pytest.approx((0.4, 0.1, -0.2))
    assert (burger["linear_x"], burger["angular_z"]) == pytest.approx((0.7, 0.3))


def test_joint_states_become_one_row_per_joint(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    joints = bundles["env_0_jackal"].robot_joint_states.collect().sort("t_ns")
    assert joints["joint_name"].to_list() == ["left", "right"]
    assert joints["velocity"].to_list() == pytest.approx([1.0, 1.1])
    assert joints["effort"].to_list() == pytest.approx([0.0, 0.5])


def test_arena_peds_win_over_engine_agent_states(every_topic: tuple[pathlib.Path, dict]):
    out, bundles = every_topic

    peds = bundles["env_0_jackal"].pedestrian_arena_peds.collect()
    physics = pl.read_parquet(out / "pedestrian_agent_states.parquet")
    assert peds.select("t_ns", "env_id", "ped_id", "ped_name", "x", "y", "yaw", "vx", "vy").to_dicts() == [
        {"t_ns": 2 * SEC, "env_id": "env_0", "ped_id": "Pedestrian_0", "ped_name": "alice", "x": 3.0, "y": 4.0, "yaw": pytest.approx(math.pi / 2), "vx": 0.3, "vy": -0.1},
    ]
    assert physics["ped_id"].to_list() == ["Pedestrian_4", "Pedestrian_5"]


def test_acoustics_and_characterization_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    acoustics = jackal.robot_acoustics.collect().row(0, named=True)
    assert acoustics["total_level_af_dba"] == pytest.approx(55.5)
    assert acoustics["validity_flags"] == 2
    assert jackal.robot_characterization_phase.collect()["label"].to_list() == ["ramp_up"]
    schedule = jackal.robot_characterization_schedule.collect().drop(["t_ns", "env_id", "robot_id"]).to_dicts()
    assert schedule == [{"phase_idx": 0, "phase_label": "c", "phase_kind": "ramp", "vx_target": 0.6, "vy_target": 0.0, "wz_target": 0.0, "duration_s": 2.0, "ramp_s": 0.5, "turn_radius_m": 0.0}]


def test_episode_record_stores_the_full_name_params_as_yaml(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    records = bundles["env_0_jackal"].episode_lifecycle.collect()
    record = records.row(0, named=True)
    params = yaml.safe_load(record["params_yaml"])
    assert records["outcome_state"].to_list() == [2, 3]
    assert (record["episode_id"], record["outcome_state"], record["outcome_info"], record["goal_uuid"]) == (7, 2, "done", "g")
    assert record["phases_yaml"] == '{"jackal": {"map_poses": []}}'
    assert {name: value for name, value in params.items() if name.startswith("jackal.")} == {"jackal.enabled": True, "jackal.count": 3, "jackal.radius": 0.25, "jackal.name": "j"}
    assert params["raw"] == [1, 2]
    assert params["flags"] == [True, False]
    assert params["ints"] == [4, 5]
    assert params["floats"] == [0.5]
    assert params["names"] == ["a", "b"]
    assert params["unset"] is None


def test_semantic_snapshot_flattens_each_entity_field_into_a_row(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    rows = bundles["env_0_jackal"].world_semantics.collect().select("field_name", "field_kind", "value_str", "value_num", "value_bool").to_dicts()
    assert rows == [
        {"field_name": "state", "field_kind": "discrete", "value_str": "open", "value_num": None, "value_bool": None},
        {"field_name": "angle", "field_kind": "continuous", "value_str": None, "value_num": 1.5, "value_bool": None},
        {"field_name": "locked", "field_kind": "predicate", "value_str": None, "value_num": None, "value_bool": True},
        {"field_name": "members", "field_kind": "members", "value_str": "ped_1", "value_num": None, "value_bool": None},
    ]


def test_collision_events_count_kinds(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    events = bundles["env_0_jackal"].robot_collision_events.collect()
    assert events.select("event_count", "wall_count", "static_count", "ped_count").rows() == [(4, 1, 1, 1), (0, 0, 0, 0)]
    burger = bundles["env_0_burger"].robot_collision_events.collect().row(0, named=True)
    assert (burger["event_count"], burger["wall_count"], burger["static_count"], burger["ped_count"]) == (1, 1, 0, 0)
    assert sorted(set(bundles["env_0_jackal"].robot_collision_contacts.collect()["obstacle_id"].to_list())) == ["p1", "s1", "w1"]


def test_monitor_power_energy_and_plan_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    assert jackal.robot_collision_monitor.collect().select("action_type", "polygon_name").row(0) == (2, "stop_zone")
    power = jackal.robot_power.collect().row(0, named=True)
    assert (power["total_power_w"], power["total_mech_power_w"], power["total_thermal_power_w"]) == (12.0, 6.0, 1.0)
    assert jackal.robot_joint_power.collect().select("joint_name", "total_power_w").row(0) == ("left", 7.0)
    assert jackal.robot_energy.collect().select("total_energy_consumed_wh", "battery_soc_percent").row(0) == (0.5, 99.0)
    plan = jackal.robot_plans.collect().sort("waypoint_idx")
    assert plan["x"].to_list() == [0.0, 1.0]
    assert plan["yaw"].to_list() == pytest.approx([0.0, math.pi / 2])


def test_task_pose_and_initialpose_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    task_pose = jackal.robot_task_pose.collect().row(0, named=True)
    assert (task_pose["stamp_ns"], task_pose["pos_x"], task_pose["pos_y"]) == (2 * SEC, 1.5, 2.5)
    assert task_pose["yaw"] == pytest.approx(math.pi / 2)
    assert jackal.robot_initialpose.collect().select("pos_x", "pos_y", "yaw").row(0) == (0.5, 0.25, 0.0)


def test_tf_keeps_every_transform_and_extracts_map_to_base_ground_truth(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    assert len(jackal.tf.collect()) == 3
    assert len(jackal.tf_static.collect()) == 3
    tracks = jackal.robot_tracks.collect()
    assert tracks.select("t_ns", "x", "y", "yaw", "pose_source").rows() == [(0, 0.0, 0.0, 0.0, "tf_gt"), (4 * SEC, 4.0, 5.0, pytest.approx(math.pi / 2), "tf_gt")]


@pytest.mark.parametrize("topic", ["/odom", "/cmd_vel"])
def test_un_namespaced_topic_is_refused_rather_than_filed_under_a_guessed_robot(tmp_path: pathlib.Path, topic: str):
    odom = {"header": _header(), "pose": {"pose": {"position": {"x": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    message = (ODOMETRY, odom) if topic == "/odom" else (TWIST_MSG, {"linear": {"x": 2.0}})
    bag = _write_bag(tmp_path / "bare.mcap", [(topic, message[0], message[1], SEC)])

    with pytest.raises(ValueError, match="names no environment"):
        MCAPReader(bag).read(tmp_path / "out")


@pytest.mark.parametrize("node", ["task_generator_node", "arena_env_node"])
def test_a_robot_under_the_old_or_new_env_node_namespace_reads_the_same(tmp_path: pathlib.Path, node: str):
    odom = {"header": _header(), "pose": {"pose": {"position": {"x": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    bag = _write_bag(tmp_path / "imported.mcap", [(f"/arena/env_0/{node}/human/odom", ODOMETRY, odom, SEC)])

    bundles = MCAPReader(bag).read(tmp_path / "out")

    assert set(bundles) == {"env_0_human"}


def test_reader_rejects_a_directory_without_bags(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="No MCAP file"):
        MCAPReader(tmp_path).read(tmp_path / "out")


def test_reader_rejects_a_missing_bag(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        MCAPReader(tmp_path / "missing.mcap").read(tmp_path / "out")


def _foreign_bag(path: pathlib.Path) -> pathlib.Path:
    with path.open("wb") as fh:
        writer_ = McapWriter(fh)
        writer_.start()
        json_schema = writer_.register_schema(name="Status", encoding="jsonschema", data=b"{}")
        json_channel = writer_.register_channel(topic="/env_0/jackal/odom", message_encoding="json", schema_id=json_schema)
        writer_.add_message(channel_id=json_channel, log_time=SEC, publish_time=SEC, data=b"{}")
        cdr_schema = writer_.register_schema(name="nav_msgs/msg/Odometry", encoding="ros2msg", data=ODOMETRY[1].encode())
        cdr_channel = writer_.register_channel(topic="/env_0/burger/odom", message_encoding="cdr", schema_id=cdr_schema)
        writer_.add_message(channel_id=cdr_channel, log_time=SEC, publish_time=SEC, data=b"\x00\x01")
        writer_.finish()
    return path


def test_directory_of_bags_skips_undecodable_and_corrupt_messages(tmp_path: pathlib.Path):
    bags = tmp_path / "bags"
    bags.mkdir()
    _foreign_bag(bags / "a.mcap")
    odom = {"header": _header(3, 0), "pose": {"pose": {"position": {"x": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    _write_bag(bags / "b.mcap", [("/env_0/robot/odom", ODOMETRY, odom, 3 * SEC)])

    bundles = MCAPReader(bags).read(tmp_path / "out")

    assert set(bundles) == {"env_0_robot"}
    assert bundles["env_0_robot"].robot_odom.collect()["odom_x"].to_list() == [2.0]


def test_global_and_env_writers_append_across_buffer_flushes(tmp_path: pathlib.Path):
    peds = {"header": _header(frame_id="map"), "pedestrians": [{"name": "p", "id": 1, "pose": {"position": {"x": 1.0}, "orientation": {"w": 1.0}}, "twist": {}, "animation_state": 0, "gestures": [], "interaction_id": -1, "interaction_type": 0, "agent_type": "adult", "gait_phase": 0.0, "gait_cadence": 0.0}]}
    tf = {"transforms": [_transform("map", "env_0/map", 0.0, 0.0)]}
    odom = {"header": _header(), "pose": {"pose": {"position": {}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    records = [("/env_0/jackal/odom", ODOMETRY, odom, 0)]
    for i in range(5001):
        records.append(("/tf", TF_MESSAGE, tf, i + 1))
        records.append(("/env_0/peds", PEDESTRIANS, peds, i + 1))
    records.append(("/env_0/jackal/odom", ODOMETRY, odom, 6000))

    bundles = MCAPReader(_write_bag(tmp_path / "long.mcap", records), flush_rows=1_000).read(tmp_path / "out")

    assert len(bundles["env_0_jackal"].tf.collect()) == 5001
    assert len(bundles["env_0_jackal"].pedestrian_arena_peds.collect()) == 5001
    assert len(bundles["env_0_jackal"].robot_odom.collect()) == 2


def test_anchor_at_picks_the_last_anchor_not_after_the_time():
    anchors = [(0, 1.0, 1.0), (10, 2.0, 2.0), (20, 3.0, 3.0)]

    assert anchor_at(anchors, -5) == (1.0, 1.0)
    assert anchor_at(anchors, 10) == (2.0, 2.0)
    assert anchor_at(anchors, 15) == (2.0, 2.0)
    assert anchor_at(anchors, 99) == (3.0, 3.0)
