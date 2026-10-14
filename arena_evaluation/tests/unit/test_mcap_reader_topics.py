from __future__ import annotations

import json
import math
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import pyarrow.parquet as pq
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
PARAM = "MSG: rcl_interfaces/Parameter\nstring name\nParameterValue value"
TRANSFORM_STAMPED = "MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform"
TRANSFORM = "MSG: geometry_msgs/Transform\nVector3 translation\nQuaternion rotation"
ENTITY = "MSG: task_generator_msgs/SemanticEntityState\nstring entity\nstring kind\nstring[] discrete_names\nstring[] discrete_values\nstring[] continuous_names\nfloat64[] continuous_values\nstring[] predicate_names\nbool[] predicate_values\nstring[] members"
PEDESTRIAN = "MSG: arena_people_msgs/Pedestrian\nstring name\nuint32 id\ngeometry_msgs/Pose pose\ngeometry_msgs/Twist twist"


def _msgdef(body: str, *deps: str) -> str:
    return SEP.join([body, *deps])


ODOMETRY = ("nav_msgs/msg/Odometry", _msgdef("std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist", HEADER, TIME, POSE_COV, POSE, POINT, QUAT, TWIST_COV, TWIST, VEC3))
LASER_SCAN = ("sensor_msgs/msg/LaserScan", _msgdef("std_msgs/Header header\nfloat32 angle_min\nfloat32 angle_max\nfloat32 angle_increment\nfloat32 range_min\nfloat32 range_max\nfloat32[] ranges", HEADER, TIME))
TWIST_MSG = ("geometry_msgs/msg/Twist", _msgdef("geometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular", VEC3))
TWIST_STAMPED = ("geometry_msgs/msg/TwistStamped", _msgdef("std_msgs/Header header\ngeometry_msgs/Twist twist", HEADER, TIME, TWIST, VEC3))
JOINT_STATE = ("sensor_msgs/msg/JointState", _msgdef("std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort", HEADER, TIME))
PEDESTRIANS = ("arena_people_msgs/msg/Pedestrians", _msgdef("std_msgs/Header header\nPedestrian[] pedestrians", HEADER, TIME, PEDESTRIAN, POSE, POINT, QUAT, TWIST, VEC3))
AGENT_STATES = (
    "arena_humansim_msgs/msg/AgentStates",
    _msgdef("std_msgs/Header header\nAgentState[] agents", HEADER, TIME, "MSG: arena_humansim_msgs/AgentState\nuint32 agent_id\nuint8 kind\ngeometry_msgs/Pose2D pose\ngeometry_msgs/Vector3 velocity", "MSG: geometry_msgs/Pose2D\nfloat64 x\nfloat64 y\nfloat64 theta", VEC3),
)
ACOUSTICS = ("arena_robots_msgs/msg/Acoustics", _msgdef("uint32 FLAG_NO_EFFORT=2\nstd_msgs/Header header\nfloat32 total_level_af_dba\nfloat32 total_level_zf_db\nfloat32 baseline_level_dba\nfloat32 drivetrain_level_dba\nfloat32 uncertainty_1sigma_dba\nuint32 validity_flags", HEADER, TIME))
STRING = ("std_msgs/msg/String", "string data")
EPISODE_RECORD = (
    "task_generator_msgs/msg/EpisodeRecord",
    _msgdef("uint32 episode_id\nuint8 RUNNING = 1\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid\nrcl_interfaces/Parameter[] robots_params\nstring phases", PARAM, PARAM_VALUE),
)
EPISODE_RECORD_V1 = ("task_generator_msgs/msg/EpisodeRecord", "uint32 episode_id\nuint8 outcome_state\nstring outcome_info\nstring goal_uuid")
SEMANTIC_SNAPSHOT = ("task_generator_msgs/msg/SemanticSnapshot", _msgdef("builtin_interfaces/Time stamp\nuint32 env_id\nstring world\nSemanticEntityState[] entities", TIME, ENTITY))
COLLISION_EVENTS = (
    "arena_robots_msgs/msg/CollisionEvents",
    _msgdef("std_msgs/Header header\narena_robots_msgs/CollisionEvent[] events", HEADER, TIME, 'MSG: arena_robots_msgs/CollisionEvent\nstring KIND_WALL="wall"\nstring kind\nstring obstacle_id\nfloat32 distance'),
)
COLLISION_EVENTS_V1 = ("arena_robots_msgs/msg/CollisionEvents", _msgdef("std_msgs/Header header\narena_robots_msgs/CollisionEvent[] events", HEADER, TIME, "MSG: arena_robots_msgs/CollisionEvent\nstring obstacle_id\nfloat32 distance"))
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
    pedestrian = {"name": "alice", "pose": {"position": {"x": 3.0, "y": 4.0, "z": 0.0}, "orientation": YAW_HALF_PI}, "twist": {"linear": {"x": 0.3, "y": -0.1}}}
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
    events = [{"kind": "wall", "obstacle_id": "w1"}, {"kind": "static", "obstacle_id": "s1"}, {"kind": "pedestrian", "obstacle_id": "p1"}, {"kind": "robot", "obstacle_id": "w1"}]
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
        ("/env_0/jackal/acoustics", ACOUSTICS, {"header": _header(), "total_level_af_dba": 55.5, "total_level_zf_db": 60.0, "baseline_level_dba": 40.0, "drivetrain_level_dba": 50.0, "uncertainty_1sigma_dba": 2.0, "validity_flags": 2}, 2 * SEC),
        ("/env_0/jackal/characterization_phase", STRING, {"data": "ramp_up"}, 2 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, _schedule(("a", 0.2), ("b", 0.4)), 2 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, {"data": "not json"}, 3 * SEC),
        ("/env_0/jackal/characterization_schedule", STRING, _schedule(("c", 0.6)), 4 * SEC),
        ("/env_0/state/episode", EPISODE_RECORD, {"episode_id": 7, "outcome_state": 2, "outcome_info": "done", "goal_uuid": "g", "robots_params": params, "phases": '{"jackal": {"map_poses": []}}'}, 5 * SEC),
        ("/env_0/state/episode", EPISODE_RECORD, {"episode_id": 7, "outcome_state": 3, "robots_params": [], "phases": ""}, 6 * SEC),
        ("/env_1/agent_states", AGENT_STATES, {"header": _header(frame_id="map"), "agents": [{"agent_id": 5, "kind": 0, "pose": {"x": 1.0, "y": 2.0, "theta": 0.5}, "velocity": {"x": 0.1}}]}, 2 * SEC),
        ("/env_1/state/episode", EPISODE_RECORD_V1, {"episode_id": 8, "outcome_state": 3, "outcome_info": "old", "goal_uuid": "h"}, 5 * SEC),
        (
            "/arena/state/semantics",
            SEMANTIC_SNAPSHOT,
            {"env_id": 0, "world": "w", "entities": [{"entity": "env_0/door_1", "kind": "door", "discrete_names": ["state"], "discrete_values": ["open"], "continuous_names": ["angle"], "continuous_values": [1.5], "predicate_names": ["locked"], "predicate_values": [True], "members": ["ped_1"]}]},
            1 * SEC,
        ),
        ("/env_0/jackal/collision_events", COLLISION_EVENTS, {"header": _header(), "events": events}, 3 * SEC),
        ("/env_0/jackal/collision_events", COLLISION_EVENTS, {"header": _header(), "events": []}, 4 * SEC),
        ("/env_0/burger/collision_events", COLLISION_EVENTS_V1, {"header": _header(), "events": [{"obstacle_id": "x"}]}, 3 * SEC),
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


def test_odometry_prefers_the_controller_topic_and_keeps_header_stamps(every_topic: tuple[pathlib.Path, dict]):
    out, bundles = every_topic

    controller = bundles["env_0_jackal"].odom.collect()
    raw = pl.read_parquet(out / "env_0_jackal" / "odom.parquet")
    assert controller["pos_x"].to_list() == [7.0]
    assert raw.row(0, named=True) == {"time_ns": SEC, "stamp_ns": SEC + 5, "pos_x": 1.0, "pos_y": 2.0, "yaw": pytest.approx(math.pi / 2), "vel_linear": 0.5, "vel_angular": 0.1}


def test_scan_and_lidar_share_one_table_with_the_nearest_valid_return(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    scan = bundles["env_0_jackal"].scan.collect()
    assert scan["scan_min"].to_list() == [1.5, 10.0]
    assert scan["scan_range_max"].to_list() == [10.0, 10.0]
    assert scan["scan_ranges"].to_list()[0] == pytest.approx([0.05, 2.0, 1.5, 30.0])


def test_cmd_vel_reads_twist_and_falls_back_to_the_controller_twist_stamped(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    jackal = bundles["env_0_jackal"].cmd_vel.collect().row(0, named=True)
    burger = bundles["env_0_burger"].cmd_vel.collect().row(0, named=True)
    assert (jackal["linear_x"], jackal["linear_y"], jackal["angular_z"]) == pytest.approx((0.4, 0.1, -0.2))
    assert (burger["linear_x"], burger["angular_z"]) == pytest.approx((0.7, 0.3))


def test_joint_states_keep_their_arrays(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    row = bundles["env_0_jackal"].joint_states.collect().row(0, named=True)
    assert row["name"] == ["left", "right"]
    assert row["velocity"] == pytest.approx([1.0, 1.1])
    assert row["effort"] == pytest.approx([0.0, 0.5])


def test_arena_peds_win_over_engine_agent_states(every_topic: tuple[pathlib.Path, dict]):
    out, bundles = every_topic

    peds = bundles["env_0_jackal"].peds.collect().to_dicts()
    assert (out / "env_0" / "peds_physics.parquet").exists()
    assert peds == [
        {
            "time_ns": 2 * SEC,
            "peds_frame_id": "map",
            "num_pedestrians": 1,
            "peds_names": ["alice"],
            "peds_positions": [3.0, 4.0, 0.0],
            "peds_headings": [pytest.approx(math.pi / 2)],
            "peds_twists": [pytest.approx(0.3), pytest.approx(-0.1), 0.0],
            "peds_ids": [0],
            "peds_animation_states": [0],
            "peds_interaction_ids": [-1],
            "peds_interaction_types": [0],
        }
    ]


def test_acoustics_and_characterization_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    acoustics = jackal.acoustics.collect().row(0, named=True)
    assert acoustics["total_level_af_dba"] == pytest.approx(55.5)
    assert acoustics["validity_flags"] == 2
    assert jackal.characterization_phase.collect()["label"].to_list() == ["ramp_up"]
    assert jackal.characterization_schedule.collect().to_dicts() == [{"phase_label": "c", "phase_kind": "ramp", "vx_target": 0.6, "vy_target": 0.0, "wz_target": 0.0, "duration_s": 2.0, "ramp_s": 0.5, "turn_radius_m": 0.0}]


def test_episode_record_flattens_robot_params_into_nested_yaml(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    records = bundles["env_0_jackal"].episode_record.collect()
    record = records.row(0, named=True)
    assert records["outcome_state"].to_list() == [2, 3]
    params = yaml.safe_load(record["robots_params"])
    assert (record["episode_id"], record["outcome_state"], record["outcome_info"], record["goal_uuid"]) == (7, 2, "done", "g")
    assert record["phases"] == '{"jackal": {"map_poses": []}}'
    assert params["jackal"] == {"enabled": True, "count": 3, "radius": 0.25, "name": "j"}
    assert params["raw"] == [1, 2]
    assert params["flags"] == [True, False]
    assert params["ints"] == [4, 5]
    assert params["floats"] == [0.5]
    assert params["names"] == ["a", "b"]
    assert isinstance(params["unset"], str)


def test_episode_record_of_an_older_schema_has_no_phases_or_params(every_topic: tuple[pathlib.Path, dict]):
    out, _ = every_topic

    record = pq.read_table(out / "env_1" / "episode_record.parquet").to_pylist()[0]
    assert record["phases"] == ""
    assert record["robots_params"] == ""
    assert record["outcome_info"] == "old"
    assert pq.read_table(out / "env_1" / "peds.parquet").to_pylist()[0]["peds_positions"] == [1.0, 2.0, 0.0]


def test_semantic_snapshot_flattens_each_entity_field_into_a_row(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    rows = bundles["env_0_jackal"].semantic_snapshot.collect().select("field", "field_kind", "value_str", "value_num", "value_bool", "value_list").to_dicts()
    assert rows == [
        {"field": "state", "field_kind": "discrete", "value_str": "open", "value_num": None, "value_bool": None, "value_list": None},
        {"field": "angle", "field_kind": "continuous", "value_str": None, "value_num": 1.5, "value_bool": None, "value_list": None},
        {"field": "locked", "field_kind": "predicate", "value_str": None, "value_num": None, "value_bool": True, "value_list": None},
        {"field": "members", "field_kind": "members", "value_str": None, "value_num": None, "value_bool": None, "value_list": ["ped_1"]},
    ]


def test_collision_events_count_kinds_only_when_the_schema_has_them(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic

    events = bundles["env_0_jackal"].collision_events.collect()
    jackal = events.row(0, named=True)
    assert events.row(1, named=True)["collision_wall"] == 0
    burger = bundles["env_0_burger"].collision_events.collect().row(0, named=True)
    assert (jackal["collision_event"], jackal["collision_wall"], jackal["collision_static"], jackal["collision_pedestrian"]) == (4, 1, 1, 1)
    assert jackal["collision_obstacle_ids"] == ["p1", "s1", "w1"]
    assert (burger["collision_event"], burger["collision_wall"], burger["collision_static"], burger["collision_pedestrian"]) == (1, None, None, None)


def test_monitor_power_energy_and_plan_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    assert jackal.collision_monitor_state.collect().select("action_type", "polygon_name").row(0) == (2, "stop_zone")
    power = jackal.power.collect().row(0, named=True)
    assert (power["total_power_w"], power["joint_names"], power["joint_total_power_w"]) == (12.0, ["left"], [7.0])
    assert jackal.energy.collect().select("total_energy_consumed_wh", "battery_soc_percent").row(0) == (0.5, 99.0)
    plan = jackal.plan.collect().row(0, named=True)
    assert plan["poses_x"] == [0.0, 1.0]
    assert plan["poses_yaw"] == [0.0, pytest.approx(math.pi / 2)]


def test_task_pose_and_initialpose_topics(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    task_pose = jackal.task_pose.collect().row(0, named=True)
    assert (task_pose["stamp_ns"], task_pose["pos_x"], task_pose["pos_y"]) == (2 * SEC, 1.5, 2.5)
    assert task_pose["yaw"] == pytest.approx(math.pi / 2)
    assert jackal.initialpose.collect().select("pos_x", "pos_y", "yaw").row(0) == (0.5, 0.25, 0.0)


def test_tf_keeps_every_transform_and_extracts_world_to_base_ground_truth(every_topic: tuple[pathlib.Path, dict]):
    _, bundles = every_topic
    jackal = bundles["env_0_jackal"]

    assert len(jackal.tf.collect()) == 3
    assert len(jackal.tf_static.collect()) == 3
    gt = jackal.tf_gt.collect().select("time_ns", "pos_x_gt", "pos_y_gt", "yaw_gt", "frame_id").rows()
    assert gt == [(0, 0.0, 0.0, 0.0, "map"), (4 * SEC, 4.0, 5.0, pytest.approx(math.pi / 2), "map"), (4 * SEC, 9.0, 9.0, 0.0, "env_0/jackal/odom")]


@pytest.mark.parametrize("topic", ["/odom", "/cmd_vel"])
def test_un_namespaced_topic_is_filed_under_the_unknown_robot(tmp_path: pathlib.Path, topic: str):
    odom = {"header": _header(), "pose": {"pose": {"position": {"x": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    message = (ODOMETRY, odom) if topic == "/odom" else (TWIST_MSG, {"linear": {"x": 2.0}})
    bag = _write_bag(tmp_path / "bare.mcap", [(topic, message[0], message[1], SEC)])

    bundles = MCAPReader(bag).read(tmp_path / "out")

    assert set(bundles) == {"env_0_unknown"}


def test_reader_rejects_a_directory_without_bags(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="No MCAP file"):
        MCAPReader(tmp_path).read(tmp_path / "out")


def test_reader_rejects_a_missing_bag(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="not found"):
        MCAPReader(tmp_path / "missing.mcap").read(tmp_path / "out")


def _foreign_bag(path: pathlib.Path) -> pathlib.Path:
    with path.open("wb") as fh:
        writer = McapWriter(fh)
        writer.start()
        json_schema = writer.register_schema(name="Status", encoding="jsonschema", data=b"{}")
        json_channel = writer.register_channel(topic="/env_0/jackal/odom", message_encoding="json", schema_id=json_schema)
        writer.add_message(channel_id=json_channel, log_time=SEC, publish_time=SEC, data=b"{}")
        cdr_schema = writer.register_schema(name="nav_msgs/msg/Odometry", encoding="ros2msg", data=ODOMETRY[1].encode())
        cdr_channel = writer.register_channel(topic="/env_0/burger/odom", message_encoding="cdr", schema_id=cdr_schema)
        writer.add_message(channel_id=cdr_channel, log_time=SEC, publish_time=SEC, data=b"\x00\x01")
        writer.finish()
    return path


def test_directory_of_bags_skips_undecodable_and_corrupt_messages(tmp_path: pathlib.Path):
    bags = tmp_path / "bags"
    bags.mkdir()
    _foreign_bag(bags / "a.mcap")
    odom = {"header": _header(3, 0), "pose": {"pose": {"position": {"x": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    _write_bag(bags / "b.mcap", [("/env_0/robot/odom", ODOMETRY, odom, 3 * SEC)])

    bundles = MCAPReader(bags).read(tmp_path / "out")

    assert set(bundles) == {"env_0_robot"}
    assert bundles["env_0_robot"].odom.collect()["pos_x"].to_list() == [2.0]


def test_global_and_env_writers_append_across_buffer_flushes(tmp_path: pathlib.Path):
    peds = {"header": _header(frame_id="map"), "pedestrians": [{"name": "p", "pose": {"position": {"x": 1.0}, "orientation": {"w": 1.0}}, "twist": {}}]}
    tf = {"transforms": [_transform("map", "env_0/map", 0.0, 0.0)]}
    odom = {"header": _header(), "pose": {"pose": {"position": {}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    records = [("/env_0/jackal/odom", ODOMETRY, odom, 0)]
    for i in range(5001):
        records.append(("/tf", TF_MESSAGE, tf, i + 1))
        records.append(("/env_0/peds", PEDESTRIANS, peds, i + 1))
    records.append(("/env_0/jackal/odom", ODOMETRY, odom, 6000))

    bundles = MCAPReader(_write_bag(tmp_path / "long.mcap", records)).read(tmp_path / "out")

    assert len(bundles["env_0_jackal"].tf.collect()) == 5001
    assert len(bundles["env_0_jackal"].peds.collect()) == 5001
    assert len(bundles["env_0_jackal"].odom.collect()) == 2


def test_anchor_at_picks_the_last_anchor_not_after_the_time():
    anchors = [(0, 1.0, 1.0), (10, 2.0, 2.0), (20, 3.0, 3.0)]

    assert anchor_at(anchors, -5) == (1.0, 1.0)
    assert anchor_at(anchors, 10) == (2.0, 2.0)
    assert anchor_at(anchors, 15) == (2.0, 2.0)
    assert anchor_at(anchors, 99) == (3.0, 3.0)
