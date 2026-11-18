from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")
pytest.importorskip("arena_people_msgs.msg")

import polars as pl
import pyarrow.parquet as pq
import rosbag2_py
from arena_humansim_msgs.msg import AgentFrame, AgentState, AgentStates
from arena_humansim_msgs.msg import AgentMeta as RosAgentMeta
from arena_people_msgs.msg import Gesture, Pedestrian, Pedestrians
from geometry_msgs.msg import Point, Pose, Pose2D, Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from tf2_msgs.msg import TFMessage

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry.writer import TABLE_SCHEMAS

SEC = 1_000_000_000
MS = 1_000_000


def _write_mcap(path: pathlib.Path, messages: list[tuple[str, object, int]]) -> pathlib.Path:
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    created: set[str] = set()
    for topic, msg, t_ns in messages:
        if topic not in created:
            type_name = f"{type(msg).__module__.split('.')[0]}/msg/{type(msg).__name__}"
            writer.create_topic(rosbag2_py.TopicMetadata(0, topic, type_name, "cdr"))
            created.add(topic)
        writer.write(topic, serialize_message(msg), t_ns)
    del writer
    return next(path.glob("*.mcap"))


def _robot_context_messages(t_ns: int = 0) -> list[tuple[str, object, int]]:
    odom = Odometry()
    odom.header.frame_id = "map"
    odom.child_frame_id = "env_0/jackal/base_link"
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0

    tf_msg = TFMessage()
    tf = TransformStamped()
    tf.header.frame_id = "map"
    tf.child_frame_id = "env_0/map"
    tf.transform.translation.x = 0.0
    tf.transform.translation.y = 0.0
    tf_msg.transforms.append(tf)

    return [
        ("/arena/env_0/jackal/odom", odom, t_ns),
        ("/tf_static", tf_msg, 0),
    ]


def test_agent_states_ingestion_preserves_simulation_metadata(tmp_path: pathlib.Path) -> None:
    bag_dir = tmp_path / "agent_states_bag"
    msg = AgentStates()
    msg.header.frame_id = "map"
    msg.header.stamp.sec = 1

    human_1 = AgentState(
        agent_id=1,
        kind=AgentState.KIND_HUMAN,
        name="nurse_clara",
        pose=Pose2D(x=2.5, y=3.5, theta=1.57),
        velocity=Vector3(x=0.3, y=0.4, z=0.0),
        radius=0.35,
        desired_velocity=1.2,
        policy="sfm",
        agent_type="nurse",
        animation_state=1,
    )

    human_2 = AgentState(
        agent_id=2,
        kind=AgentState.KIND_HUMAN,
        name="patron_dan",
        pose=Pose2D(x=6.0, y=-2.0, theta=-0.5),
        velocity=Vector3(x=-0.6, y=0.8, z=0.0),
        radius=0.45,
        desired_velocity=0.9,
        policy="orca",
        agent_type="wheelchair",
        animation_state=0,
    )

    robot_99 = AgentState(
        agent_id=99,
        kind=AgentState.KIND_ROBOT,
        name="robot_jackal",
        pose=Pose2D(x=0.0, y=0.0, theta=0.0),
        velocity=Vector3(x=1.0, y=0.0, z=0.0),
    )

    msg.agents = [human_1, robot_99, human_2]

    records: list[tuple[str, object, int]] = [
        *_robot_context_messages(1 * SEC),
        ("/arena/env_0/agent_states", msg, 1 * SEC),
    ]
    bag = _write_mcap(bag_dir, records)

    out_dir = tmp_path / "agent_states_out"
    bundles = MCAPReader(bag).read(out_dir)

    pedestrians_file = out_dir / "pedestrian_agent_states.parquet"
    arena_peds_file = out_dir / "pedestrian_arena_peds.parquet"

    assert pedestrians_file.exists()
    assert not arena_peds_file.exists()
    assert pq.read_schema(pedestrians_file).equals(TABLE_SCHEMAS["pedestrian_agent_states"], check_metadata=False)

    df = pl.read_parquet(pedestrians_file)
    assert df.height == 2

    nurse = df.filter(pl.col("ped_id") == "Pedestrian_1").row(0, named=True)
    assert nurse["env_id"] == "env_0"
    assert nurse["ped_name"] == "nurse_clara"
    assert nurse["x"] == 2.5
    assert nurse["y"] == 3.5
    assert nurse["yaw"] == pytest.approx(1.57)
    assert nurse["vx"] == pytest.approx(0.3)
    assert nurse["vy"] == pytest.approx(0.4)
    assert nurse["speed_mps"] == pytest.approx(0.5)
    assert nurse["radius_m"] == pytest.approx(0.35)
    assert nurse["desired_speed"] == pytest.approx(1.2)
    assert nurse["policy"] == "sfm"
    assert nurse["agent_type"] == "nurse"
    assert nurse["animation_state"] == 1

    patron = df.filter(pl.col("ped_id") == "Pedestrian_2").row(0, named=True)
    assert patron["env_id"] == "env_0"
    assert patron["ped_name"] == "patron_dan"
    assert patron["x"] == 6.0
    assert patron["y"] == -2.0
    assert patron["yaw"] == pytest.approx(-0.5)
    assert patron["vx"] == pytest.approx(-0.6)
    assert patron["vy"] == pytest.approx(0.8)
    assert patron["speed_mps"] == pytest.approx(1.0)
    assert patron["radius_m"] == pytest.approx(0.45)
    assert patron["desired_speed"] == pytest.approx(0.9)
    assert patron["policy"] == "orca"
    assert patron["agent_type"] == "wheelchair"
    assert patron["animation_state"] == 0

    assert "env_0_jackal" in bundles
    jackal_peds = bundles["env_0_jackal"].pedestrian_agent_states.collect()
    assert jackal_peds["ped_id"].to_list() == ["Pedestrian_1", "Pedestrian_2"]
    assert jackal_peds["radius_m"].to_list() == pytest.approx([0.35, 0.45])
    assert jackal_peds["desired_speed"].to_list() == pytest.approx([1.2, 0.9])
    assert jackal_peds["policy"].to_list() == ["sfm", "orca"]


def test_pedestrians_ingestion_into_arena_peds(tmp_path: pathlib.Path) -> None:
    bag_dir = tmp_path / "arena_peds_bag"
    ped = Pedestrian(
        name="anim_actor",
        id=5,
        pose=Pose(position=Point(x=1.0, y=2.0, z=0.0), orientation=Quaternion(w=1.0)),
        twist=Twist(linear=Vector3(x=0.5, y=0.0, z=0.0)),
        animation_state=2,
        interaction_id=3,
        interaction_type=7,
        gait_phase=0.8,
        gait_cadence=1.6,
        agent_type="adult",
        gestures=[Gesture(slot="right_arm", clip="wave", hand="r", at=Point(x=1.0, y=2.0, z=1.2), render_pose_override=False)],
    )
    peds_msg = Pedestrians(pedestrians=[ped])
    peds_msg.header.frame_id = "map"

    records: list[tuple[str, object, int]] = [
        *_robot_context_messages(1 * SEC),
        ("/arena/env_0/arena_peds", peds_msg, 1 * SEC),
    ]
    bag = _write_mcap(bag_dir, records)

    out_dir = tmp_path / "arena_peds_out"
    bundles = MCAPReader(bag).read(out_dir)

    arena_peds_file = out_dir / "pedestrian_arena_peds.parquet"
    pedestrians_file = out_dir / "pedestrian_agent_states.parquet"
    gestures_file = out_dir / "pedestrian_gestures.parquet"

    assert arena_peds_file.exists()
    assert not pedestrians_file.exists()
    assert gestures_file.exists()
    assert pq.read_schema(arena_peds_file).equals(TABLE_SCHEMAS["pedestrian_arena_peds"], check_metadata=False)

    df = pl.read_parquet(arena_peds_file)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["ped_id"] == "Pedestrian_5"
    assert row["ped_name"] == "anim_actor"
    assert row["x"] == 1.0
    assert row["y"] == 2.0
    assert row["animation_state"] == 2
    assert row["interaction_id"] == 3
    assert row["interaction_type"] == 7
    assert row["gait_phase"] == pytest.approx(0.8)
    assert row["gait_cadence"] == pytest.approx(1.6)
    assert row["agent_type"] == "adult"
    assert row["radius_m"] is None
    assert row["desired_speed"] is None
    assert row["policy"] is None

    gestures_df = pl.read_parquet(gestures_file)
    assert gestures_df.height == 1
    g_row = gestures_df.row(0, named=True)
    assert g_row["ped_id"] == "Pedestrian_5"
    assert g_row["slot"] == "right_arm"
    assert g_row["clip"] == "wave"

    jackal_peds = bundles["env_0_jackal"].pedestrian_arena_peds.collect()
    assert jackal_peds.height == 1
    assert jackal_peds["ped_id"].to_list() == ["Pedestrian_5"]


def test_concurrent_dual_topic_ingestion_eliminates_double_counting(tmp_path: pathlib.Path) -> None:
    bag_dir = tmp_path / "concurrent_bag"

    # Physics published at 50 Hz (20 ms interval = 20_000_000 ns)
    physics_stamps = [0, 20 * MS, 40 * MS, 60 * MS, 80 * MS, 100 * MS]
    # Visual published at 30 Hz (33.33 ms interval)
    visual_stamps = [0, 33 * MS, 66 * MS, 100 * MS]

    records: list[tuple[str, object, int]] = [*_robot_context_messages(0)]

    for t_ns in physics_stamps:
        state_msg = AgentStates()
        state_msg.header.frame_id = "map"
        state_msg.header.stamp.nanosec = t_ns
        agent = AgentState(
            agent_id=10,
            kind=AgentState.KIND_HUMAN,
            name="dr_smith",
            pose=Pose2D(x=float(t_ns) / SEC, y=1.0, theta=0.0),
            velocity=Vector3(x=1.0, y=0.0, z=0.0),
            radius=0.4,
            desired_velocity=1.1,
            policy="social_force",
            agent_type="doctor",
        )
        state_msg.agents = [agent]
        records.append(("/arena/env_0/agent_states", state_msg, t_ns))

    for t_ns in visual_stamps:
        peds_msg = Pedestrians()
        peds_msg.header.frame_id = "map"
        peds_msg.header.stamp.nanosec = t_ns
        ped = Pedestrian(
            name="smith_mesh",
            id=10,
            pose=Pose(position=Point(x=float(t_ns) / SEC, y=1.0, z=0.0), orientation=Quaternion(w=1.0)),
            twist=Twist(linear=Vector3(x=1.0, y=0.0, z=0.0)),
            animation_state=1,
            interaction_id=-1,
        )
        peds_msg.pedestrians = [ped]
        records.append(("/arena/env_0/arena_peds", peds_msg, t_ns))

    bag = _write_mcap(bag_dir, records)

    out_dir = tmp_path / "concurrent_out"
    bundles = MCAPReader(bag).read(out_dir)

    pedestrians_file = out_dir / "pedestrian_agent_states.parquet"
    arena_peds_file = out_dir / "pedestrian_arena_peds.parquet"

    assert pedestrians_file.exists()
    assert arena_peds_file.exists()

    ped_df = pl.read_parquet(pedestrians_file)
    arena_df = pl.read_parquet(arena_peds_file)

    # 1. Physics table has strictly 6 frames at 50 Hz, zero interleaving with 30 Hz visual frames
    assert ped_df.height == len(physics_stamps)
    assert ped_df["t_ns"].to_list() == physics_stamps
    assert ped_df["radius_m"].null_count() == 0
    assert ped_df["desired_speed"].null_count() == 0
    assert ped_df["policy"].null_count() == 0
    assert ped_df["agent_type"].null_count() == 0
    assert ped_df["policy"].to_list() == ["social_force"] * 6
    assert ped_df["agent_type"].to_list() == ["doctor"] * 6
    assert ped_df["radius_m"].to_list() == pytest.approx([0.4] * 6)
    assert ped_df["desired_speed"].to_list() == pytest.approx([1.1] * 6)

    # 2. Visual table has strictly 4 frames at 30 Hz, zero mixing with 50 Hz physics frames
    assert arena_df.height == len(visual_stamps)
    assert arena_df["t_ns"].to_list() == visual_stamps
    assert arena_df["radius_m"].null_count() == 4
    assert arena_df["desired_speed"].null_count() == 4
    assert arena_df["policy"].null_count() == 4

    # 3. `pedestrian_tracks` binds the map-frame arena stream, which shares the robot-track frame;
    #    the physics telemetry stays available as `pedestrian_tracks_physics`.
    bundle_tracks = bundles["env_0_jackal"].pedestrian_arena_peds.collect()
    assert bundle_tracks.height == len(visual_stamps)
    assert bundle_tracks["t_ns"].to_list() == visual_stamps
    bundle_physics = bundles["env_0_jackal"].pedestrian_agent_states.collect()
    assert bundle_physics.height == len(physics_stamps)
    assert bundle_physics["t_ns"].to_list() == physics_stamps
    assert bundle_physics["radius_m"].to_list() == pytest.approx([0.4] * 6)
    assert bundle_physics["desired_speed"].to_list() == pytest.approx([1.1] * 6)
    assert bundle_physics["policy"].to_list() == ["social_force"] * 6


def test_agent_frame_flat_layout_metadata_ingestion(tmp_path: pathlib.Path) -> None:
    bag_dir = tmp_path / "flat_agent_frame_bag"

    meta_msg = RosAgentMeta()
    meta_msg.policies = ["teleport", "sfm", "orca"]
    meta_msg.agent_id = [3, 7]
    meta_msg.agent_type = ["nurse", "visitor"]
    meta_msg.robot_awareness = [1.0, 0.0]
    meta_msg.robot_repulsion_gain = [1.0, 1.0]
    meta_msg.robot_fov_gain = [1.0, 0.3]
    meta_msg.robot_reaction_gain = [1.0, 0.5]

    frame_msg = AgentFrame()
    frame_msg.header.frame_id = "map"
    frame_msg.agent_id = [3, 99, 7]
    frame_msg.kind = [AgentState.KIND_HUMAN, AgentState.KIND_ROBOT, AgentState.KIND_HUMAN]
    frame_msg.x = [1.0, 0.0, 4.0]
    frame_msg.y = [2.0, 0.0, 5.0]
    frame_msg.theta = [0.1, 0.0, 0.2]
    frame_msg.vx = [0.5, 0.0, -0.3]
    frame_msg.vy = [0.0, 0.0, 0.4]
    frame_msg.desired_velocity = [1.25, 0.0, 0.85]
    frame_msg.radius = [0.38, 0.0, 0.42]
    frame_msg.policy_idx = [1, -1, 2]
    frame_msg.animation_state = [1, 0, 0]
    frame_msg.interaction_id = [-1, -1, 8]
    frame_msg.interaction_type = [0, 0, 3]
    frame_msg.gait_phase = [0.2, 0.0, 0.7]
    frame_msg.gait_cadence = [1.4, 0.0, 1.8]

    records: list[tuple[str, object, int]] = [
        *_robot_context_messages(1 * SEC),
        ("/arena/env_0/agent_meta", meta_msg, 0),
        ("/arena/env_0/agent_states", frame_msg, 1 * SEC),
    ]
    bag = _write_mcap(bag_dir, records)

    out_dir = tmp_path / "flat_agent_frame_out"
    MCAPReader(bag).read(out_dir)

    pedestrians_file = out_dir / "pedestrian_agent_states.parquet"
    assert pedestrians_file.exists()

    df = pl.read_parquet(pedestrians_file)
    assert df.height == 2

    p3 = df.filter(pl.col("ped_id") == "Pedestrian_3").row(0, named=True)
    assert p3["radius_m"] == pytest.approx(0.38)
    assert p3["desired_speed"] == pytest.approx(1.25)
    assert p3["policy"] == "sfm"
    assert p3["agent_type"] == "nurse"
    assert p3["animation_state"] == 1
    assert p3["gait_phase"] == pytest.approx(0.2)
    assert p3["gait_cadence"] == pytest.approx(1.4)
    assert p3["robot_awareness"] == pytest.approx(1.0)
    assert p3["robot_fov_gain"] == pytest.approx(1.0)

    p7 = df.filter(pl.col("ped_id") == "Pedestrian_7").row(0, named=True)
    assert p7["radius_m"] == pytest.approx(0.42)
    assert p7["robot_awareness"] == pytest.approx(0.0)
    assert p7["robot_reaction_gain"] == pytest.approx(0.5)
    assert p7["desired_speed"] == pytest.approx(0.85)
    assert p7["policy"] == "orca"
    assert p7["agent_type"] == "visitor"
    assert p7["interaction_id"] == 8
    assert p7["interaction_type"] == 3
    assert p7["gait_phase"] == pytest.approx(0.7)
    assert p7["gait_cadence"] == pytest.approx(1.8)
