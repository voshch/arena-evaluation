from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")
pytest.importorskip("arena_people_msgs.msg")

import polars as pl
import pyarrow.parquet as pq
import rosbag2_py
from arena_humansim_msgs.msg import AgentFrame, AgentState
from arena_humansim_msgs.msg import AgentMeta as RosAgentMeta
from arena_people_msgs.msg import Gesture, Pedestrian, Pedestrians
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
    TransformStamped,
    Twist,
    Vector3,
)
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
        gestures=[
            Gesture(
                slot="right_arm",
                clip="wave",
                hand="r",
                at=Point(x=1.0, y=2.0, z=1.2),
                render_pose_override=False,
            )
        ],
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


def test_agent_frame_flat_layout_metadata_ingestion(tmp_path: pathlib.Path) -> None:
    bag_dir = tmp_path / "flat_agent_frame_bag"

    meta_msg = RosAgentMeta()
    meta_msg.policies = ["teleport", "sfm", "orca"]
    meta_msg.agent_id = [3, 7]
    meta_msg.agent_type = ["nurse", "visitor"]
    meta_msg.robot_range_offset = [0.0, -1.5]
    meta_msg.robot_fov_offset = [0.0, -30.0]
    meta_msg.robot_repulsion_gain = [1.0, 1.0]
    meta_msg.robot_reaction_gain = [1.0, 0.5]
    meta_msg.robot_blind = [False, True]

    frame_msg = AgentFrame()
    frame_msg.header.frame_id = "map"
    frame_msg.agent_id = [3, 99, 7]
    frame_msg.kind = [
        AgentState.KIND_HUMAN,
        AgentState.KIND_ROBOT,
        AgentState.KIND_HUMAN,
    ]
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
    assert p3["robot_range_offset"] == pytest.approx(0.0)
    assert p3["robot_fov_offset"] == pytest.approx(0.0)
    assert p3["robot_repulsion_gain"] == pytest.approx(1.0)
    assert p3["robot_reaction_gain"] == pytest.approx(1.0)
    assert p3["robot_blind"] is False

    p7 = df.filter(pl.col("ped_id") == "Pedestrian_7").row(0, named=True)
    assert p7["radius_m"] == pytest.approx(0.42)
    assert p7["robot_range_offset"] == pytest.approx(-1.5)
    assert p7["robot_fov_offset"] == pytest.approx(-30.0)
    assert p7["robot_repulsion_gain"] == pytest.approx(1.0)
    assert p7["robot_reaction_gain"] == pytest.approx(0.5)
    assert p7["robot_blind"] is True
    assert p7["desired_speed"] == pytest.approx(0.85)
    assert p7["policy"] == "orca"
    assert p7["agent_type"] == "visitor"
    assert p7["interaction_id"] == 8
    assert p7["interaction_type"] == 3
    assert p7["gait_phase"] == pytest.approx(0.7)
    assert p7["gait_cadence"] == pytest.approx(1.8)
