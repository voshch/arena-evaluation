import math
import pathlib

import polars as pl
import pytest
from mcap_ros2.writer import Writer

from arena_evaluation.processing.mcap_reader import MCAPReader

_POSE_STAMPED = """std_msgs/Header header
geometry_msgs/Pose pose
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""


def test_goal_pose_is_read_per_robot_in_the_map_frame(tmp_path: pathlib.Path) -> None:
    bag = tmp_path / "run.mcap"
    yaw = 0.7
    with bag.open("wb") as f:
        writer = Writer(f)
        schema = writer.register_msgdef("geometry_msgs/msg/PoseStamped", _POSE_STAMPED)
        for i, x in enumerate([4.0, 6.5]):
            msg = {
                "header": {"stamp": {"sec": i, "nanosec": 0}, "frame_id": "map"},
                "pose": {
                    "position": {"x": x, "y": -1.5, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)},
                },
            }
            writer.write_message("/env_0/jackal/goal_pose", schema, msg, log_time=i + 1, publish_time=i + 1)
        writer.finish()
    out = tmp_path / "topics"
    out.mkdir()
    MCAPReader(bag).read(out)
    goal = pl.read_parquet(out / "env_0_jackal" / "goal.parquet")
    assert goal["pos_x"].to_list() == pytest.approx([4.0, 6.5])
    assert goal["pos_y"].to_list() == pytest.approx([-1.5, -1.5])
    assert goal["yaw"].to_list() == pytest.approx([yaw, yaw])
