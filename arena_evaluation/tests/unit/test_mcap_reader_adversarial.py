from types import SimpleNamespace

import pytest

pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry.messages import yaw_from_orientation
from test_mcap_reader_topics import ODOMETRY, _header, _write_bag


def _odom(x: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": x, "y": 0.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}


def test_duplicate_timestamps_keep_arrival_order_and_out_of_order_messages_are_sorted(tmp_path):
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/odom", ODOMETRY, _odom(5.0), 5000),
            ("/env_0/odom", ODOMETRY, _odom(1.0), 1000),
            ("/env_0/odom", ODOMETRY, _odom(1.1), 1000),
            ("/env_0/odom", ODOMETRY, _odom(3.0), 3000),
        ],
    )

    odom = MCAPReader(bag).read(tmp_path / "raw")["env_0"].robot_odom.collect()

    assert odom["t_ns"].to_list() == [1000, 1000, 3000, 5000]
    assert odom["odom_x"].to_list() == [1.0, 1.1, 3.0, 5.0]


def test_an_all_zero_quaternion_has_no_heading():
    assert yaw_from_orientation(SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0)) == 0.0
