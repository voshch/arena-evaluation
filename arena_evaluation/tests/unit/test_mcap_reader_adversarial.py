import pytest

pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import ODOMETRY, _header, _write_bag


def _odom(x: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": x, "y": 0.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}


def test_mcap_reader_duplicate_timestamps_and_out_of_order(tmp_path):
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/odom", ODOMETRY, _odom(5.0), 5000),
            ("/env_0/odom", ODOMETRY, _odom(1.0), 1000),
            ("/env_0/odom", ODOMETRY, _odom(1.1), 1000),
            ("/env_0/odom", ODOMETRY, _odom(3.0), 3000),
        ],
    )

    odom = MCAPReader(bag).read(tmp_path / "raw")["env_0"].odom.collect()

    assert odom["time_ns"].to_list() == [1000, 1000, 3000, 5000]
    assert odom["pos_x"].to_list() == [1.0, 1.1, 3.0, 5.0]


def test_unflatten_dict_adversarial():
    # What if dict has overlapping paths?
    d = {"a.b": 1, "a.b.c": 2}
    res = MCAPReader._unflatten_dict(d)
    assert res == {"a": {"b": {"c": 2}}}


def test_quaternion_to_yaw_adversarial():
    # Invalid quaternion (0,0,0,0)
    yaw = MCAPReader._quaternion_to_yaw(0, 0, 0, 0)
    assert yaw == 0.0
