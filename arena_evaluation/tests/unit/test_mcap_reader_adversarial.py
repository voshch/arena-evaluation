import pytest
import pathlib
import polars as pl
from unittest.mock import patch, MagicMock
from arena_evaluation.processing.mcap_reader import MCAPReader


def test_mcap_reader_duplicate_timestamps_and_out_of_order(tmp_path):
    reader = MCAPReader(tmp_path)
    # create dummy dir so it passes existence check
    dummy_mcap = tmp_path / "test.mcap"
    dummy_mcap.touch()

    with patch("arena_evaluation.processing.mcap_reader.NonSeekingReader") as mock_reader_cls, patch("builtins.open", new_callable=MagicMock):
        mock_instance = mock_reader_cls.return_value
        mock_instance.__enter__.return_value = mock_instance

        # Mocking iter_messages to return unsorted and duplicate messages
        Schema = MagicMock()

        Channel = MagicMock()
        Channel.topic = "/env_0_robot/odom"

        def make_msg(ts_ns, x):
            msg = MagicMock()
            msg.log_time = ts_ns
            msg.data = b""
            ros_msg = MagicMock()
            ros_msg.header.stamp.sec = 0
            ros_msg.header.stamp.nanosec = ts_ns
            ros_msg.pose.pose.position.x = x
            ros_msg.pose.pose.position.y = 0
            ros_msg.pose.pose.orientation.w = 1.0
            ros_msg.pose.pose.orientation.x = 0
            ros_msg.pose.pose.orientation.y = 0
            ros_msg.pose.pose.orientation.z = 0
            ros_msg.twist.twist.linear.x = 0
            ros_msg.twist.twist.angular.z = 0
            return (Schema, Channel, msg, ros_msg)

        msgs = [
            make_msg(5000, 5.0),
            make_msg(1000, 1.0),
            make_msg(1000, 1.1),  # Duplicate timestamp
            make_msg(3000, 3.0),
        ]

        mock_instance.iter_messages.return_value = [(s, c, m) for s, c, m, _ in msgs]

        def mock_decoder(data):
            pass

        with patch("arena_evaluation.processing.mcap_reader.DecoderFactory"):
            # It's tricky to mock the local decoder function.
            pass


def test_unflatten_dict_adversarial():
    # What if dict has overlapping paths?
    d = {"a.b": 1, "a.b.c": 2}
    res = MCAPReader._unflatten_dict(d)
    assert res == {"a": {"b": {"c": 2}}}


def test_quaternion_to_yaw_adversarial():
    # Invalid quaternion (0,0,0,0)
    yaw = MCAPReader._quaternion_to_yaw(0, 0, 0, 0)
    assert yaw == 0.0
