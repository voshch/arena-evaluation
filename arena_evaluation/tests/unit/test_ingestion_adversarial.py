import os
import time
import pytest
import shutil
import threading
from unittest.mock import patch, MagicMock

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from task_generator_msgs.msg import EpisodeRecord
from arena_evaluation.ingestion.recorder import DataRecorderNode
from arena_evaluation.storage.manifest import MetadataWriter


@pytest.fixture
def recorder_node(tmp_path):
    if not rclpy.ok():
        rclpy.init()
    node = DataRecorderNode()
    node.episodes_root = tmp_path / "episodes"
    node.episodes_root.mkdir(parents=True, exist_ok=True)
    yield node
    node.destroy_node()


def test_backward_time_jump(recorder_node):
    """Test that if /clock goes backward, the recorder resets its throttle timers and doesn't drop messages."""
    msg = EpisodeRecord()
    msg.episode_id = 1
    recorder_node.episode_record_callback(msg)

    clock_msg = Clock()
    clock_msg.clock.sec = 10
    recorder_node.clock_callback(clock_msg)

    recorder_node._register_topic("/test_topic", Clock)
    cb = recorder_node._create_throttled_callback("/test_topic")

    recorder_node._write_to_bag_at = MagicMock()

    cb(Clock())
    recorder_node._write_to_bag_at.assert_called_once()
    recorder_node._write_to_bag_at.reset_mock()

    clock_msg.clock.sec = 5
    recorder_node.clock_callback(clock_msg)

    cb(Clock())
    # It should write it because we cleared last_recorded_times
    recorder_node._write_to_bag_at.assert_called_once()


def test_malformed_metadata_write(recorder_node, tmp_path):
    """Test when the metadata.yaml is completely un-writable, the recorder doesn't crash."""
    msg = EpisodeRecord()
    msg.episode_id = 2

    ep_dir = recorder_node.episodes_root / "episode_002"
    ep_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = ep_dir / "episode_002.yaml"

    # Create a directory where the file should be, causing a write error
    metadata_path.mkdir()

    # This should internally fail to write metadata but not crash
    recorder_node._start_episode_recording(2)

    # Finalize should also not crash
    recorder_node.finalize()


def test_high_frequency_spam(recorder_node):
    """Test high frequency messages concurrently locking writer_lock."""
    msg = EpisodeRecord()
    msg.episode_id = 3
    recorder_node.episode_record_callback(msg)

    clock_msg = Clock()
    clock_msg.clock.sec = 1
    recorder_node.clock_callback(clock_msg)

    recorder_node._register_topic("/spam_topic", Clock)
    cb = recorder_node._create_unthrottled_callback("/spam_topic")

    def spammer():
        for i in range(100):
            cb(Clock())

    threads = [threading.Thread(target=spammer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert recorder_node._write_success_count >= 1000
