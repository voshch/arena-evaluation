import threading

import pytest
import rclpy
from mcap.reader import make_reader
from rosgraph_msgs.msg import Clock
from task_generator_msgs.msg import EpisodeRecord

from arena_evaluation.ingestion.recorder import DataRecorderNode


@pytest.fixture
def recorder_node(tmp_path):
    if not rclpy.ok():
        rclpy.init()
    node = DataRecorderNode()
    node.episodes_root = tmp_path / "episodes"
    node.episodes_root.mkdir(parents=True, exist_ok=True)
    yield node
    node.destroy_node()


def _log_times(node, topic: str) -> list[int]:
    node._close_current_writer()
    with open(node.current_episode_dir / f"{node.current_episode_dir.name}.mcap", "rb") as f:
        return [message.log_time for _, channel, message in make_reader(f).iter_messages(log_time_order=False) if channel.topic == topic]


def test_backward_time_jump(recorder_node):
    """Test that if /clock goes backward, the recorder resets its throttle timers and doesn't drop messages."""
    msg = EpisodeRecord()
    msg.episode_id = 1
    msg.outcome_state = EpisodeRecord.RUNNING
    recorder_node.episode_record_callback(msg)

    clock_msg = Clock()
    clock_msg.clock.sec = 10
    recorder_node.clock_callback(clock_msg)

    recorder_node._register_topic("/test_topic", Clock)
    cb = recorder_node._create_throttled_callback("/test_topic")

    cb(Clock())

    clock_msg.clock.sec = 5
    recorder_node.clock_callback(clock_msg)

    cb(Clock())
    assert _log_times(recorder_node, "test_topic") == [10 * 10**9, 5 * 10**9]


def test_malformed_metadata_write(recorder_node):
    """Test when the metadata.yaml is completely un-writable, the recorder doesn't crash."""
    ep_dir = recorder_node.episodes_root / "episode_000"
    ep_dir.mkdir(parents=True, exist_ok=True)

    (ep_dir / "episode_000.yaml").mkdir()

    recorder_node._start_episode_recording(2)
    assert recorder_node.writer is not None

    recorder_node.finalize()

    log = recorder_node.log_file_path.read_text()
    assert "Failed to write initial episode metadata" in log
    assert "Failed to finalize episode metadata" in log
    assert (ep_dir / "episode_000.mcap").exists()


def test_high_frequency_spam(recorder_node):
    """Test high frequency messages concurrently locking writer_lock."""
    msg = EpisodeRecord()
    msg.episode_id = 3
    msg.outcome_state = EpisodeRecord.RUNNING
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
    assert len(_log_times(recorder_node, "spam_topic")) == 1000
