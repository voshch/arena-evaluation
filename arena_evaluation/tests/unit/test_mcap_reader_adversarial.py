import pytest

pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import _header


def _odom(x: float) -> dict:
    return {
        "header": _header(frame_id="odom"),
        "pose": {"pose": {"position": {"x": x, "y": 0.0}, "orientation": {"w": 1.0}}},
        "twist": {"twist": {}},
    }
