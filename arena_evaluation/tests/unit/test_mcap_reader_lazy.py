
import pytest

pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import _header


def _odom(x: float, y: float) -> dict:
    return {"header": _header(frame_id="odom"), "pose": {"pose": {"position": {"x": x, "y": y}, "orientation": {"w": 1.0}}}, "twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}}


def _tf_row(t_ns: int, parent: str, child: str, tx: float, ty: float) -> dict:
    return {"t_ns": t_ns, "parent_frame": parent, "child_frame": child, "tx": tx, "ty": ty, "tz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}


