import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")


from test_mcap_reader_topics import _header


def _ped(name: str, ped_id: int, x: float, y: float) -> dict:
    return {
        "name": name,
        "id": ped_id,
        "pose": {"position": {"x": x, "y": y, "z": 0.0}, "orientation": {"w": 1.0}},
        "twist": {"linear": {"x": 0.1}},
        "gestures": [],
    }


def _peds(*peds: dict) -> dict:
    return {"header": _header(frame_id="map"), "pedestrians": list(peds)}
