import pathlib

import pyarrow.parquet as pq
import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import polars as pl

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry.writer import TABLE_SCHEMAS
from test_mcap_reader_topics import PEDESTRIANS, _header, _write_bag


def _ped(name: str, ped_id: int, x: float, y: float) -> dict:
    return {"name": name, "id": ped_id, "pose": {"position": {"x": x, "y": y, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {"linear": {"x": 0.1}}, "gestures": []}


def _peds(*peds: dict) -> dict:
    return {"header": _header(frame_id="map"), "pedestrians": list(peds)}


def test_pedestrian_ids_and_names_survive_an_empty_first_frame(tmp_path: pathlib.Path):
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/arena_peds", PEDESTRIANS, _peds(), 1),
            ("/env_0/arena_peds", PEDESTRIANS, _peds(_ped("env_0/Pedestrian_1", 1, 1.0, 2.0), _ped("env_0/Pedestrian_2", 2, 3.0, 4.0)), 2),
        ],
    )

    MCAPReader(bag).read(tmp_path / "raw")

    path = tmp_path / "raw" / "pedestrian_arena_peds.parquet"
    assert pq.read_schema(path).equals(TABLE_SCHEMAS["pedestrian_arena_peds"], check_metadata=False)
    assert pl.read_parquet(path).select("t_ns", "ped_id", "ped_name", "x", "y").rows() == [
        (2, "Pedestrian_1", "env_0/Pedestrian_1", 1.0, 2.0),
        (2, "Pedestrian_2", "env_0/Pedestrian_2", 3.0, 4.0),
    ]
