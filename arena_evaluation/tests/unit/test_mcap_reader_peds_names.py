import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import PEDESTRIANS, _header, _write_bag


def _ped(name: str, ped_id: int, x: float, y: float) -> dict:
    return {"name": name, "id": ped_id, "pose": {"position": {"x": x, "y": y, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {"linear": {"x": 0.1}}}


def _peds(*peds: dict) -> dict:
    return {"header": _header(frame_id="map"), "pedestrians": list(peds)}


def test_reader_types_ped_columns_when_the_first_sample_is_empty(tmp_path: pathlib.Path):
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/arena_peds", PEDESTRIANS, _peds(), 1),
            ("/env_0/arena_peds", PEDESTRIANS, _peds(_ped("env_0/Pedestrian_1", 1, 1.0, 2.0), _ped("env_0/Pedestrian_2", 2, 3.0, 4.0)), 2),
        ],
    )

    MCAPReader(bag).read(tmp_path / "raw")

    peds = pl.read_parquet(tmp_path / "raw" / "env_0" / "peds.parquet")
    assert peds.schema["peds_names"] == pl.List(pl.Utf8)
    assert {peds.schema[name] for name in ("peds_positions", "peds_headings", "peds_twists")} == {pl.List(pl.Float64)}
    assert peds["peds_names"].to_list() == [[], ["env_0/Pedestrian_1", "env_0/Pedestrian_2"]]
    assert peds["peds_positions"].to_list()[1][:2] == [1.0, 2.0]
