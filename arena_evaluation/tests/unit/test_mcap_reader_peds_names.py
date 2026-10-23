import pathlib
from unittest import mock

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_lazy import MockChannel, MockHeader, MockMessage, MockOdomMsg, MockPose, MockVector3, _identity_decoder_factory


class _Schema:
    id = 2
    data = b"std_msgs/Header header\nPedestrian[] pedestrians\n"

    def __init__(self, name):
        self.name = name


class _Ped:
    __slots__ = ("name", "id", "pose", "twist")

    def __init__(self, name, ped_id, x, y):
        self.name = name
        self.id = ped_id
        self.pose = MockPose(x, y)
        self.twist = mock.Mock(linear=MockVector3(0.1, 0.0))


class _PedsMsg:
    def __init__(self, peds):
        self.header = MockHeader("map")
        self.pedestrians = peds


def test_reader_types_ped_columns_when_the_first_sample_is_empty(tmp_path: pathlib.Path):
    mcap_file = tmp_path / "dummy.mcap"
    mcap_file.touch()
    peds_schema = _Schema("arena_people_msgs/msg/Pedestrians")
    messages = [
        (None, MockChannel("/env_0/jackal/odom"), MockMessage(0, channel_id=0, data=MockOdomMsg())),
        (peds_schema, MockChannel("/env_0/arena_peds"), MockMessage(1, channel_id=1, data=_PedsMsg([]))),
        (peds_schema, MockChannel("/env_0/arena_peds"), MockMessage(2, channel_id=1, data=_PedsMsg([_Ped("env_0/Pedestrian_1", 1, 1.0, 2.0), _Ped("env_0/Pedestrian_2", 2, 3.0, 4.0)]))),
    ]

    with mock.patch("arena_evaluation.processing.mcap_reader.NonSeekingReader") as reader_cls:
        reader_cls.return_value.iter_messages.return_value = messages
        reader_cls.return_value._decoder_factories = [_identity_decoder_factory()]
        MCAPReader(mcap_file).read(tmp_path / "raw")

    peds = pl.read_parquet(tmp_path / "raw" / "env_0" / "peds.parquet")
    assert peds.schema["peds_names"] == pl.List(pl.Utf8)
    assert {peds.schema[name] for name in ("peds_positions", "peds_headings", "peds_twists")} == {pl.List(pl.Float64)}
    assert peds["peds_names"].to_list() == [[], ["env_0/Pedestrian_1", "env_0/Pedestrian_2"]]
    assert peds["peds_positions"].to_list()[1][:2] == [1.0, 2.0]
