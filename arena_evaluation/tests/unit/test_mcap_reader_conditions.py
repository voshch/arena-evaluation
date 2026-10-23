import json
import pathlib
from unittest import mock

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_lazy import MockChannel, MockMessage, MockOdomMsg, _identity_decoder_factory


class _EpisodeMsg:
    __slots__ = ("episode_id", "outcome_state", "outcome_info", "goal_uuid", "robots_params", "conditions")

    def __init__(self, outcome_state, conditions):
        self.episode_id = 1
        self.outcome_state = outcome_state
        self.outcome_info = ""
        self.goal_uuid = ""
        self.robots_params = []
        self.conditions = conditions


class _Schema:
    id = 1
    data = b"uint32 episode_id\nstring conditions\n"


def test_reader_keeps_episode_conditions(tmp_path: pathlib.Path):
    """Smoke run regression: conditions were dropped, so condition_success was always None."""
    mcap_file = tmp_path / "dummy.mcap"
    mcap_file.touch()
    clauses = json.dumps([{"op": "never", "p": "robot in restricted", "q": None, "text": "never robot in restricted"}])
    messages = [
        (None, MockChannel("/env_0/jackal/odom"), MockMessage(0, channel_id=0, data=MockOdomMsg())),
        (_Schema(), MockChannel("/env_0/state/episode"), MockMessage(1, channel_id=1, data=_EpisodeMsg(0, ""))),
        (_Schema(), MockChannel("/env_0/state/episode"), MockMessage(2, channel_id=1, data=_EpisodeMsg(2, clauses))),
    ]

    with mock.patch("arena_evaluation.processing.mcap_reader.NonSeekingReader") as reader_cls:
        reader_cls.return_value.iter_messages.return_value = messages
        reader_cls.return_value._decoder_factories = [_identity_decoder_factory()]
        MCAPReader(mcap_file).read(tmp_path / "raw")

    er = pl.read_parquet(tmp_path / "raw" / "env_0" / "episode_record.parquet")
    assert er["conditions"].to_list() == ["", clauses]
