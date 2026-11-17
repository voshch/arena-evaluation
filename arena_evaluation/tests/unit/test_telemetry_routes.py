"""Topic routing: accepted topics without an environment, unknown topics, and environment-level initial poses."""

from __future__ import annotations

import logging
import pathlib

import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

import polars as pl
from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import (
    ODOMETRY,
    POSE_COV_STAMPED,
    SEC,
    _header,
    _write_bag,
)

_READER_LOGGER = "arena_evaluation.processing.mcap_reader"


def _odometry(x: float) -> dict:
    return {
        "header": _header(frame_id="odom"),
        "pose": {"pose": {"position": {"x": x, "y": 0.0}, "orientation": {"w": 1.0}}},
        "twist": {"twist": {}},
    }


def _initial_pose(x: float) -> dict:
    return {
        "header": _header(),
        "pose": {"pose": {"position": {"x": x, "y": 0.0}, "orientation": {"w": 1.0}}},
    }


def _warnings_about(caplog: pytest.LogCaptureFixture, topic: str) -> list[str]:
    return [
        record.message
        for record in caplog.records
        if record.name == _READER_LOGGER and topic in record.message
    ]


def test_an_accepted_topic_without_an_environment_is_skipped_with_one_warning(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    bag = _write_bag(
        tmp_path / "bare.mcap",
        [
            ("/odom", ODOMETRY, _odometry(2.0), SEC),
            ("/odom", ODOMETRY, _odometry(3.0), 2 * SEC),
        ],
    )

    with caplog.at_level(logging.WARNING, logger=_READER_LOGGER):
        bundles = MCAPReader(bag).read(tmp_path / "out")

    assert bundles == {}
    assert len(_warnings_about(caplog, "/odom")) == 1


def test_an_environment_level_initial_pose_is_not_a_robot_row(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    bag = _write_bag(
        tmp_path / "initialpose.mcap",
        [
            ("/env_0/initialpose", POSE_COV_STAMPED, _initial_pose(1.0), SEC),
            (
                "/env_0/jackal/initialpose",
                POSE_COV_STAMPED,
                _initial_pose(2.0),
                2 * SEC,
            ),
        ],
    )

    with caplog.at_level(logging.WARNING, logger=_READER_LOGGER):
        bundles = MCAPReader(bag).read(tmp_path / "out")

    assert set(bundles) == {"env_0_jackal"}
    rows = pl.read_parquet(tmp_path / "out" / "robot_initialpose.parquet")
    assert rows["robot_id"].to_list() == ["env_0_jackal"]
    assert len(_warnings_about(caplog, "/env_0/initialpose")) == 1
