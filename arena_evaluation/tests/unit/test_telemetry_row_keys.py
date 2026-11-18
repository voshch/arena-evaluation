"""Row keys of the telemetry tables: duplicate rows, distinct elements at one stamp, and null keys."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.storage.contracts import check_benchmark
from arena_evaluation.storage.registry.telemetry import (
    RobotJointStates,
    RobotTracks,
    Tf,
    WorldSemantics,
)


def test_discriminator_tables_pin_their_row_key() -> None:
    assert RobotTracks.row_key == ("env_id", "robot_id", "t_ns")
    assert RobotJointStates.row_key == (
        "env_id",
        "robot_id",
        "t_ns",
        "joint_name",
    )
    assert Tf.row_key == ("t_ns", "parent_frame", "child_frame")
    assert WorldSemantics.row_key == (
        "t_ns",
        "env_id",
        "entity_name",
        "kind",
        "field_name",
        "field_kind",
    )


def _benchmark_with_telemetry(
    tmp_path: pathlib.Path, frame: pl.DataFrame
) -> pathlib.Path:
    """A benchmark whose only stored table is one episode's robot telemetry."""
    topics = tmp_path / "episodes" / "episode_000" / "topics"
    topics.mkdir(parents=True)
    frame.write_parquet(topics / "robot_tracks.parquet")
    return tmp_path


def test_the_contract_reports_a_duplicated_telemetry_row_key(
    tmp_path: pathlib.Path,
) -> None:
    frame = pl.DataFrame(
        {
            "env_id": ["env_0", "env_0"],
            "robot_id": ["env_0_jackal", "env_0_jackal"],
            "t_ns": [10, 10],
            "x": [1.0, 2.0],
            "y": [0.0, 0.0],
        }
    )

    report = check_benchmark(_benchmark_with_telemetry(tmp_path, frame))

    duplicates = [v for v in report.violations if v.check == "row_key_duplicate"]
    assert duplicates, report.summary()
    assert duplicates[0].table == "robot_tracks"
    assert "episode_000" in duplicates[0].detail


def test_the_contract_reports_a_null_telemetry_row_key(tmp_path: pathlib.Path) -> None:
    frame = pl.DataFrame(
        {
            "env_id": ["env_0", "env_0"],
            "robot_id": ["env_0_jackal", None],
            "t_ns": [10, 11],
            "x": [1.0, 2.0],
            "y": [0.0, 0.0],
        }
    )

    report = check_benchmark(_benchmark_with_telemetry(tmp_path, frame))

    assert [v for v in report.violations if v.check == "row_key_null"]
