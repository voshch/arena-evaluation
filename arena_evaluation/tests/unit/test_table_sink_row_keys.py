"""The table sink keeps one row per declared key even when a key's copies straddle a flush."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.telemetry.writer import TableSink


def _tf(t_ns: int, child: str, tx: float) -> dict:
    return {
        "t_ns": t_ns,
        "parent_frame": "map",
        "child_frame": child,
        "tx": tx,
        "ty": 0.0,
        "tz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": 0.0,
        "qw": 1.0,
    }


def test_later_record_wins_when_its_key_straddles_a_flush(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "topics"
    sink = TableSink(out, flush_rows=2)
    sink.append("tf", _tf(10, "body_env_0_agent_6", 1.0))
    sink.append("tf", _tf(10, "body_env_0_agent_1", 5.0))
    sink.append("tf", _tf(10, "body_env_0_agent_6", 3.0))
    sink.close()

    stored = pl.read_parquet(out / "tf.parquet")

    assert stored.filter(pl.col("child_frame") == "body_env_0_agent_6")["tx"].to_list() == [3.0]
    assert stored.height == 2
