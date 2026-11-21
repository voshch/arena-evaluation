"""Environment alignment of the recorded transforms: map-frame transforms move with the anchor, rotated or non-map-aligned frames are refused."""

from __future__ import annotations

import math
import pathlib

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing import mcap_reader

ENV = "env_3"
PEDESTRIAN_FRAMES = {(ENV, "pedestrian_arena_peds"): frozenset({"map", ""})}


def _tf_row(
    t_ns: int, parent: str, child: str, tx: float, ty: float, yaw: float = 0.0
) -> dict:
    return {
        "t_ns": t_ns,
        "parent_frame": parent,
        "child_frame": child,
        "tx": tx,
        "ty": ty,
        "tz": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": math.sin(yaw / 2),
        "qw": math.cos(yaw / 2),
    }


def _write(topics: pathlib.Path, name: str, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(topics / f"{name}.parquet")


def _row(topics: pathlib.Path, name: str, child: str) -> dict:
    return (
        pl.read_parquet(topics / f"{name}.parquet")
        .filter(pl.col("child_frame") == child)
        .row(0, named=True)
    )


def test_recorded_transforms_from_the_map_frame_move_with_the_anchor(
    tmp_path: pathlib.Path,
) -> None:
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(
        topics,
        "tf_static",
        [
            _tf_row(0, "map", f"{ENV}/map", 10.0, 20.0),
            _tf_row(0, "map", f"{ENV}/jackal/odom", 0.0, 0.0),
            _tf_row(0, f"{ENV}/jackal/odom", f"{ENV}/jackal/base_link", 6.0, 7.0),
        ],
    )
    _write(
        topics,
        "tf",
        [
            _tf_row(50, "map", f"body_{ENV}_agent_1", 25.0, 22.0),
            _tf_row(50, f"{ENV}/jackal/odom", f"{ENV}/jackal/base_link", 6.0, 7.0),
        ],
    )

    mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)

    anchor = _row(topics, "tf_static", f"{ENV}/map")
    odom = _row(topics, "tf_static", f"{ENV}/jackal/odom")
    body = _row(topics, "tf", f"body_{ENV}_agent_1")
    relative = _row(topics, "tf_static", f"{ENV}/jackal/base_link")
    assert (anchor["tx"], anchor["ty"]) == pytest.approx((0.0, 0.0))
    assert (odom["tx"], odom["ty"]) == pytest.approx((-10.0, -20.0))
    assert (body["tx"], body["ty"]) == pytest.approx((15.0, 2.0))
    assert (relative["tx"], relative["ty"]) == pytest.approx((6.0, 7.0))


def test_a_rotated_environment_frame_is_refused(tmp_path: pathlib.Path) -> None:
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(
        topics,
        "tf_static",
        [_tf_row(0, "map", f"{ENV}/map", 10.0, 20.0, yaw=math.pi / 2)],
    )

    with pytest.raises(ValueError, match="rotated"):
        mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)


def test_an_odometry_frame_that_is_not_map_aligned_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    topics = tmp_path / "topics"
    topics.mkdir()
    _write(
        topics,
        "tf_static",
        [
            _tf_row(0, "map", f"{ENV}/map", 10.0, 20.0),
            _tf_row(0, "map", f"{ENV}/jackal/odom", 1.0, 0.0),
        ],
    )

    with pytest.raises(ValueError, match="odom"):
        mcap_reader.align_to_environment(topics, PEDESTRIAN_FRAMES)
