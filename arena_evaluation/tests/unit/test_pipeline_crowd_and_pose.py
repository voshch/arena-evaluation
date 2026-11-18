from __future__ import annotations

import pytest

pytest.importorskip("polars")

import polars as pl

from arena_evaluation.processing.pose_anchor import pose_frame


def test_pose_frame_is_the_most_frequent_parent_inside_the_window() -> None:
    tf_gt = pl.DataFrame(
        {
            "t_ns": [1, 2, 3, 4],
            "parent_frame": ["map", "env_0/jackal/odom", "env_0/jackal/odom", "map"],
        }
    )
    assert pose_frame(tf_gt, (2, 3)) == "env_0/jackal/odom"
    assert pose_frame(tf_gt, (None, None)) == "env_0/jackal/odom"
    assert pose_frame(tf_gt.lazy(), (4, None)) == "map"
    assert pose_frame(None, (None, None)) is None


def test_pose_frame_is_none_when_the_window_excludes_every_row() -> None:
    tf_gt = pl.DataFrame({"t_ns": [1, 2], "parent_frame": ["map", "map"]})
    assert pose_frame(tf_gt, (10, 20)) is None


def test_pose_frame_ignores_null_frame_ids() -> None:
    tf_gt = pl.DataFrame({"t_ns": [1, 2, 3], "parent_frame": [None, None, "map"]})
    assert pose_frame(tf_gt, (None, None)) == "map"
