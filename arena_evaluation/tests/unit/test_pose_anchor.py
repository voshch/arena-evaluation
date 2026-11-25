"""The ground-truth pose stream of an episode is dense, sparse or absent; odometry is never a pose source."""

import polars as pl

from arena_evaluation.processing.pose_anchor import resolve_pose_source
from arena_evaluation.storage.schemas import TopicBundle

SEC = 1_000_000_000


def _track(stamps_s: list[float]) -> pl.DataFrame:
    stamps = [round(stamp * SEC) for stamp in stamps_s]
    return pl.DataFrame(
        {
            "t_ns": stamps,
            "x": [float(i) for i in range(len(stamps))],
            "y": [0.0] * len(stamps),
            "yaw": [0.0] * len(stamps),
        }
    )


def test_a_single_sample_is_sparse_without_a_gap():
    kept, source = resolve_pose_source(TopicBundle(robot_tracks=_track([5.0])), (None, None))
    assert (source.kind, source.samples, source.median_gap_s) == ("tf_sparse", 1, None)
    assert kept is not None
