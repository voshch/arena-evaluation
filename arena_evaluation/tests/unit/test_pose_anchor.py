"""The ground-truth pose stream of an episode is dense, sparse or absent; odometry is never a pose source."""

import polars as pl
import pytest

from arena_evaluation.processing.pose_anchor import PoseSource, classify, resolve_pose_source
from arena_evaluation.storage.schemas import TopicBundle

SEC = 1_000_000_000


def _track(stamps_s: list[float]) -> pl.DataFrame:
    stamps = [round(stamp * SEC) for stamp in stamps_s]
    return pl.DataFrame({"t_ns": stamps, "x": [float(i) for i in range(len(stamps))], "y": [0.0] * len(stamps), "yaw": [0.0] * len(stamps)})


def test_a_dense_track_is_evaluated_as_recorded():
    kept, source = resolve_pose_source(TopicBundle(robot_tracks=_track([0.0, 0.1, 0.2, 0.3])), (None, None))
    assert source.kind == "tf"
    assert source.samples == 4
    assert kept is not None and kept.height == 4


def test_a_sparse_track_is_kept_raw_and_never_filled_from_odometry():
    track = _track([0.0, 1.0, 2.0])
    bundle = TopicBundle(robot_tracks=track, robot_odom=_track([0.0, 0.1, 0.2, 0.3]))
    kept, source = resolve_pose_source(bundle, (None, None))
    assert source.kind == "tf_sparse"
    assert source.median_gap_s == pytest.approx(1.0)
    assert kept is not None and kept.get_column("t_ns").to_list() == track.get_column("t_ns").to_list()


def test_a_single_sample_is_sparse_without_a_gap():
    kept, source = resolve_pose_source(TopicBundle(robot_tracks=_track([5.0])), (None, None))
    assert (source.kind, source.samples, source.median_gap_s) == ("tf_sparse", 1, None)
    assert kept is not None


def test_no_track_is_absent():
    assert resolve_pose_source(TopicBundle(), (None, None)) == (None, PoseSource("absent", 0, None))


def test_the_window_bounds_the_samples_that_are_classified():
    track = _track([0.0, 0.1, 5.0, 5.1, 5.2])
    _, source = resolve_pose_source(TopicBundle(robot_tracks=track), (4 * SEC, None))
    assert (source.kind, source.samples) == ("tf", 3)


def test_classify_reads_the_gap_between_native_time_stamps():
    assert classify(_track([0.0, 0.1, 0.2])) == "dense"
    assert classify(_track([0.0, 0.5])) == "sparse"
    assert classify(_track([0.0])) == "sparse"
    assert classify(None) == "absent"
