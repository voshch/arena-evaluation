import polars as pl
import pytest
from arena_evaluation.storage.schemas import TopicBundle
from arena_evaluation.processing.topic_aligner import TopicAligner


def test_negative_time_deltas_and_unsorted():
    # Data is entirely backwards
    odom_df = pl.DataFrame({"time_ns": [5000, 4000, 3000, 2000, 1000], "pos_x": [5.0, 4.0, 3.0, 2.0, 1.0]})

    scan_df = pl.DataFrame({"time_ns": [5000, 1000], "scan_min": [0.5, 0.1]})

    bundle = TopicBundle(odom=odom_df, scan=scan_df)
    aligner = TopicAligner(tolerance_ns=500)
    aligned_df = aligner.align(bundle)

    assert aligned_df is not None
    assert len(aligned_df) == 5
    # Since it was sorted by aligner, time_ns should be 1000, 2000, ...
    assert aligned_df["time_ns"].to_list() == [1000, 2000, 3000, 4000, 5000]


def test_extreme_tolerance():
    odom_df = pl.DataFrame({"time_ns": [1000, 2000], "pos_x": [1.0, 2.0]})
    scan_df = pl.DataFrame({"time_ns": [1100, 2100], "scan_min": [0.5, 0.6]})
    bundle = TopicBundle(odom=odom_df, scan=scan_df)

    # 0 tolerance (should match nothing because times don't match exactly)
    aligner = TopicAligner(tolerance_ns=0)
    aligned_df = aligner.align(bundle)
    assert aligned_df["scan_min"].null_count() == 2


def test_start_greater_than_end_time():
    odom_df = pl.DataFrame({"time_ns": [1000, 2000], "pos_x": [1.0, 2.0]})
    bundle = TopicBundle(odom=odom_df)

    aligner = TopicAligner(tolerance_ns=500)
    # Start time is after end time, should return None or empty
    aligned_df = aligner.align(bundle, start_time_ns=3000, end_time_ns=1000)
    assert aligned_df is None


def test_missing_and_empty_bundles():
    bundle = TopicBundle(odom=None)
    aligner = TopicAligner()
    assert aligner.align(bundle) is None

    empty_odom = pl.DataFrame({"time_ns": [], "pos_x": []}, schema={"time_ns": pl.Int64, "pos_x": pl.Float64})
    bundle2 = TopicBundle(odom=empty_odom)
    assert aligner.align(bundle2) is None
