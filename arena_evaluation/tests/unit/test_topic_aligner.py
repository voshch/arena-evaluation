import polars as pl
from arena_evaluation.storage.schemas import TopicBundle
from arena_evaluation.processing.topic_aligner import TopicAligner


def test_align_topics_backward():
    # Synthetic odom data (10Hz)
    odom_df = pl.DataFrame({"time_ns": [1000, 2000, 3000, 4000], "pos_x": [1.0, 2.0, 3.0, 4.0]}).sort("time_ns")

    # Synthetic scan data (5Hz), arrive slightly after odom
    scan_df = pl.DataFrame({"time_ns": [1100, 3100], "scan_min": [0.5, 0.6]}).sort("time_ns")

    bundle = TopicBundle(odom=odom_df, scan=scan_df)

    # TopicAligner aligns backward (previous reading)
    aligner = TopicAligner(tolerance_ns=500)
    aligned_df = aligner.align(bundle)

    assert aligned_df is not None
    assert len(aligned_df) == 4


def test_align_topics_forward():
    odom_df = pl.DataFrame({"time_ns": [1000, 2000, 3000, 4000], "pos_x": [1.0, 2.0, 3.0, 4.0]}).sort("time_ns")

    scan_df = pl.DataFrame({"time_ns": [1100, 3100], "scan_min": [0.5, 0.6]}).sort("time_ns")

    bundle = TopicBundle(odom=odom_df, scan=scan_df)
    aligner = TopicAligner(tolerance_ns=200)

    aligned_df = aligner.align(bundle)

    # If strategy is 'backward', it joins the PREVIOUS row (scan <= odom)
    assert aligned_df is not None
    assert "scan_min" in aligned_df.columns
