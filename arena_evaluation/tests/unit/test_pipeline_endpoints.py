import numpy as np
import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _episode_endpoints, _episode_window
from arena_evaluation.processing.pose_anchor import resolve_pose_source
from arena_evaluation.processing.topic_aligner import TopicAligner
from arena_evaluation.storage.schemas import TopicBundle


def _aligned(with_gt: bool) -> pl.DataFrame:
    cols = {
        "time_ns": [0, 1_000_000_000],
        "pos_x": [0.05, 3.0],
        "pos_y": [0.02, 1.0],
        "yaw": [0.5, 0.6],
    }
    if with_gt:
        cols |= {"pos_x_gt": [28.45, 31.0], "pos_y_gt": [9.8, 10.5], "yaw_gt": [1.4, 1.5]}
    return pl.DataFrame(cols)


def test_start_prefers_ground_truth_over_odom():
    # Odom starts wherever the robot's odometry happened to be, ground truth is map frame.
    start, goal = _episode_endpoints(_aligned(with_gt=True), None)
    assert start == [28.45, 9.8, 1.4]
    assert goal == [31.0, 10.5, 1.5]


def test_start_falls_back_to_odom_without_ground_truth():
    start, goal = _episode_endpoints(_aligned(with_gt=False), None)
    assert start == [0.05, 0.02, 0.5]
    assert goal == [3.0, 1.0, 0.6]


def test_goal_is_the_last_planned_pose_when_a_plan_exists():
    plan = pl.DataFrame({"time_ns": [0], "poses_x": [[1.0, 18.9]], "poses_y": [[1.0, 17.95]], "poses_yaw": [[0.0, -1.2]]})
    start, goal = _episode_endpoints(_aligned(with_gt=True), plan.lazy())
    assert start == [28.45, 9.8, 1.4]
    assert goal == [18.9, 17.95, -1.2]


def test_plans_after_the_aligned_window_do_not_define_the_goal():
    # The next episode's first plan can land in the tail of this recording.
    plan = pl.DataFrame(
        {
            "time_ns": [500_000_000, 5_000_000_000],
            "poses_x": [[1.0, 18.9], [0.0, 11.2]],
            "poses_y": [[1.0, 17.95], [0.0, 2.05]],
            "poses_yaw": [[0.0, -1.2], [0.0, -2.4]],
        }
    )
    _, goal = _episode_endpoints(_aligned(with_gt=True), plan)
    assert goal == [18.9, 17.95, -1.2]


def test_episode_window_spans_running_to_terminal_record():
    # A latched terminal record from the previous episode can precede this episode's RUNNING row.
    record = pl.DataFrame({"time_ns": [50, 100, 900], "outcome_state": [2, 1, 4]})
    assert _episode_window(record.lazy()) == (100, 900)
    assert _episode_window(None) == (None, None)
    assert _episode_window(pl.DataFrame({"time_ns": []})) == (None, None)


def test_episode_window_stays_open_when_the_terminal_record_was_not_captured():
    # previous episode's latched terminal row, then this episode's RUNNING row, and the recorder closed first
    record = pl.DataFrame({"time_ns": [50, 100], "outcome_state": [3, 1]})
    assert _episode_window(record) == (100, None)
    assert _episode_window(pl.DataFrame({"time_ns": [100], "outcome_state": [1]})) == (100, None)


def test_start_is_the_anchored_pose_not_the_odom_origin():
    t = (np.arange(100) * 33_333_333).astype(np.int64)
    x = np.linspace(0.0, 10.0, 100)
    odom = pl.DataFrame({"time_ns": t, "stamp_ns": t, "pos_x": x, "pos_y": np.zeros(100), "yaw": np.zeros(100)})
    # One ground-truth sample near the end of the run, the map frame is 3 m / -2 m away and rotated.
    c, s = np.cos(0.7), np.sin(0.7)
    gt = pl.DataFrame({"time_ns": t[[90]], "stamp_ns_gt": t[[90]], "pos_x_gt": [3.0 + c * x[90]], "pos_y_gt": [-2.0 + s * x[90]], "yaw_gt": [0.7], "frame_id": ["env_0/jackal/odom"]})

    bundle = TopicBundle(odom=odom, tf_gt=gt)
    bundle.tf_gt, source = resolve_pose_source(bundle, (None, None))
    assert source.kind == "anchored"

    start, _ = _episode_endpoints(TopicAligner().align(bundle), None)
    assert start == pytest.approx([3.0, -2.0, 0.7])
