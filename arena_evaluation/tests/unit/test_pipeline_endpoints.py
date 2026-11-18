import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _episode_endpoints, _episode_window


def _aligned(with_gt: bool) -> pl.DataFrame:
    cols = {
        "t_ns": [0, 1_000_000_000],
        "pos_x": [0.05, 3.0],
        "pos_y": [0.02, 1.0],
        "yaw": [0.5, 0.6],
    }
    if with_gt:
        cols |= {"pos_x_gt": [28.45, 31.0], "pos_y_gt": [9.8, 10.5], "yaw_gt": [1.4, 1.5]}
    return pl.DataFrame(cols)


def _plan(poses: list[tuple[float, float, float]], plan_id: int = 0, t_ns: int = 0) -> pl.DataFrame:
    """One planned path as flat waypoint rows."""
    return pl.DataFrame(
        {
            "t_ns": [t_ns] * len(poses),
            "plan_id": [plan_id] * len(poses),
            "waypoint_idx": list(range(len(poses))),
            "x": [pose[0] for pose in poses],
            "y": [pose[1] for pose in poses],
            "yaw": [pose[2] for pose in poses],
        }
    )


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
    start, goal = _episode_endpoints(_aligned(with_gt=True), _plan([(1.0, 1.0, 0.0), (18.9, 17.95, -1.2)]).lazy())
    assert start == [28.45, 9.8, 1.4]
    assert goal == [18.9, 17.95, -1.2]


def test_plans_after_the_aligned_window_do_not_define_the_goal():
    # The next episode's first plan can land in the tail of this recording.
    plans = pl.concat([_plan([(1.0, 1.0, 0.0), (18.9, 17.95, -1.2)], plan_id=0, t_ns=500_000_000), _plan([(0.0, 0.0, 0.0), (11.2, 2.05, -2.4)], plan_id=1, t_ns=5_000_000_000)])

    _, goal = _episode_endpoints(_aligned(with_gt=True), plans)
    assert goal == [18.9, 17.95, -1.2]


def _phases(map_poses: list[list[float] | None]) -> dict:
    return {"phases": [{} for _ in map_poses], "conditions": [], "map_poses": map_poses}


def test_final_goto_of_the_recorded_phases_defines_the_goal_over_the_plan():
    _, end = _episode_endpoints(_aligned(with_gt=True), _plan([(1.0, 1.0, 0.0), (18.9, 17.95, -1.2)]), _phases([[4.0, 1.0, 0.1], [19.0, 18.0, -1.1]]))
    assert end == [19.0, 18.0, -1.1]


def test_final_goto_of_the_recorded_phases_defines_the_goal_without_a_plan():
    _, end = _episode_endpoints(_aligned(with_gt=True), None, _phases([[7.5, -2.0, 0.3], None]))
    assert end == [7.5, -2.0, 0.3]


def test_phases_without_a_goto_leave_the_goal_to_the_plan():
    _, end = _episode_endpoints(_aligned(with_gt=True), _plan([(1.0, 1.0, 0.0), (18.9, 17.95, -1.2)]), _phases([None]))
    assert end == [18.9, 17.95, -1.2]


def test_episode_window_spans_running_to_terminal_record():
    # A latched terminal record from the previous episode can precede this episode's RUNNING row.
    record = pl.DataFrame({"episode_id": [7, 7, 7], "t_ns": [50, 100, 900], "outcome_state": [2, 1, 4]})
    assert _episode_window(record.lazy()) == (100, 900)
    assert _episode_window(None) == (None, None)
    assert _episode_window(pl.DataFrame({"t_ns": []})) == (None, None)


def test_episode_window_stays_open_when_the_terminal_record_was_not_captured():
    record = pl.DataFrame({"episode_id": [7, 7], "t_ns": [50, 100], "outcome_state": [3, 1]})
    assert _episode_window(record) == (100, None)
    assert _episode_window(pl.DataFrame({"episode_id": [7], "t_ns": [100], "outcome_state": [1]})) == (100, None)


def test_episode_window_stays_open_without_a_terminal_or_running_record():
    assert _episode_window(pl.DataFrame({"episode_id": [7, 7], "t_ns": [50, 100], "outcome_state": [2, 1]})) == (100, None)
    assert _episode_window(pl.DataFrame({"episode_id": [7], "t_ns": [700], "outcome_state": [3]})) == (None, 700)
    assert _episode_window(pl.DataFrame({"t_ns": [5, 9]})) == (5, 9)


def test_start_is_the_first_ground_truth_pose_not_the_odometry_origin():
    aligned = pl.DataFrame({"t_ns": [0, 1_000_000_000], "pos_x": [0.0, 10.0], "pos_y": [0.0, 0.0], "yaw": [0.0, 0.0], "pos_x_gt": [3.0, 13.0], "pos_y_gt": [-2.0, -2.0], "yaw_gt": [0.7, 0.7]})

    start, _ = _episode_endpoints(aligned, None)
    assert start == pytest.approx([3.0, -2.0, 0.7])


def test_episode_window_without_outcome_states_spans_the_whole_record():
    assert _episode_window(pl.DataFrame({"t_ns": [300, 100, 900]})) == (100, 900)
