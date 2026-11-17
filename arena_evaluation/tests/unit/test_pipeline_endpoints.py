import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _episode_endpoints, _episode_window


def _aligned() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "x": [28.45, 31.0],
            "y": [9.8, 10.5],
            "yaw": [1.4, 1.5],
        }
    )


def _plan(
    poses: list[tuple[float, float, float]], plan_id: int = 0, t_ns: int = 0
) -> pl.DataFrame:
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


def _phases(map_poses: list[list[float] | None]) -> dict:
    return {"phases": [{} for _ in map_poses], "conditions": [], "map_poses": map_poses}


def test_final_goto_of_the_recorded_phases_defines_the_goal_over_the_plan():
    _, end = _episode_endpoints(
        _aligned(),
        _plan([(1.0, 1.0, 0.0), (18.9, 17.95, -1.2)]),
        _phases([[4.0, 1.0, 0.1], [19.0, 18.0, -1.1]]),
    )
    assert end == [19.0, 18.0, -1.1]


def test_episode_window_spans_running_to_terminal_record():
    # A latched terminal record from the previous episode can precede this episode's RUNNING row.
    record = pl.DataFrame(
        {"episode_id": [7, 7, 7], "t_ns": [50, 100, 900], "outcome_state": [2, 1, 4]}
    )
    assert _episode_window(record.lazy()) == (100, 900)
    assert _episode_window(None) == (None, None)
    assert _episode_window(pl.DataFrame({"t_ns": []})) == (None, None)


def test_episode_window_stays_open_without_a_terminal_or_running_record():
    assert _episode_window(
        pl.DataFrame({"episode_id": [7, 7], "t_ns": [50, 100], "outcome_state": [2, 1]})
    ) == (100, None)
    assert _episode_window(
        pl.DataFrame({"episode_id": [7], "t_ns": [700], "outcome_state": [3]})
    ) == (None, 700)


