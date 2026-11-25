import pytest

pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")


def _record(outcome_state: int, conditions: str) -> dict:
    return {
        "episode_id": 1,
        "world": "test_world",
        "seed": 42,
        "task_layer": "standard",
        "modes": [],
        "robots": ["jackal"],
        "outcome_state": outcome_state,
        "outcome_info": "",
        "goal_uuid": "",
        "goal_dist_start": 5.0,
        "goal_dist_min": 0.1,
        "path_length": 6.2,
        "start_time": {"sec": 1, "nanosec": 0},
        "integrity": [],
        "params": [],
        "human_params": [],
        "conditions": conditions,
        "phases": "",
    }
