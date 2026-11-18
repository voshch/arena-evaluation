from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.pipeline import _episode_rows, _episode_window, _record_conditions, _record_integrity, _robot_phases

_OLD_CONDITIONS = '[{"op": "always", "p": "old episode"}]'
_OLD_PHASES = '{"jackal": {"phases": [], "conditions": [], "map_poses": [[1.0, 2.0, 0.0]], "goal_inputs": [], "instructions": []}}'


def _bag_records() -> pl.DataFrame:
    """The previous episode's latched result, this episode's run and result, then the next episode's queued record."""
    return pl.DataFrame(
        {
            "t_ns": [50, 100, 900, 950],
            "episode_id": [22, 23, 23, 24],
            "outcome_state": [3, 1, 2, 0],
            "outcome_info": ["collision", "", "", ""],
            "integrity": [["sim_death"], [], [], ["queued"]],
            "conditions": [_OLD_CONDITIONS, "[]", "[]", "[]"],
            "phases": [_OLD_PHASES, "", "", ""],
        }
    )


def test_episode_rows_keep_only_the_running_episodes_records() -> None:
    assert _episode_rows(_bag_records())["episode_id"].to_list() == [23, 23]


def test_episode_window_spans_the_running_episode_only() -> None:
    assert _episode_window(_bag_records()) == (100, 900)


def test_episode_window_keeps_its_end_on_equal_time_stamps() -> None:
    record = pl.DataFrame(
        {
            "t_ns": [100, 100],
            "episode_id": [23, 23],
            "outcome_state": [1, 2],
            "outcome_info": ["", ""],
            "integrity": [[], []],
            "conditions": ["[]", "[]"],
            "phases": ["", ""],
        }
    )
    assert _episode_window(record) == (100, 100)


def test_record_conditions_ignore_the_previous_episodes_latched_record() -> None:
    assert _record_conditions(_bag_records()) is None


def test_robot_phases_ignore_the_previous_episodes_latched_record() -> None:
    assert _robot_phases(_episode_rows(_bag_records()), "env_0_jackal") is None


def test_record_integrity_ignores_the_next_episodes_queued_record() -> None:
    assert _record_integrity(_bag_records(), None, None) == []
