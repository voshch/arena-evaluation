import polars as pl
import pytest
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.processing.pipeline import (
    _metadata_robot,
    _record_conditions,
    _record_outcome,
    _status_row,
)
from arena_evaluation.storage.schemas import EpisodeDescriptor, RunMetadata, TopicBundle


def _ep() -> EpisodeDescriptor:
    return EpisodeDescriptor(
        episode_dir="/nowhere",
        benchmark_id="b",
        episode_id=8,
        planner="bev-policy",
        stage="s8",
        map="scene_08",
    )


def _meta(outcome_state: int | None, outcome_info: str = "") -> RunMetadata:
    return RunMetadata(
        benchmark_id="b",
        planner="bev-policy",
        map="scene_08",
        stage="s8",
        robot_model=["jackal"],
        outcome_state=outcome_state,
        outcome_info=outcome_info,
        recording_started_at="2026-01-01T00:00:00+00:00",
        python_version="3.12",
        ros_distro="jazzy",
        modes={},
        params={},
    )


@pytest.mark.parametrize(
    ("state", "info", "expected"),
    [
        (2, "", ("GOAL_REACHED", True)),
        (3, "", ("FAILED", False)),
        (4, "", ("CANCELLED", False)),
        (5, "", ("FATAL", False)),
        (3, "collision", ("COLLISION", False)),
        (1, "", ("UNRESOLVED", False)),
        (0, "", ("UNRESOLVED", False)),
        (None, "", ("UNRESOLVED", False)),
    ],
)
def test_outcome_verdict(
    state: int | None, info: str, expected: tuple[str, bool]
) -> None:
    assert outcome_verdict(state, info) == expected


def test_record_outcome_prefers_last_record_over_metadata() -> None:
    record = pl.DataFrame(
        {
            "episode_id": [3, 3],
            "t_ns": [5, 0],
            "outcome_state": [3, 1],
            "outcome_info": ["collision", ""],
        }
    )
    assert _record_outcome(record, _meta(2)) == (3, "collision")
    assert _record_outcome(None, _meta(2, "finished")) == (2, "finished")
    assert _record_outcome(None, None) == (None, None)


def test_record_conditions_skip_the_next_episodes_empty_reset_records() -> None:
    kept = '[{"op": "always", "p": {"each": "chest.openness", "le": 0.05}}]'
    record = pl.DataFrame(
        {
            "episode_id": [1, 1, 1, 1],
            "t_ns": [0, 2, 9, 9],
            "outcome_state": [0, 1, 2, 2],
            "conditions_yaml": ["[]", kept, "[]", "not json"],
        }
    )
    assert _record_conditions(record) == [
        {"op": "always", "p": {"each": "chest.openness", "le": 0.05}}
    ]
    ended = pl.DataFrame(
        {
            "episode_id": [3, 3, 3, 4],
            "t_ns": [0, 2, 9, 9],
            "outcome_state": [0, 1, 2, 0],
            "outcome_info": ["", "", "", ""],
            "conditions_yaml": [
                "[]",
                kept,
                kept,
                '[{"op": "never", "p": "robot in x"}]',
            ],
        }
    )
    assert _record_conditions(ended) == [
        {"op": "always", "p": {"each": "chest.openness", "le": 0.05}}
    ]
    assert (
        _record_conditions(
            pl.DataFrame(
                {
                    "episode_id": [1],
                    "t_ns": [0],
                    "outcome_state": [2],
                    "conditions_yaml": ["[]"],
                }
            )
        )
        is None
    )
    assert (
        _record_conditions(
            pl.DataFrame(
                {
                    "episode_id": [1],
                    "t_ns": [0],
                    "outcome_state": [2],
                    "conditions_yaml": [""],
                }
            )
        )
        is None
    )
    assert (
        _record_conditions(
            pl.DataFrame(
                {
                    "episode_id": [1],
                    "t_ns": [0],
                    "outcome_state": [1],
                    "conditions_yaml": ["not json"],
                }
            )
        )
        is None
    )
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        _record_conditions(
            pl.DataFrame({"episode_id": [1], "t_ns": [0], "outcome_state": [1]})
        )


def test_record_outcome_ignores_the_next_episodes_queued_record_at_the_same_time() -> (
    None
):
    record = pl.DataFrame(
        {
            "episode_id": [3, 3, 3, 4, 4],
            "t_ns": [0, 1, 9, 9, 9],
            "outcome_state": [0, 1, 2, 0, 0],
            "outcome_info": ["", "", "finished", "", ""],
        }
    )
    assert _record_outcome(record, None) == (2, "finished")


def test_status_row_carries_identity_and_verdict() -> None:
    row = _status_row(
        _ep(), _meta(3, "collision"), "jackal", "no_trajectory", "no odom rows"
    )
    assert row["episode"] == 8
    assert row["stage"] == "s8"
    assert row["robot"] == "jackal"
    assert row["status"] == "no_trajectory"
    assert row["status_reason"] == "no odom rows"
    assert row["result"] == "COLLISION"
    assert row["success"] is False
    assert row["local_planner"] == "bev"
    assert "time" not in row


def test_metadata_robot_uses_bundle_key_style() -> None:
    meta = _meta(3)
    meta.robot_model = ["jackal[top=camera/oakd_rgbd]", "jackal"]
    assert _metadata_robot(meta, {"env_0": TopicBundle()}) == "env_0_jackal"
    assert _metadata_robot(meta, None) == "jackal"
    assert _metadata_robot(None, {"env_0": TopicBundle()}) == ""


def test_record_outcome_skips_the_previous_episodes_latched_record() -> None:
    record = pl.DataFrame(
        {
            "episode_id": [3, 3],
            "t_ns": [50, 100],
            "outcome_state": [3, 1],
            "outcome_info": ["timeout", ""],
        }
    )
    assert _record_outcome(record, _meta(2, "finished after 77.2s")) == (
        2,
        "finished after 77.2s",
    )
    assert _record_outcome(record, None) == (1, "")
    with_terminal = pl.DataFrame(
        {
            "episode_id": [3, 3, 3],
            "t_ns": [50, 100, 900],
            "outcome_state": [3, 1, 2],
            "outcome_info": ["timeout", "", "done"],
        }
    )
    assert _record_outcome(with_terminal, _meta(3)) == (2, "done")
