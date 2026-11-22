"""An episode whose robot left no trajectory still gets its invalid episode fact row."""

from arena_evaluation.processing.scoped_evaluator import unaligned_episode_facts
from arena_evaluation.storage.registry.facts import FactEpisodes


def test_an_episode_without_a_trajectory_gets_an_invalid_row_with_a_reason_for_every_null() -> None:
    facts = unaligned_episode_facts(run_id="run", episode_id=5, robot_id="env_0_jackal", result="CANCELLED", outcome_state=4, integrity=())

    (row,) = facts.episodes.to_dicts()
    assert (row["run_id"], row["episode_id"], row["robot_id"], row["result"]) == ("run", 5, "env_0_jackal", "CANCELLED")
    assert row["invalid_reason"] == "invalid_episode:missing_stream:robot_tracks;invalid_episode:incomplete_recording"
    reasons = dict(part.split(":", 1) for part in row["null_reason"].split(";"))
    assert set(reasons) == {column for column in FactEpisodes.null_reason_columns() if row[column] is None}
    assert set(reasons.values()) == {"topic_missing:robot_tracks"}
    assert facts.robots.is_empty() and facts.encounters.is_empty()
