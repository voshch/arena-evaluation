"""Contact rows of the scoped evaluator: robot-actor frames, and one detector for count and rows."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.processing.scoped_evaluator import (
    ScopedEvaluator,
)
from arena_evaluation.processing.windows import (
    contact_windows_lf,
    empty_encounter_windows,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

NS_PER_S = 1_000_000_000
ROBOT_RADIUS_M = 0.3
OTHER_ROBOT_X_M = 2.0
OTHER_ROBOT_Y_M = 1.0
SAMPLES = 11


def _robot_track(*, samples: int = SAMPLES, step_ns: int = NS_PER_S) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [i * step_ns for i in range(samples)],
            "env_id": ["env_0"] * samples,
            "robot_id": ["env_0_jackal_0"] * samples,
            "x": [0.0] * samples,
            "y": [0.0] * samples,
            "yaw": [0.0] * samples,
            "vx": [0.0] * samples,
            "vy": [0.0] * samples,
        }
    )


def _other_robot_task_poses() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t_ns": [i * NS_PER_S for i in range(SAMPLES)],
            "pos_x": [OTHER_ROBOT_X_M] * SAMPLES,
            "pos_y": [OTHER_ROBOT_Y_M] * SAMPLES,
            "radius_m": [ROBOT_RADIUS_M] * SAMPLES,
        }
    )


def _bundle(
    topics: dict[str, pl.DataFrame], fleet: dict[str, pl.DataFrame] | None = None
) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=0,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0],
        goal_pos=[10.0, 0.0],
        robot_name="env_0_jackal_0",
        topics=topics,
        fleet=fleet or {},
        map="world",
        outcome_state=2,
        outcome_info="finished",
    )


def _robot_contact_windows(obstacle_id: str) -> pl.DataFrame:
    intervals = pl.DataFrame(
        {
            "t_onset_ns": [6 * NS_PER_S],
            "t_end_ns": [6 * NS_PER_S],
            "collider_kind": ["robot"],
            "obstacle_id": [obstacle_id],
            "obstacle_x": [None],
            "obstacle_y": [None],
        },
        schema_overrides={"obstacle_x": pl.Float64, "obstacle_y": pl.Float64},
    )
    return contact_windows_lf(intervals, empty_encounter_windows()).collect()


@pytest.mark.parametrize(
    "obstacle_id",
    ["env_0_jackal_1", "env_0/jackal_1", "jackal_1"],
    ids=["recorded_id", "slash_id", "bare_name"],
)
def test_a_robot_contact_window_carries_the_other_robots_track(
    obstacle_id: str,
) -> None:
    episode = _bundle(
        {"robot_tracks": _robot_track()},
        fleet={"env_0_jackal_1": _other_robot_task_poses()},
    )
    evaluator = ScopedEvaluator(make_robot_params(robot_radius=ROBOT_RADIUS_M))

    slices = evaluator._build_contact_slices(
        episode,
        _robot_contact_windows(obstacle_id),
        {},
        run_id="run",
        episode_id=0,
        robot_id="env_0_jackal_0",
        robot_token="jackal_0",
    )

    assert [contact.identity["collider_kind"] for contact in slices] == ["robot"]
    assert [contact.identity["obstacle_id"] for contact in slices] == ["env_0_jackal_1"]
    actor_track = slices[0].window_bundle.topics["pedestrian_arena_peds"]
    assert actor_track["x"].to_list() == [OTHER_ROBOT_X_M]


def test_the_contact_count_matches_the_contact_rows_when_only_the_bumper_stream_is_recorded() -> (
    None
):
    episode = _bundle(
        {
            "robot_tracks": _robot_track(),
            "robot_collision_events": pl.DataFrame(
                {"t_ns": [5 * NS_PER_S], "wall_count": [1], "static_count": [0]}
            ),
        }
    )

    result = ScopedEvaluator(make_robot_params(robot_radius=ROBOT_RADIUS_M)).evaluate(
        episode,
        run_id="run",
        episode_id=0,
        world=None,
        episode_context={},
        pedsim_available=False,
        available_topics=set(episode.topics or {}),
    )

    assert result.episode_metrics["contact_count"] == 1
    assert result.facts.contacts.get_column("collider_kind").to_list() == ["wall"]
    assert len(result.facts.contacts) == result.episode_metrics["contact_count"]


def test_the_collisions_table_reads_the_recorded_contact_stream_of_its_obstacles() -> (
    None
):
    contacts = pl.DataFrame(
        {
            "t_ns": [5 * NS_PER_S],
            "env_id": ["env_0"],
            "robot_id": ["env_0_jackal_0"],
            "obstacle_id": ["Pedestrian_0"],
            "kind": ["pedestrian"],
            "obstacle_x": [1.0],
            "obstacle_y": [2.0],
            "obstacle_z": [0.0],
            "distance": [0.0],
        }
    )
    episode = _bundle(
        {"robot_tracks": _robot_track(), "robot_collision_contacts": contacts}
    )

    frame = ScopedEvaluator(
        make_robot_params(robot_radius=ROBOT_RADIUS_M)
    )._collision_fact_frame(
        episode,
        run_id="run",
        episode_id=0,
        robot_id="env_0_jackal_0",
        metrics={"collision_obstacles": ["Pedestrian_0"]},
    )

    assert frame.get_column("obstacle_id").to_list() == ["Pedestrian_0"]


def _pedestrian_contact_messages(t_ns: list[int]) -> pl.DataFrame:
    count = len(t_ns)
    return pl.DataFrame(
        {
            "t_ns": t_ns,
            "env_id": ["env_0"] * count,
            "robot_id": ["env_0_jackal_0"] * count,
            "obstacle_id": ["Pedestrian_0"] * count,
            "kind": ["pedestrian"] * count,
            "obstacle_x": [1.0] * count,
            "obstacle_y": [2.0] * count,
            "obstacle_z": [0.0] * count,
            "distance": [0.0] * count,
        }
    )


def _evaluated_contact_window_lengths_ns(episode: AlignedEpisodeBundle) -> list[int]:
    result = ScopedEvaluator(make_robot_params(robot_radius=ROBOT_RADIUS_M)).evaluate(
        episode,
        run_id="run",
        episode_id=0,
        world=None,
        episode_context={},
        pedsim_available=False,
        available_topics=set(episode.topics or {}),
    )
    return (
        result.facts.contacts.select(pl.col("t_end_ns") - pl.col("t_onset_ns"))
        .to_series()
        .to_list()
    )


def test_a_pedestrian_contact_seen_at_one_robot_sample_lasts_one_robot_period() -> None:
    episode = _bundle(
        {
            "robot_tracks": _robot_track(),
            "robot_collision_contacts": _pedestrian_contact_messages([5 * NS_PER_S]),
        }
    )

    assert _evaluated_contact_window_lengths_ns(episode) == [NS_PER_S]


def test_a_bumper_contact_seen_at_one_robot_sample_lasts_one_robot_period() -> None:
    episode = _bundle(
        {
            "robot_tracks": _robot_track(),
            "robot_collision_events": pl.DataFrame(
                {"t_ns": [5 * NS_PER_S], "wall_count": [1], "static_count": [0]}
            ),
        }
    )

    assert _evaluated_contact_window_lengths_ns(episode) == [NS_PER_S]


def test_a_contact_spanning_several_robot_samples_keeps_its_span() -> None:
    tenth_s_ns = NS_PER_S // 10
    episode = _bundle(
        {
            "robot_tracks": _robot_track(samples=80, step_ns=tenth_s_ns),
            "robot_collision_contacts": _pedestrian_contact_messages(
                [5 * NS_PER_S, 5 * NS_PER_S + tenth_s_ns, 5 * NS_PER_S + 2 * tenth_s_ns]
            ),
        }
    )

    assert _evaluated_contact_window_lengths_ns(episode) == [2 * tenth_s_ns]
