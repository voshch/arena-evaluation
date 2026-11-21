import pytest
from robot_params_factory import make_robot_params

import polars as pl

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.ecological.semantic_metrics import (
    SemanticInteractionMetricsCalculator,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from snapshot_rows import snapshot_frame


def _calc():
    return SemanticInteractionMetricsCalculator(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )


def _episode(snapshot, end_t_ns=6_000_000_000):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0, end_t_ns]}),
        start_pos=[],
        goal_pos=[],
        semantic_snapshot=snapshot,
    )


def _robot_scope(episode):
    """Robot-scope values, computed the way the registry hands the robot slice over."""
    scope_slice = ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot",
        t_start_ns=0,
        t_end_ns=6_000_000_000,
        identity={},
        window_bundle=episode,
        full_episode=episode,
    )
    return _calc().calculate_for_scope(episode, {}, scope_slice)


def test_no_semantic_snapshot_is_undefined():
    results = _calc().calculate(_episode(None), {})

    assert results["time_waiting_at_doors_anyone"] is None
    assert results["elevator_rides_anyone"] is None


def _members(rows):
    """Members rows: (t_ns, entity, kind, [names])."""
    return snapshot_frame(
        [(r[0], r[1], r[2], "members", "members", r[3]) for r in rows]
    )


def _with_members(snapshot, members):
    return pl.concat([snapshot, members], how="vertical")


def _robot_episode(snapshot, robot_name="env_0_jackal"):
    episode = _episode(snapshot)
    episode.robot_name = robot_name
    return episode


def _door_snapshot():
    return snapshot_frame(
        [
            (1_000_000_000, "env_0/door_1/0", "door", "triggered", "predicate", True),
            (1_000_000_000, "env_0/door_1/0", "door", "state", "discrete", "closed"),
            (4_000_000_000, "env_0/door_1/0", "door", "state", "discrete", "open"),
        ]
    )


def test_time_waiting_at_doors_robot_only_while_robot_is_member():
    members = _members(
        [
            (0, "env_0/door_1/0", "door", []),
            (1_000_000_000, "env_0/door_1/0", "door", ["env_0/Pedestrian_1"]),
            (
                2_000_000_000,
                "env_0/door_1/0",
                "door",
                ["env_0/Pedestrian_1", "env_0/jackal"],
            ),
        ]
    )
    episode = _robot_episode(_with_members(_door_snapshot(), members))

    assert _calc().calculate(episode, {})[
        "time_waiting_at_doors_anyone"
    ] == pytest.approx(3.0)
    assert _robot_scope(episode)["time_waiting_at_doors_robot"] == pytest.approx(2.0)


def test_time_waiting_at_doors_robot_none_without_robot_name():
    members = _members([(2_000_000_000, "env_0/door_1/0", "door", ["env_0/jackal"])])

    assert (
        _robot_scope(_episode(_with_members(_door_snapshot(), members)))[
            "time_waiting_at_doors_robot"
        ]
        is None
    )


def test_elevator_rides_robot_counts_arrivals_with_robot_aboard():
    snapshot = snapshot_frame(
        [
            (0, "env_0/elev/1", "elevator", "just_arrived", "predicate", False),
            (1_500_000_000, "env_0/elev/1", "elevator", "occupants", "continuous", 1.0),
            (
                2_000_000_000,
                "env_0/elev/1",
                "elevator",
                "just_arrived",
                "predicate",
                True,
            ),
            (
                3_000_000_000,
                "env_0/elev/1",
                "elevator",
                "just_arrived",
                "predicate",
                False,
            ),
            (
                4_000_000_000,
                "env_0/elev/1",
                "elevator",
                "just_arrived",
                "predicate",
                True,
            ),
        ]
    )
    members = _members(
        [
            (0, "env_0/elev/1", "elevator", []),
            (3_000_000_000, "env_0/elev/1", "elevator", ["env_0/jackal"]),
            (3_500_000_000, "env_0/elev/1", "elevator", []),
            (4_000_000_000, "env_0/elev/1", "elevator", ["env_0/Pedestrian_2"]),
        ]
    )
    episode = _robot_episode(_with_members(snapshot, members))

    assert _calc().calculate(episode, {})["elevator_rides_anyone"] == 2
    assert _robot_scope(episode)["elevator_rides_robot"] == 1


def test_elevator_rides_robot_zero_when_no_rides():
    snapshot = snapshot_frame(
        [(0, "env_0/elev/1", "elevator", "just_arrived", "predicate", False)]
    )
    members = _members([(0, "env_0/elev/1", "elevator", [])])
    robot = _robot_scope(_robot_episode(_with_members(snapshot, members)))

    assert robot["elevator_rides_robot"] == 0
    assert robot["time_waiting_at_doors_robot"] is None
