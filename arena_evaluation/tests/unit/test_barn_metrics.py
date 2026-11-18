import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from arena_evaluation.processing.metrics.performance.barn_metrics import (
    BARNScoreCalculator,
    parse_reference,
    score,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

PATH_0 = (
    Path(__file__).parent.parent
    / "fixtures"
    / "barn"
    / "jackal_helper"
    / "worlds"
    / "BARN"
    / "path_files"
    / "path_0.npy"
)
X_SHIFT = 4.5
START = (-2.25 + X_SHIFT, 3.0)
GOAL = (-2.25 + X_SHIFT, 13.0)
RATE_HZ = 30.0
NULLS = {
    "barn_score": None,
    "barn_optimal_time": None,
    "barn_time": None,
    "barn_success": None,
    "barn_collided": None,
    "barn_timeout": None,
}


def _upstream_path_length() -> float:
    radius = 0.075
    points = [
        (x * radius * 2 - radius - 30 * radius * 2, y * radius * 2 + radius + 5)
        for x, y in np.load(PATH_0)
    ]
    points = [(-2.25, 3.0), *points, (-2.25, 13.0)]
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True))


def _upstream_run(
    position, collision_at: float | None, start_time: float, path_length: float
) -> tuple[bool, bool, bool, float, float]:
    """run.py sections 2 and 3 on a position function of sim time, polled every 0.1 s."""

    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    curr_time = start_time
    curr_coor = position(curr_time)
    collided = False
    poll = 0
    while (
        distance(GOAL, curr_coor) > 1 and not collided and curr_time - start_time < 100
    ):
        curr_time = start_time + 0.1 * poll
        curr_coor = position(curr_time)
        collided = collision_at is not None and collision_at <= curr_time
        poll += 1
    success = False
    if collided:
        pass
    elif curr_time - start_time >= 100:
        pass
    else:
        success = True
    optimal_time = path_length / 2
    actual_time = curr_time - start_time
    nav_metric = (
        int(success)
        * optimal_time
        / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)
    )
    return (
        success,
        collided,
        (curr_time - start_time) >= 100,
        actual_time,
        float(nav_metric),
    )


def _drive(speed: float, wait: float = 0.0, stop_at: float | None = None):
    """Position of a robot that waits at the start, then drives the straight line to the goal, optionally halting."""

    def position(t: float) -> tuple[float, float]:
        travelled = max(0.0, t - wait) * speed
        if stop_at is not None:
            travelled = min(travelled, stop_at)
        return START[0], START[1] + min(travelled, 10.0)

    return position


def _episode(
    position,
    duration: float,
    collision_at: float | None = None,
    world: str = "barn_0",
    t0: float = 5.0,
) -> AlignedEpisodeBundle:
    stamps = t0 + np.arange(0.0, duration + 1e-9, 1.0 / RATE_HZ)
    xy = np.array([position(t - t0) for t in stamps])
    t_ns = np.round(stamps * 1e9).astype(np.int64)
    robot_tracks = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "yaw": np.full(len(stamps), math.pi / 2),
            "pose_source": ["tf_gt"] * len(stamps),
        }
    )
    hits = (
        np.zeros(len(stamps), dtype=np.int64)
        if collision_at is None
        else (stamps - t0 >= collision_at).astype(np.int64)
    )
    collisions = pl.DataFrame({"t_ns": t_ns, "event_count": hits})
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[*START, math.pi / 2],
        goal_pos=list(GOAL),
        map=world,
        topics={"robot_tracks": robot_tracks, "robot_collision_events": collisions},
    )


def _config(path_length: float) -> dict:
    return {
        "barn": parse_reference(
            {
                "start": list(START),
                "goal": list(GOAL),
                "path_length": {"barn_0": path_length},
            }
        )
    }


def _calc(path_length: float | None) -> BARNScoreCalculator:
    calc = BARNScoreCalculator(make_robot_params(model="jackal"))
    if path_length is not None:
        calc.metrics_config = _config(path_length)
    return calc


def _first_motion(position, wait: float) -> float:
    stamps = np.arange(0.0, wait + 10.0, 1.0 / RATE_HZ)
    return float(next(t for t in stamps if math.dist(position(t), START) >= 0.1))


@pytest.mark.parametrize(
    ("speed", "wait", "stop_at", "collision_at", "duration"),
    [
        (2.0, 0.0, None, None, 6.0),
        (1.0, 1.7, None, None, 12.0),
        (0.5, 0.4, None, None, 20.0),
        (0.2, 0.0, None, None, 60.0),
        (0.05, 3.0, None, None, 108.0),
        (1.0, 0.5, None, 4.2, 4.3),
        (1.0, 0.5, 5.0, None, 104.0),
        (0.5, 0.0, None, 17.95, 18.5),
    ],
)
def test_recorded_run_scores_what_the_challenge_loop_gives(
    speed: float,
    wait: float,
    stop_at: float | None,
    collision_at: float | None,
    duration: float,
):
    path_length = _upstream_path_length()
    position = _drive(speed, wait, stop_at)
    start_time = _first_motion(position, wait)
    success, collided, timeout, actual_time, nav_metric = _upstream_run(
        position, collision_at, start_time, path_length
    )
    results = _calc(path_length).calculate(
        _episode(position, duration, collision_at), {}
    )
    assert results["barn_success"] is success
    assert results["barn_collided"] is collided
    assert results["barn_timeout"] is timeout
    assert results["barn_time"] == pytest.approx(actual_time, abs=1e-6)
    assert results["barn_score"] == pytest.approx(nav_metric, abs=1e-9)
    assert results["barn_optimal_time"] == pytest.approx(path_length / 2)
    assert success + collided + timeout == 1


def test_parametrized_runs_cover_every_challenge_outcome():
    path_length = _upstream_path_length()
    outcomes = set()
    for speed, wait, stop_at, collision_at in (
        (2.0, 0.0, None, None),
        (1.0, 0.5, None, 4.2),
        (1.0, 0.5, 5.0, None),
    ):
        position = _drive(speed, wait, stop_at)
        outcomes.add(
            _upstream_run(
                position, collision_at, _first_motion(position, wait), path_length
            )[:3]
        )
    assert outcomes == {
        (True, False, False),
        (False, True, False),
        (False, False, True),
    }


def test_failed_episode_scores_zero():
    assert score(False, 10.0, 10.0) == 0.0
    results = _calc(10.0).calculate(_episode(_drive(1.0), 4.0, collision_at=3.0), {})
    assert results["barn_score"] == 0.0
    assert results["barn_collided"] is True
    assert results["barn_optimal_time"] == pytest.approx(5.0)


def test_goal_is_reached_one_meter_out_not_at_the_goal():
    results = _calc(10.0).calculate(_episode(_drive(1.0, stop_at=9.2), 12.0), {})
    assert results["barn_success"] is True
    assert results["barn_time"] == pytest.approx(9.0, abs=0.11)
    short = _calc(10.0).calculate(_episode(_drive(1.0, stop_at=8.9), 12.0), {})
    assert short["barn_success"] is False


def test_collision_after_reaching_the_goal_radius_keeps_the_success():
    results = _calc(10.0).calculate(_episode(_drive(1.0), 10.0, collision_at=9.6), {})
    assert results["barn_success"] is True
    assert results["barn_collided"] is False


def test_robot_that_never_moves_scores_zero_with_no_time():
    results = _calc(10.0).calculate(_episode(_drive(0.0), 20.0), {})
    assert results["barn_score"] == 0.0
    assert results["barn_time"] is None
    assert (
        results["barn_success"],
        results["barn_collided"],
        results["barn_timeout"],
    ) == (False, False, False)


def test_missing_pose_yields_null():
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=[],
        map="barn_0",
        topics={},
    )
    assert _calc(10.0).calculate(episode, {}) == NULLS


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"start": [0, 0], "goal": [0, 1]},
        {"start": [0, 0], "goal": [0, 1], "path_length": {}},
        {"start": [0], "goal": [0, 1], "path_length": {"a": 1.0}},
        {"start": [0, 0], "goal": [0, 1], "path_length": {"a": 0.0}},
        {"start": [0, 0], "goal": [0, 1], "path_length": {"a": "long"}},
        {"start": [0, 0], "goal": [0, 1], "path_length": {"a": 1.0}, "speed": 2},
    ],
)
def test_reference_block_rejects_junk(raw: object):
    with pytest.raises(ValueError, match="barn"):
        parse_reference(raw)


def _robot_slice(episode):
    """Robot slice of the whole episode, the shape scoped_evaluator hands the ROBOT pass."""
    return ScopeSlice(
        scope=MetricScope.ROBOT,
        slice_id="robot",
        t_start_ns=0,
        t_end_ns=0,
        identity={},
        window_bundle=episode,
        full_episode=episode,
    )


def test_registry_reports_the_score_beside_the_native_metrics():
    episode = _episode(_drive(1.0), 10.0)
    topics = {"robot_tracks", "robot_odom", "robot_collision_events"}
    with_reference = MetricRegistry(
        make_robot_params(model="jackal"), metrics_config=_config(10.0)
    )
    assert with_reference.calculators["barn_score"].metrics_config == _config(10.0)
    results = {
        **with_reference.run(episode, available_topics=topics, pedsim_available=False),
        **with_reference.run_scope(
            MetricScope.ROBOT,
            episode,
            available_topics=topics,
            pedsim_available=False,
            scope_slice=_robot_slice(episode),
        ).results,
    }
    assert results["barn_success"] is True
    assert results["barn_score"] == pytest.approx(0.5)
    assert {"success", "collision_amount", "time_to_goal"} <= set(results)
    without_registry = MetricRegistry(make_robot_params(model="jackal"))
    without_episode = _episode(_drive(1.0), 10.0)
    without = {
        **without_registry.run(
            without_episode, available_topics=topics, pedsim_available=False
        ),
        **without_registry.run_scope(
            MetricScope.ROBOT,
            without_episode,
            available_topics=topics,
            pedsim_available=False,
            scope_slice=_robot_slice(without_episode),
        ).results,
    }
    assert {key: without[key] for key in NULLS} == NULLS
    skipped_episode = _episode(_drive(1.0), 10.0)
    skipped = {
        **with_reference.run(
            skipped_episode,
            available_topics={"robot_tracks", "robot_odom"},
            pedsim_available=False,
        ),
        **with_reference.run_scope(
            MetricScope.ROBOT,
            skipped_episode,
            available_topics={"robot_tracks", "robot_odom"},
            pedsim_available=False,
            scope_slice=_robot_slice(skipped_episode),
        ).results,
    }
    assert {key: skipped[key] for key in NULLS if key != "barn_optimal_time"} == {
        key: NULLS[key] for key in NULLS if key != "barn_optimal_time"
    }
    assert skipped["barn_optimal_time"] == pytest.approx(5.0)


