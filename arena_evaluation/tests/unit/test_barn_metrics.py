import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from arena_evaluation.processing.metrics.performance import barn_metrics
from arena_evaluation.processing.metrics.performance.barn_metrics import BARNScoreCalculator, parse_reference, replay, score
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

PATH_0 = Path(__file__).parent.parent / "fixtures" / "barn" / "jackal_helper" / "worlds" / "BARN" / "path_files" / "path_0.npy"
X_SHIFT = 4.5
START = (-2.25 + X_SHIFT, 3.0)
GOAL = (-2.25 + X_SHIFT, 13.0)
RATE_HZ = 30.0
NULLS = {"barn_score": None, "barn_optimal_time": None, "barn_time": None, "barn_success": None, "barn_collided": None, "barn_timeout": None}


def _upstream_path_length() -> float:
    radius = 0.075
    points = [(x * radius * 2 - radius - 30 * radius * 2, y * radius * 2 + radius + 5) for x, y in np.load(PATH_0)]
    points = [(-2.25, 3.0), *points, (-2.25, 13.0)]
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True))


def _upstream_run(position, collision_at: float | None, start_time: float, path_length: float) -> tuple[bool, bool, bool, float, float]:
    """run.py sections 2 and 3 on a position function of sim time, polled every 0.1 s."""

    def distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    curr_time = start_time
    curr_coor = position(curr_time)
    collided = False
    poll = 0
    while distance(GOAL, curr_coor) > 1 and not collided and curr_time - start_time < 100:
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
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)
    return success, collided, (curr_time - start_time) >= 100, actual_time, float(nav_metric)


def _drive(speed: float, wait: float = 0.0, stop_at: float | None = None):
    """Position of a robot that waits at the start, then drives the straight line to the goal, optionally halting."""

    def position(t: float) -> tuple[float, float]:
        travelled = max(0.0, t - wait) * speed
        if stop_at is not None:
            travelled = min(travelled, stop_at)
        return START[0], START[1] + min(travelled, 10.0)

    return position


def _episode(position, duration: float, collision_at: float | None = None, world: str = "barn_0", t0: float = 5.0) -> AlignedEpisodeBundle:
    stamps = t0 + np.arange(0.0, duration + 1e-9, 1.0 / RATE_HZ)
    xy = np.array([position(t - t0) for t in stamps])
    time_ns = np.round(stamps * 1e9).astype(np.int64)
    tf_gt = pl.DataFrame({"time_ns": time_ns, "pos_x_gt": xy[:, 0], "pos_y_gt": xy[:, 1], "yaw_gt": np.full(len(stamps), math.pi / 2)})
    hits = np.zeros(len(stamps), dtype=np.int64) if collision_at is None else (stamps - t0 >= collision_at).astype(np.int64)
    collisions = pl.DataFrame({"time_ns": time_ns, "collision_event": hits})
    return AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[*START, math.pi / 2], goal_pos=list(GOAL), map=world, topics={"tf_gt": tf_gt, "collision_events": collisions})


def _config(path_length: float) -> dict:
    return {"barn": parse_reference({"start": list(START), "goal": list(GOAL), "path_length": {"barn_0": path_length}})}


def _calc(path_length: float | None, max_speed: float = 2.0) -> BARNScoreCalculator:
    calc = BARNScoreCalculator(RobotParams(model="jackal", max_linear_velocity=max_speed))
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
def test_recorded_run_scores_what_the_challenge_loop_gives(speed: float, wait: float, stop_at: float | None, collision_at: float | None, duration: float):
    path_length = _upstream_path_length()
    position = _drive(speed, wait, stop_at)
    start_time = _first_motion(position, wait)
    success, collided, timeout, actual_time, nav_metric = _upstream_run(position, collision_at, start_time, path_length)
    results = _calc(path_length).calculate(_episode(position, duration, collision_at), {})
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
    for speed, wait, stop_at, collision_at in ((2.0, 0.0, None, None), (1.0, 0.5, None, 4.2), (1.0, 0.5, 5.0, None)):
        position = _drive(speed, wait, stop_at)
        outcomes.add(_upstream_run(position, collision_at, _first_motion(position, wait), path_length)[:3])
    assert outcomes == {(True, False, False), (False, True, False), (False, False, True)}


def test_score_inside_clip_band():
    assert score(True, 10.0, 10.0) == pytest.approx(0.5)
    assert score(True, 20.0, 10.0) == pytest.approx(0.25)


def test_fast_run_caps_at_half():
    assert score(True, 4.0, 10.0) == pytest.approx(0.5)
    results = _calc(10.0).calculate(_episode(_drive(2.0), 6.0), {})
    assert results["barn_success"] is True
    assert results["barn_time"] < 10.0
    assert results["barn_score"] == pytest.approx(0.5)


def test_slow_run_floors_at_eighth():
    assert score(True, 100.0, 10.0) == pytest.approx(0.125)
    results = _calc(10.0).calculate(_episode(_drive(0.2), 60.0), {})
    assert results["barn_success"] is True
    assert results["barn_time"] > 40.0
    assert results["barn_score"] == pytest.approx(0.125)


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


def test_clock_starts_at_first_motion_not_at_recording_start():
    waited = _calc(10.0).calculate(_episode(_drive(1.0, wait=7.0), 17.0), {})
    prompt = _calc(10.0).calculate(_episode(_drive(1.0), 10.0), {})
    assert waited["barn_time"] == pytest.approx(prompt["barn_time"], abs=0.11)
    assert waited["barn_score"] == pytest.approx(prompt["barn_score"])


def test_collision_after_reaching_the_goal_radius_keeps_the_success():
    results = _calc(10.0).calculate(_episode(_drive(1.0), 10.0, collision_at=9.6), {})
    assert results["barn_success"] is True
    assert results["barn_collided"] is False


def test_recording_cut_before_any_outcome_scores_zero_with_no_flag():
    results = _calc(10.0).calculate(_episode(_drive(0.1), 30.0), {})
    assert results["barn_score"] == 0.0
    assert (results["barn_success"], results["barn_collided"], results["barn_timeout"]) == (False, False, False)
    assert results["barn_time"] == pytest.approx(29.0, abs=0.11)


def test_robot_that_never_moves_scores_zero_with_no_time():
    results = _calc(10.0).calculate(_episode(_drive(0.0), 20.0), {})
    assert results["barn_score"] == 0.0
    assert results["barn_time"] is None
    assert (results["barn_success"], results["barn_collided"], results["barn_timeout"]) == (False, False, False)


def test_speed_is_the_challenge_constant_whatever_the_robot():
    assert barn_metrics.MAX_SPEED_MPS == 2.0
    episode = _episode(_drive(1.0), 10.0)
    slow = _calc(10.0, max_speed=1.0).calculate(episode, {})
    unknown = _calc(10.0, max_speed=0.0).calculate(episode, {})
    assert slow["barn_optimal_time"] == pytest.approx(5.0)
    assert unknown["barn_optimal_time"] == pytest.approx(5.0)
    assert slow["barn_score"] == pytest.approx(unknown["barn_score"])


def test_no_reference_means_no_score():
    episode = _episode(_drive(1.0), 10.0)
    assert _calc(None).calculate(episode, {"success": True, "time_to_goal": 9.0, "theta_star_length": 10.0}) == NULLS
    assert _calc(10.0).calculate(_episode(_drive(1.0), 10.0, world="map_empty"), {}) == NULLS


def test_missing_pose_yields_null():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], map="barn_0", topics={})
    assert _calc(10.0).calculate(episode, {}) == NULLS


def test_replay_holds_the_last_pose_for_the_poll_after_the_recording():
    t = np.array([0.0, 0.05, 0.1, 0.12])
    y = np.array([3.0, 3.2, 11.5, 12.05])
    run = replay(t, np.full(4, START[0]), y, START, GOAL, None)
    assert run == (True, False, False, pytest.approx(0.1))
    assert replay(t[:3], np.full(3, START[0]), y[:3], START, GOAL, None) == (False, False, False, pytest.approx(0.05))
    assert replay(t[:3], np.full(3, START[0]), y[:3], START, GOAL, 0.11) == (False, True, False, pytest.approx(0.1))


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


def test_reference_block_normalizes_numbers():
    assert parse_reference({"start": [2, 3], "goal": (2.25, 13), "path_length": {"barn_0": 11}}) == {"start": [2.0, 3.0], "goal": [2.25, 13.0], "path_length": {"barn_0": 11.0}}


def test_registry_reports_the_score_beside_the_native_metrics():
    episode = _episode(_drive(1.0), 10.0)
    topics = {"tf_gt", "odom", "collision_events"}
    with_reference = MetricRegistry(RobotParams(model="jackal"), metrics_config=_config(10.0))
    assert with_reference.calculators["barn_score"].metrics_config == _config(10.0)
    results = with_reference.run(episode, available_topics=topics, pedsim_available=False)
    assert results["barn_success"] is True
    assert results["barn_score"] == pytest.approx(0.5)
    assert {"success", "collision_amount", "time_to_goal"} <= set(results)
    without = MetricRegistry(RobotParams(model="jackal")).run(_episode(_drive(1.0), 10.0), available_topics=topics, pedsim_available=False)
    assert {key: without[key] for key in NULLS} == NULLS
    skipped = with_reference.run(_episode(_drive(1.0), 10.0), available_topics={"tf_gt", "odom"}, pedsim_available=False)
    assert {key: skipped[key] for key in NULLS} == NULLS


def test_output_keys_cover_units_and_primary():
    keys = set(BARNScoreCalculator.output_keys())
    assert set(BARNScoreCalculator.UNITS) <= keys
    assert set(BARNScoreCalculator.PRIMARY_OUTPUTS) <= keys
    assert set(BARNScoreCalculator.OUTPUT_DIRECTIONS) <= keys
    assert keys == set(NULLS)
