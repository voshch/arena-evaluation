import math

import numpy as np
import polars as pl
import pytest
import shapely
from arena_evaluation.processing.metrics.ecological.compliance_metrics import _ZoneGeometry
from arena_evaluation.processing.metrics.ecological.condition_metrics import _world_cache
from arena_evaluation.processing.metrics.performance.sidewalk_metrics import SidewalkMetricsCalculator, excursions, freezes, walkable_area
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

WORLD = "sidewalk_synthetic"
BARE = "road_only_synthetic"
RATE_HZ = 30.0
T0 = 5.0
GOAL = [21.05, 0.0]
NULLS = dict.fromkeys(SidewalkMetricsCalculator.output_keys())


def _zone(name: str, x0: float, y0: float, x1: float, y1: float) -> _ZoneGeometry:
    return _ZoneGeometry(name=name, polygon=shapely.box(x0, y0, x1, y1), max_speed=None, quiet=False, restricted=False)


@pytest.fixture(autouse=True)
def zones():
    _world_cache[WORLD] = [
        _zone("sidewalk_0", 0, -2, 10, 2),
        _zone("crosswalk", 10, -2, 14, 2),
        _zone("sidewalk_1", 14, -2, 30, 2),
        _zone("road", 0, 2, 30, 10),
        _zone("curb", 0, -3, 30, -2),
    ]
    _world_cache[BARE] = [_zone("road", 0, 2, 30, 10)]
    yield
    _world_cache.pop(WORLD, None)
    _world_cache.pop(BARE, None)


def _calc(world: str = WORLD) -> SidewalkMetricsCalculator:
    calc = SidewalkMetricsCalculator(RobotParams(model="jackal"))
    calc.world = world
    return calc


def _episode(position, duration: float, collision_at: float | None = None, goal: list[float] | None = None) -> AlignedEpisodeBundle:
    stamps = T0 + np.arange(0.0, duration + 1e-9, 1.0 / RATE_HZ)
    xy = np.array([position(t - T0) for t in stamps])
    time_ns = np.round(stamps * 1e9).astype(np.int64)
    tf_gt = pl.DataFrame({"time_ns": time_ns, "pos_x_gt": xy[:, 0], "pos_y_gt": xy[:, 1], "yaw_gt": np.zeros(len(stamps))})
    hits = np.zeros(len(stamps), dtype=np.int64) if collision_at is None else (stamps - T0 >= collision_at).astype(np.int64)
    collisions = pl.DataFrame({"time_ns": time_ns, "collision_event": hits})
    return AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[0.0, 0.0, 0.0], goal_pos=GOAL if goal is None else goal, map=WORLD, topics={"tf_gt": tf_gt, "collision_events": collisions})


def _lane(t: float) -> tuple[float, float]:
    return t, 0.0


def _weave(t: float) -> tuple[float, float]:
    knots = [(0, 0), (5, 0), (6, 3), (8, 3), (9, 0), (13, 0), (14, 3), (15, 3), (16, 0), (30, 0)]
    return t, float(np.interp(t, [k[0] for k in knots], [k[1] for k in knots]))


def _stopping(start: float, end: float):
    def position(t: float) -> tuple[float, float]:
        return min(t, start) + max(0.0, t - end), 0.0

    return position


def _waiting(until: float):
    def position(t: float) -> tuple[float, float]:
        return max(0.0, t - until), 0.0

    return position


def test_clean_run_along_the_sidewalk_and_crosswalk_completes():
    results = _calc().calculate(_episode(_lane, 22.0), {})
    assert results["sidewalk_route_completion"] is True
    assert results["sidewalk_off_sidewalk"] is False
    assert results["sidewalk_freezes"] == 0
    assert results["sidewalk_failures_per_100m"] == 0.0
    assert results["sidewalk_mean_speed"] == pytest.approx(1.0, abs=1e-6)


def test_leaving_the_sidewalk_twice_and_one_collision_count_three_failures():
    results = _calc().calculate(_episode(_weave, 22.0, collision_at=10.0), {})
    path = 20.1 - 4.0 + 4 * math.sqrt(10.0)
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_off_sidewalk"] is True
    assert results["sidewalk_freezes"] == 0
    assert results["sidewalk_failures_per_100m"] == pytest.approx(3 / path * 100.0, abs=1e-6)
    assert results["sidewalk_mean_speed"] == pytest.approx(path / 20.1, abs=1e-6)


def test_off_sidewalk_alone_fails_the_route():
    results = _calc().calculate(_episode(_weave, 22.0), {})
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_failures_per_100m"] == pytest.approx(2 / (20.1 - 4.0 + 4 * math.sqrt(10.0)) * 100.0, abs=1e-6)


def test_collision_alone_fails_the_route():
    results = _calc().calculate(_episode(_lane, 22.0, collision_at=7.0), {})
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_off_sidewalk"] is False
    assert results["sidewalk_failures_per_100m"] == pytest.approx(1 / 20.1 * 100.0, abs=1e-6)


def test_driving_through_the_road_zone_is_off_the_sidewalk():
    def through_road(t: float) -> tuple[float, float]:
        return t, 0.0 if t < 3 else 4.0

    results = _calc().calculate(_episode(through_road, 22.0, goal=[21.05, 4.0]), {})
    assert results["sidewalk_off_sidewalk"] is True
    assert results["sidewalk_route_completion"] is False


def test_curb_is_not_walkable():
    def curb_walk(t: float) -> tuple[float, float]:
        return t, -2.5

    assert _calc().calculate(_episode(curb_walk, 22.0, goal=[21.05, -2.5]), {})["sidewalk_off_sidewalk"] is True


def test_stop_longer_than_five_seconds_is_one_freeze():
    results = _calc().calculate(_episode(_stopping(5.0, 13.0), 30.0), {})
    assert results["sidewalk_freezes"] == 1
    assert results["sidewalk_failures_per_100m"] == pytest.approx(1 / 20.1 * 100.0, abs=1e-6)
    assert results["sidewalk_route_completion"] is True


def test_stop_of_three_seconds_is_not_a_freeze():
    assert _calc().calculate(_episode(_stopping(5.0, 8.0), 30.0), {})["sidewalk_freezes"] == 0


def test_two_separate_stops_are_two_freezes():
    def position(t: float) -> tuple[float, float]:
        return float(np.interp(t, [0, 4, 11, 14, 21, 40], [0, 4, 4, 7, 7, 33])), 0.0

    assert _calc().calculate(_episode(position, 40.0), {})["sidewalk_freezes"] == 2


def test_standing_at_the_start_is_excused_for_two_seconds():
    assert _calc().calculate(_episode(_waiting(7.0), 40.0), {})["sidewalk_freezes"] == 0
    assert _calc().calculate(_episode(_waiting(8.0), 40.0), {})["sidewalk_freezes"] == 1


def test_freeze_window_is_exclusive_of_five_seconds():
    assert freezes(np.zeros(200), 20, 200, 50) == 1
    assert freezes(np.zeros(200), 20, 70, 50) == 0
    assert freezes(np.zeros(200), 20, 71, 50) == 1


def test_standing_at_the_goal_after_arrival_is_not_a_freeze():
    def arrive(t: float) -> tuple[float, float]:
        return min(t, 21.0), 0.0

    results = _calc().calculate(_episode(arrive, 60.0), {})
    assert results["sidewalk_freezes"] == 0
    assert results["sidewalk_route_completion"] is True


def test_goal_not_reached_is_not_a_completion_and_speed_spans_the_recording():
    def slow(t: float) -> tuple[float, float]:
        return 0.5 * t, 0.0

    results = _calc().calculate(_episode(slow, 20.0), {})
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_off_sidewalk"] is False
    assert results["sidewalk_mean_speed"] == pytest.approx(0.5, abs=1e-6)
    assert results["sidewalk_failures_per_100m"] == 0.0


def test_robot_that_never_moves_has_no_failure_rate():
    results = _calc().calculate(_episode(lambda t: (0.0, 0.0), 20.0), {})
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_freezes"] == 1
    assert results["sidewalk_failures_per_100m"] is None
    assert results["sidewalk_mean_speed"] == 0.0


def test_episode_without_goal_has_no_completion():
    results = _calc().calculate(_episode(_lane, 22.0, goal=[]), {})
    assert results["sidewalk_route_completion"] is None
    assert results["sidewalk_off_sidewalk"] is False
    assert results["sidewalk_mean_speed"] == pytest.approx(1.0, abs=1e-6)


def test_world_without_sidewalk_zones_yields_nulls():
    assert _calc(BARE).calculate(_episode(_lane, 22.0), {}) == NULLS
    assert SidewalkMetricsCalculator(RobotParams(model="jackal")).calculate(_episode(_lane, 22.0), {}) == NULLS


def test_missing_pose_yields_nulls():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=GOAL, map=WORLD, topics={})
    assert _calc().calculate(episode, {}) == NULLS


def test_walkable_area_unions_sidewalk_and_crosswalk_pieces_only():
    assert walkable_area(_world_cache[WORLD]).area == pytest.approx(30 * 4)
    assert walkable_area(_world_cache[BARE]) is None


def test_walkable_area_unions_a_self_touching_zone_ring():
    pinched = shapely.Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2.0001, 1.9999), (2, 2), (0, 2)])
    assert not pinched.is_valid
    zone = _ZoneGeometry(name="sidewalk_0", polygon=pinched, max_speed=None, quiet=False, restricted=False)
    assert walkable_area([zone, _zone("crosswalk", 4, 0, 6, 2)]).area == pytest.approx(12.0, abs=1e-3)


def test_excursions_merge_gaps_shorter_than_the_merge_window():
    outside = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=bool)
    assert excursions(outside, 2) == 3
    assert excursions(outside, 3) == 2
    assert excursions(outside, 5) == 2
    assert excursions(outside, 6) == 1
    assert excursions(np.zeros(5, dtype=bool), 2) == 0


def test_registry_runs_the_calculator_beside_the_native_metrics():
    topics = {"tf_gt", "odom", "collision_events"}
    registry = MetricRegistry(RobotParams(model="jackal"), world=WORLD)
    results = registry.run(_episode(_lane, 22.0), available_topics=topics, pedsim_available=False)
    assert results["sidewalk_route_completion"] is True
    assert {"success", "collision_amount", "time_to_goal"} <= set(results)
    skipped = registry.run(_episode(_lane, 22.0), available_topics={"tf_gt", "odom"}, pedsim_available=False)
    assert {key: skipped[key] for key in NULLS} == NULLS


def test_output_keys_cover_units_primary_and_directions():
    keys = set(SidewalkMetricsCalculator.output_keys())
    assert set(SidewalkMetricsCalculator.UNITS) == keys
    assert set(SidewalkMetricsCalculator.PRIMARY_OUTPUTS) <= keys
    assert set(SidewalkMetricsCalculator.OUTPUT_DIRECTIONS) <= keys
