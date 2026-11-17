
import numpy as np
import polars as pl
import pytest
import shapely
from arena_evaluation.processing.metrics.common.caching import world_cache
from arena_evaluation.processing.metrics.ecological.compliance_metrics import (
    _ZoneGeometry,
)
from arena_evaluation.processing.metrics.performance.sidewalk_metrics import (
    SidewalkMetricsCalculator,
)
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

WORLD = "sidewalk_synthetic"
BARE = "road_only_synthetic"
RATE_HZ = 30.0
T0 = 5.0
GOAL = [21.05, 0.0]
NULLS = dict.fromkeys(SidewalkMetricsCalculator.output_keys())


def _zone(name: str, x0: float, y0: float, x1: float, y1: float) -> _ZoneGeometry:
    return _ZoneGeometry(
        name=name,
        polygon=shapely.box(x0, y0, x1, y1),
        max_speed=None,
        quiet=False,
        restricted=False,
    )


@pytest.fixture(autouse=True)
def zones():
    world_cache[WORLD] = [
        _zone("sidewalk_0", 0, -2, 10, 2),
        _zone("crosswalk", 10, -2, 14, 2),
        _zone("sidewalk_1", 14, -2, 30, 2),
        _zone("road", 0, 2, 30, 10),
        _zone("curb", 0, -3, 30, -2),
    ]
    world_cache[BARE] = [_zone("road", 0, 2, 30, 10)]
    yield
    world_cache.pop(WORLD, None)
    world_cache.pop(BARE, None)


def _calc(world: str = WORLD) -> SidewalkMetricsCalculator:
    calc = SidewalkMetricsCalculator(make_robot_params(model="jackal"))
    calc.world = world
    return calc


def _episode(
    position,
    duration: float,
    collision_at: float | None = None,
    goal: list[float] | None = None,
) -> AlignedEpisodeBundle:
    stamps = T0 + np.arange(0.0, duration + 1e-9, 1.0 / RATE_HZ)
    xy = np.array([position(t - T0) for t in stamps])
    time_ns = np.round(stamps * 1e9).astype(np.int64)
    tracks = pl.DataFrame(
        {"t_ns": time_ns, "x": xy[:, 0], "y": xy[:, 1], "yaw": np.zeros(len(stamps))}
    )
    hits = (
        np.zeros(len(stamps), dtype=np.int64)
        if collision_at is None
        else (stamps - T0 >= collision_at).astype(np.int64)
    )
    collisions = pl.DataFrame({"t_ns": time_ns, "event_count": hits})
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=GOAL if goal is None else goal,
        map=WORLD,
        topics={"robot_tracks": tracks, "robot_collision_events": collisions},
    )


def _lane(t: float) -> tuple[float, float]:
    return t, 0.0


def _weave(t: float) -> tuple[float, float]:
    knots = [
        (0, 0),
        (5, 0),
        (6, 3),
        (8, 3),
        (9, 0),
        (13, 0),
        (14, 3),
        (15, 3),
        (16, 0),
        (30, 0),
    ]
    return t, float(np.interp(t, [k[0] for k in knots], [k[1] for k in knots]))


def _stopping(start: float, end: float):
    def position(t: float) -> tuple[float, float]:
        return min(t, start) + max(0.0, t - end), 0.0

    return position


def _waiting(until: float):
    def position(t: float) -> tuple[float, float]:
        return max(0.0, t - until), 0.0

    return position


def test_curb_is_not_walkable():
    def curb_walk(t: float) -> tuple[float, float]:
        return t, -2.5

    assert (
        _calc().calculate(_episode(curb_walk, 22.0, goal=[21.05, -2.5]), {})[
            "sidewalk_off_sidewalk"
        ]
        is True
    )


def test_robot_that_never_moves_has_no_failure_rate():
    results = _calc().calculate(_episode(lambda t: (0.0, 0.0), 20.0), {})
    assert results["sidewalk_route_completion"] is False
    assert results["sidewalk_freezes"] == 1
    assert results["sidewalk_failures_per_100m"] is None
    assert results["sidewalk_mean_speed"] == 0.0


def test_missing_pose_yields_nulls():
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=GOAL,
        map=WORLD,
        topics={},
    )
    assert _calc().calculate(episode, {}) == NULLS


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


def test_registry_runs_the_calculator_beside_the_native_metrics():
    topics = {"robot_tracks", "robot_odom", "robot_collision_events"}
    registry = MetricRegistry(make_robot_params(model="jackal"), world=WORLD)
    episode = _episode(_lane, 22.0)
    results = registry.run_scope(
        MetricScope.ROBOT,
        episode,
        available_topics=topics,
        pedsim_available=False,
        scope_slice=_robot_slice(episode),
    ).results
    assert results["sidewalk_route_completion"] is True
    assert {"success", "collision_amount", "time_to_goal"} <= set(results)
    skipped_episode = _episode(_lane, 22.0)
    skipped = registry.run_scope(
        MetricScope.ROBOT,
        skipped_episode,
        available_topics={"robot_tracks", "robot_odom"},
        pedsim_available=False,
        scope_slice=_robot_slice(skipped_episode),
    ).results
    assert {key: skipped[key] for key in NULLS} == NULLS


