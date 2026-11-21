import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import (
    NULL_REASON_SUFFIX,
    MetricScope,
    ScopeSlice,
)
from arena_evaluation.processing.metrics.performance.collision_metrics import (
    CollisionMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC_NS = 1_000_000_000
PARAMS = make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
_EVENTS_SCHEMA = {
    "t_ns": pl.Int64,
    "event_count": pl.Int64,
    "wall_count": pl.Int64,
    "static_count": pl.Int64,
    "ped_count": pl.Int64,
}


def _events(event_count, *, wall_count=None, static_count=None, ped_count=None) -> pl.DataFrame:
    count = len(event_count)
    return pl.DataFrame(
        {
            "t_ns": [index * SEC_NS for index in range(count)],
            "event_count": event_count,
            "wall_count": wall_count if wall_count is not None else [0] * count,
            "static_count": static_count if static_count is not None else [0] * count,
            "ped_count": ped_count if ped_count is not None else [0] * count,
        },
        schema=_EVENTS_SCHEMA,
    )


def _episode(
    events: pl.DataFrame | None,
    *,
    outcome_state: int | None = None,
    outcome_info: str | None = None,
    contacts: pl.DataFrame | None = None,
) -> AlignedEpisodeBundle:
    topics: dict[str, pl.DataFrame] = {}
    if events is not None:
        topics["robot_collision_events"] = events
    if contacts is not None:
        topics["robot_collision_contacts"] = contacts
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0]}),
        start_pos=[],
        goal_pos=[],
        outcome_state=outcome_state,
        outcome_info=outcome_info,
        topics=topics,
    )


def test_a_kind_the_stream_never_recorded_is_null():
    events = _events(
        [0, 2, 0],
        wall_count=[None, None, None],
        static_count=[None, None, None],
        ped_count=[None, None, None],
    )
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(events), {})
    assert results["collision_amount"] == 1
    assert results["collision_amount_wall"] is None
    assert results["collision_amount_static"] is None
    assert results["collision_amount_pedestrian"] is None


def test_a_missing_collision_count_leaves_success_null_without_samples():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(None, outcome_state=2), {})
    assert results["collision_amount"] is None
    assert results["success"] is None
    assert results[f"success{NULL_REASON_SUFFIX}"] == "no_samples"


def _contact_slice(
    full: AlignedEpisodeBundle,
    t_start_ns: int,
    t_end_ns: int,
    obstacle_id: str = "ped_1",
) -> ScopeSlice:
    window = AlignedEpisodeBundle(
        episode_id=full.episode_id,
        data=pl.DataFrame(),
        start_pos=full.start_pos,
        goal_pos=full.goal_pos,
        topics=dict(full.topics),
    )
    return ScopeSlice(
        scope=MetricScope.CONTACT,
        slice_id="con-001",
        parent_id=None,
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
        identity={"obstacle_id": obstacle_id},
        window_bundle=window,
        full_episode=full,
    )


def test_contact_carries_the_subject_actor():
    full = _episode(_events([0, 1, 1, 1, 0]))
    contact = _contact_slice(full, SEC_NS, 3 * SEC_NS, obstacle_id="env_0/Pedestrian_4")
    result = CollisionMetricsCalculator(PARAMS).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_obstacles"] == ["env_0/Pedestrian_4"]
    assert result["collisions"] == []


def test_other_scopes_are_rejected():
    full = _episode(_events([0, 1]))
    contact = _contact_slice(full, 0, SEC_NS)
    pedestrian_slice = ScopeSlice(
        scope=MetricScope.PEDESTRIAN,
        slice_id="ped_1",
        parent_id=None,
        t_start_ns=0,
        t_end_ns=SEC_NS,
        identity={},
        window_bundle=contact.window_bundle,
        full_episode=full,
    )
    with pytest.raises(ValueError, match="does not serve the pedestrians scope"):
        CollisionMetricsCalculator(PARAMS).calculate_for_scope(pedestrian_slice.window_bundle, {}, pedestrian_slice)
