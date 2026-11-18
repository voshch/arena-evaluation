import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import NULL_REASON_SUFFIX, MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.performance.collision_metrics import CollisionMetricsCalculator
from arena_evaluation.processing.outcome import outcome_verdict
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC_NS = 1_000_000_000
PARAMS = RobotParams(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
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


def _episode(events: pl.DataFrame | None, *, outcome_state: int | None = None, outcome_info: str | None = None, contacts: pl.DataFrame | None = None) -> AlignedEpisodeBundle:
    topics: dict[str, pl.DataFrame] = {}
    if events is not None:
        topics["collision_events"] = events
    if contacts is not None:
        topics["collision_contacts"] = contacts
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": [0]}),
        start_pos=[],
        goal_pos=[],
        outcome_state=outcome_state,
        outcome_info=outcome_info,
        topics=topics,
    )


def test_collision_amount_counts_rising_edges_of_the_native_stream():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 2, 2, 0, 1])), {})
    assert results["collision_amount"] == 2


def test_collision_amount_splits_by_kind():
    events = _events(
        [0, 2, 2, 0, 1, 1],
        wall_count=[0, 1, 1, 0, 0, 0],
        static_count=[0, 1, 1, 0, 0, 0],
        ped_count=[0, 0, 0, 0, 1, 1],
    )
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(events), {})
    assert results["collision_amount"] == 2
    assert results["collision_amount_wall"] == 1
    assert results["collision_amount_static"] == 1
    assert results["collision_amount_pedestrian"] == 1


def test_a_kind_the_stream_never_recorded_is_null():
    events = _events([0, 2, 0], wall_count=[None, None, None], static_count=[None, None, None], ped_count=[None, None, None])
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(events), {})
    assert results["collision_amount"] == 1
    assert results["collision_amount_wall"] is None
    assert results["collision_amount_static"] is None
    assert results["collision_amount_pedestrian"] is None


def test_without_the_native_stream_every_count_is_null():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(None), {})
    assert results["collision_amount"] is None
    assert results["collision_amount_wall"] is None


@pytest.mark.parametrize(
    ("outcome_state", "outcome_info", "expected_success"),
    [
        (2, None, True),
        (3, None, False),
        (4, None, False),
        (5, None, False),
        (2, "collision", False),
    ],
)
def test_success_is_the_runtime_verdict(outcome_state, outcome_info, expected_success):
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 0, 0]), outcome_state=outcome_state, outcome_info=outcome_info), {})
    assert results["success"] is expected_success


@pytest.mark.parametrize("outcome_state", [None, 0, 1])
def test_an_unresolved_record_is_not_a_success(outcome_state):
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 0, 0]), outcome_state=outcome_state), {})
    assert results["success"] is False
    assert results[f"success{NULL_REASON_SUFFIX}"] is None


def test_one_collision_fails_the_default_threshold():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 1, 0]), outcome_state=2), {})
    assert results["collision_amount"] == 1
    assert results["success"] is False


def test_two_collisions_pass_a_threshold_of_three():
    calculator = CollisionMetricsCalculator(PARAMS)
    calculator.metrics_config = {"max_collisions": 3}
    results = calculator.calculate(_episode(_events([0, 1, 0, 1]), outcome_state=2), {})
    assert results["collision_amount"] == 2
    assert results["success"] is True


def test_a_missing_collision_count_leaves_success_null_without_samples():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(None, outcome_state=2), {})
    assert results["collision_amount"] is None
    assert results["success"] is None
    assert results[f"success{NULL_REASON_SUFFIX}"] == "no_samples"


@pytest.mark.parametrize(
    ("outcome_state", "outcome_info", "expected"),
    [
        (2, None, ("GOAL_REACHED", True)),
        (3, None, ("FAILED", False)),
        (4, None, ("CANCELLED", False)),
        (5, None, ("FATAL", False)),
        (2, "collision", ("COLLISION", False)),
        (0, None, ("UNRESOLVED", False)),
        (None, None, ("UNRESOLVED", False)),
    ],
)
def test_the_runtime_result_token_and_success(outcome_state, outcome_info, expected):
    assert outcome_verdict(outcome_state, outcome_info) == expected


def test_collision_obstacles_recorded():
    contacts = pl.DataFrame(
        {
            "t_ns": [1_000_000_000, 1_000_000_000, 2_000_000_000],
            "obstacle_id": ["<wall>", "chair_1", "ped_3"],
        }
    )
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 2, 1]), contacts=contacts), {})
    assert results["collision_obstacles"] == ["<wall>", "chair_1", "ped_3"]


def test_collision_obstacles_absent():
    results = CollisionMetricsCalculator(PARAMS).calculate(_episode(_events([0, 1])), {})
    assert results["collision_obstacles"] == []


def test_robot_scope_publishes_its_own_scalars():
    assert set(CollisionMetricsCalculator.output_keys_for_scope(MetricScope.ROBOT)) == {
        "collision_amount",
        "collision_amount_wall",
        "collision_amount_static",
        "collision_amount_pedestrian",
        "success",
    }


# --- CONTACT scope -----------------------------------------------------------


def _contact_slice(full: AlignedEpisodeBundle, t_start_ns: int, t_end_ns: int, obstacle_id: str = "ped_1") -> ScopeSlice:
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


@pytest.mark.parametrize(
    ("t_start_ns", "t_end_ns", "expected_event", "expected_wall"),
    [
        (SEC_NS, 3 * SEC_NS, 1, 1),  # both onsets fall inside the contact interval
        (3 * SEC_NS, 3 * SEC_NS, 0, 0),  # a later contact window contains no onset of its own
        (0, 0, 0, 0),  # the interval before the first onset
    ],
)
def test_contact_counts_only_the_windows_own_onsets(t_start_ns, t_end_ns, expected_event, expected_wall):
    full = _episode(_events([0, 1, 1, 1, 0], wall_count=[0, 0, 1, 1, 0]))
    contact = _contact_slice(full, t_start_ns, t_end_ns)
    result = CollisionMetricsCalculator(PARAMS).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_amount"] == expected_event
    assert result["collision_amount_wall"] == expected_wall


def test_contact_counts_are_null_without_the_collision_stream():
    contact = _contact_slice(_episode(None), SEC_NS, 2 * SEC_NS)
    result = CollisionMetricsCalculator(PARAMS).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_amount"] is None
    assert result["collision_amount_wall"] is None


def test_contact_carries_the_subject_actor():
    full = _episode(_events([0, 1, 1, 1, 0]))
    contact = _contact_slice(full, SEC_NS, 3 * SEC_NS, obstacle_id="env_0/Pedestrian_4")
    result = CollisionMetricsCalculator(PARAMS).calculate_for_scope(contact.window_bundle, {}, contact)
    assert result["collision_obstacles"] == ["env_0/Pedestrian_4"]
    assert result["collisions"] == []


def test_contact_scope_publishes_the_scalar_metrics():
    assert CollisionMetricsCalculator.output_keys_for_scope(MetricScope.CONTACT) == ("collision_amount", "collision_amount_wall")


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
