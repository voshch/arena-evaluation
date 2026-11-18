"""Unit contracts for contact attribution as measured scalars.

The six golden scenarios of the PR2 brief are asserted on scalars: closing speeds, the two
closing shares (views of the stored speeds), impact speeds and the frozen-robot clearance.
No test asserts a label, a fault verdict or a threshold cut.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.safety import ContactAttributionCalculator
from arena_evaluation.processing.metrics.safety.contact_attribution import (
    COLLISION_CONTACTS_TOPIC,
    CONTACT_MERGE_GAP_S,
    ContactAttributionRecord,
    ContactContext,
    ContactEvent,
    attribute_single_contact,
    closing_shares,
    compute_closing_speeds,
    frozen_robot_clearance,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    Trajectory2D,
    extract_episode_ped_tracks,
    resolve_robot_trajectory,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SECOND_NS = 1_000_000_000
SAMPLE_S = 0.05
ROBOT_RADIUS_M = 0.3
PED_RADIUS_M = 0.3

_ROBOT_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "env_id": pl.String,
        "robot_id": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
    }
)
_PED_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "ped_id": pl.String,
        "ped_name": pl.String,
        "x": pl.Float64,
        "y": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "radius_m": pl.Float64,
        "desired_speed": pl.Float64,
    }
)
_CONTACT_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "obstacle_id": pl.String,
        "kind": pl.String,
        "obstacle_x": pl.Float64,
        "obstacle_y": pl.Float64,
    }
)


def _grid(duration_s: float = 3.0) -> np.ndarray:
    return np.arange(0.0, duration_s, SAMPLE_S)


def _times(t_s: np.ndarray) -> np.ndarray:
    """Nanosecond stamps of an array of seconds."""
    return (np.asarray(t_s, dtype=np.float64) * SECOND_NS).astype(np.int64)


def _robot_tracks(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> pl.DataFrame:
    n = len(t_s)
    return pl.DataFrame(
        {
            "t_ns": _times(t_s),
            "env_id": ["env_0"] * n,
            "robot_id": ["env_0_jackal"] * n,
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "yaw": np.zeros(n, dtype=np.float64),
            "vx": np.asarray(vx, dtype=np.float64),
            "vy": np.asarray(vy, dtype=np.float64),
        },
        schema=_ROBOT_SCHEMA,
    )


def _pedestrian_tracks(ped_id: str, t_s: np.ndarray, x: np.ndarray, vx: np.ndarray) -> pl.DataFrame:
    n = len(t_s)
    return pl.DataFrame(
        {
            "t_ns": _times(t_s),
            "ped_id": [ped_id] * n,
            "ped_name": [ped_id] * n,
            "x": np.asarray(x, dtype=np.float64),
            "y": np.zeros(n, dtype=np.float64),
            "vx": np.asarray(vx, dtype=np.float64),
            "vy": np.zeros(n, dtype=np.float64),
            "radius_m": np.full(n, PED_RADIUS_M, dtype=np.float64),
            "desired_speed": np.full(n, np.nan, dtype=np.float64),
        },
        schema=_PED_SCHEMA,
    )


def _hit(t_ns: int, obstacle_id: str, x: float, kind: str = "pedestrian") -> dict[str, object]:
    """One recorded collision row at ``x`` metres on the x axis."""
    return {"t_ns": t_ns, "obstacle_id": obstacle_id, "kind": kind, "obstacle_x": x, "obstacle_y": 0.0}


def _scene(
    t_s: np.ndarray,
    robot_x: np.ndarray,
    robot_vx: np.ndarray,
    *,
    ped: tuple[str, np.ndarray, np.ndarray] | None = None,
    contacts: list[dict[str, object]] | None = None,
) -> AlignedEpisodeBundle:
    """Robot on the x axis, an optional pedestrian ``(id, x, vx)`` and optional collision rows."""
    zeros = np.zeros_like(t_s)
    topics: dict[str, pl.DataFrame] = {"robot_tracks": _robot_tracks(t_s, robot_x, zeros, robot_vx, zeros)}
    if ped is not None:
        ped_id, ped_x, ped_vx = ped
        topics["pedestrian_tracks"] = _pedestrian_tracks(ped_id, t_s, ped_x, ped_vx)
    if contacts is not None:
        topics[COLLISION_CONTACTS_TOPIC] = pl.DataFrame(contacts, schema=_CONTACT_SCHEMA)
    return _episode(topics)


def _episode(topics: Mapping[str, pl.DataFrame] | None) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[3.0, 0.0, 0.0],
        topics=None if topics is None else dict(topics),
    )


def _calculator(config: Mapping[str, object] | None = None) -> ContactAttributionCalculator:
    calc = ContactAttributionCalculator(RobotParams(robot_radius=ROBOT_RADIUS_M))
    calc.metrics_config = dict(config or {})
    return calc


def _still_trajectory(duration_s: float) -> Trajectory2D:
    t_s = _grid(duration_s)
    zeros = np.zeros_like(t_s)
    return Trajectory2D(t_ns=_times(t_s), x=zeros, y=zeros, vx=zeros, vy=zeros, radius_m=ROBOT_RADIUS_M)


def _contact_slice(
    episode: AlignedEpisodeBundle,
    *,
    collider_kind: str,
    obstacle_id: str,
    t0_ns: int,
    static_point: tuple[float, float] | None = None,
) -> ScopeSlice:
    return ScopeSlice(
        scope=MetricScope.CONTACT,
        slice_id="contact-0",
        parent_id=None,
        t_start_ns=t0_ns,
        t_end_ns=t0_ns,
        identity={
            "collider_kind": collider_kind,
            "obstacle_id": obstacle_id,
            "static_point": static_point,
            "pre_contact_window_s": 0.5,
            "post_contact_window_s": 0.5,
        },
        window_bundle=episode,
        full_episode=episode,
    )


# --- the six golden scenarios -------------------------------------------------


def test_golden_1_stopped_robot_is_rammed_by_a_walking_pedestrian() -> None:
    """A stopped robot and a pedestrian walking into it: robot share 0, positive target closing, clearance at most 0."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_10", 2.0 - t_s, np.full_like(t_s, -1.0)), contacts=[_hit(int(1.5 * SECOND_NS), "env_0/Pedestrian_10", 0.5)])

    records = _calculator().attribute_all_contacts(episode)

    assert len(records) == 1
    record = records[0]
    assert record.v_close_robot_mps == pytest.approx(0.0, abs=1e-9)
    assert record.v_close_target_mps == pytest.approx(1.0, abs=1e-6)
    assert record.robot_closing_share == pytest.approx(0.0, abs=1e-9)
    assert record.freeze_clearance_m is not None
    assert record.freeze_clearance_m <= 0.0


def test_golden_2_robot_drives_into_a_standing_pedestrian() -> None:
    """A robot driving into a stationary pedestrian: share 1 and a positive frozen-robot clearance."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, 1.2 * t_s, np.full_like(t_s, 1.2), ped=("Pedestrian_5", np.full_like(t_s, 2.0), zeros), contacts=[_hit(SECOND_NS, "env_0/Pedestrian_5", 2.0)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_robot_mps == pytest.approx(1.2, abs=1e-6)
    assert record.v_close_target_mps == pytest.approx(0.0, abs=1e-9)
    assert record.robot_closing_share == pytest.approx(1.0, abs=1e-9)
    assert record.target_closing_share == pytest.approx(0.0, abs=1e-9)
    assert record.freeze_clearance_m == pytest.approx(0.8, abs=1e-6)
    assert record.freeze_clearance_m > 0.0


def test_golden_3_equal_closing_speeds_give_an_even_split() -> None:
    """Both bodies closing at equal speed: share 0.5 and the hand-computed frozen-robot clearance."""
    t_s = _grid()
    episode = _scene(t_s, 0.5 * t_s, np.full_like(t_s, 0.5), ped=("Pedestrian_3", 2.0 - 0.5 * t_s, np.full_like(t_s, -0.5)), contacts=[_hit(SECOND_NS, "env_0/Pedestrian_3", 1.5)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_robot_mps == pytest.approx(0.5, abs=1e-6)
    assert record.v_close_target_mps == pytest.approx(0.5, abs=1e-6)
    assert record.robot_closing_share == pytest.approx(0.5, abs=1e-9)
    assert record.target_closing_share == pytest.approx(0.5, abs=1e-9)
    # Frozen at x = 0.25 m at t = 0.5 s, the pedestrian reaches x = 1.25 m at t = 1.5 s:
    # closest centre distance 1.0 m minus the two 0.3 m radii.
    assert record.freeze_clearance_m == pytest.approx(0.4, abs=1e-6)


def test_golden_4_a_wall_has_a_full_robot_share_and_no_freeze_clearance() -> None:
    """A wall: share 1 when the robot closes in, freeze clearance NULL not_applicable."""
    t_s = _grid()
    episode = _scene(t_s, t_s, np.ones_like(t_s))
    calc = _calculator({"contact_t0_ns": SECOND_NS, "obstacle_kind": "wall", "static_point": [2.0, 0.0]})

    out = calc.calculate_for_scope(episode, {}, _contact_slice(episode, collider_kind="wall", obstacle_id="wall_north", t0_ns=SECOND_NS, static_point=(2.0, 0.0)))

    assert out["v_close_robot_mps"] == pytest.approx(1.0, abs=1e-6)
    assert out["v_close_target_mps"] == pytest.approx(0.0, abs=1e-9)
    assert out["freeze_clearance_m"] is None
    assert out["freeze_clearance_m_null_reason"] == "not_applicable"
    assert closing_shares(out["v_close_robot_mps"], out["v_close_target_mps"]) == (pytest.approx(1.0), pytest.approx(0.0))


def test_golden_5_two_bodies_at_rest_have_zero_closing_speeds_and_no_share() -> None:
    """Two bodies at rest touching: closing speeds 0 and shares NULL zero_denominator."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_2", np.full_like(t_s, 0.6), zeros), contacts=[_hit(SECOND_NS, "env_0/Pedestrian_2", 0.6)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_robot_mps == pytest.approx(0.0, abs=1e-9)
    assert record.v_close_target_mps == pytest.approx(0.0, abs=1e-9)
    assert record.robot_closing_share is None
    assert record.target_closing_share is None


def test_golden_6_empty_recorded_collision_stream_is_zero_contacts_and_absent_is_null() -> None:
    """An empty recorded collision stream reports 0 contacts; no stream at all reports NULL."""
    t_s = _grid(2.0)
    zeros = np.zeros_like(t_s)
    ped = ("Pedestrian_9", np.full_like(t_s, 5.0), zeros)

    recorded_empty = _calculator().calculate(_scene(t_s, zeros, zeros, ped=ped, contacts=[]), {})
    absent = _calculator().calculate(_scene(t_s, zeros, zeros, ped=ped), {})

    assert recorded_empty["contact_count"] == 0
    assert "contact_count_null_reason" not in recorded_empty
    assert absent["contact_count"] is None
    assert absent["contact_count_null_reason"] == "topic_missing:collision_contacts"


# --- scalar helpers and output contract ---------------------------------------


@pytest.mark.parametrize(
    ("v_robot", "v_target", "expected"),
    [
        (1.0, 0.0, (1.0, 0.0)),
        (0.0, 1.0, (0.0, 1.0)),
        (0.5, 0.5, (0.5, 0.5)),
        (2.0, -1.0, (1.0, 0.0)),
        (-2.0, 2.0, (0.0, 1.0)),
        (0.0, 0.0, (None, None)),
        (None, 1.0, (None, None)),
        (1.0, None, (None, None)),
        (float("nan"), 1.0, (None, None)),
    ],
)
def test_closing_shares_clip_the_speeds_and_leave_an_empty_ratio_null(v_robot: float | None, v_target: float | None, expected: tuple[float | None, float | None]) -> None:
    """The two shares are a view of the stored speeds, clipped at 0 and NULL without a denominator."""
    robot_share, target_share = closing_shares(v_robot, v_target)

    if expected[0] is None:
        assert robot_share is None
        assert target_share is None
    else:
        assert robot_share == pytest.approx(expected[0])
        assert target_share == pytest.approx(expected[1])
        assert robot_share + target_share == pytest.approx(1.0)


def test_the_calculator_publishes_no_label_and_no_stored_share() -> None:
    """D-01: no label, no verdict and no stored copy of a share is an output."""
    keys = set(ContactAttributionCalculator.output_keys())

    assert keys == {
        "v_close_robot_mps",
        "v_close_target_mps",
        "v_robot_impact_mps",
        "v_target_impact_mps",
        "relative_impact_speed_mps",
        "freeze_clearance_m",
        "contact_count",
    }
    assert keys.isdisjoint({"closer_label", "fault_attributed_to", "is_robot_fault", "is_robot_fault_lower_bound", "counterfactual_regime", "robot_closing_share", "target_closing_share", "contact_records", "success_no_robot_fault"})
    assert frozenset(ContactAttributionCalculator.effective_scopes()) == frozenset({MetricScope.CONTACT, MetricScope.EPISODE})


def test_the_contact_scope_publishes_exactly_its_declared_scalars() -> None:
    """A contact slice returns the six CONTACT metrics, their null reasons and nothing else."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_10", 2.0 - t_s, np.full_like(t_s, -1.0)))

    out = _calculator().calculate_for_scope(episode, {}, _contact_slice(episode, collider_kind="pedestrian", obstacle_id="Pedestrian_10", t0_ns=int(1.5 * SECOND_NS)))

    declared = set(ContactAttributionCalculator.output_keys_for_scope(MetricScope.CONTACT))
    assert {key for key in out if not key.endswith("_null_reason")} == declared
    assert out["v_close_robot_mps"] == pytest.approx(0.0, abs=1e-9)
    assert out["freeze_clearance_m"] == pytest.approx(-0.6, abs=1e-6)


def test_impact_speeds_are_measured_at_the_contact_onset() -> None:
    """Impact speeds come from the pose at t0; the relative speed is their vector difference."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, t_s, np.ones_like(t_s), ped=("Pedestrian_4", np.full_like(t_s, 2.0), zeros), contacts=[_hit(SECOND_NS, "env_0/Pedestrian_4", 2.0)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_robot_impact_mps == pytest.approx(1.0, abs=1e-6)
    assert record.v_target_impact_mps == pytest.approx(0.0, abs=1e-9)
    assert record.relative_impact_speed_mps == pytest.approx(1.0, abs=1e-6)


# --- degenerate and missing inputs: no measurement is never a zero ------------


def test_a_pedestrian_without_a_recorded_track_has_no_target_scalars() -> None:
    """A tracked body without a track has no measured speed: no zero stands in for it."""
    t_s = _grid()
    episode = _scene(t_s, t_s, np.ones_like(t_s), contacts=[_hit(SECOND_NS, "Missing_7", 1.5)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_target_mps is None
    assert record.v_target_impact_mps is None
    assert record.relative_impact_speed_mps is None
    assert record.freeze_clearance_m is None
    assert record.robot_closing_share is None


def test_a_wall_without_a_position_has_no_closing_speed() -> None:
    """A bumper-event wall has no line of sight: the robot closing speed is NULL, the wall speed 0."""
    t_s = _grid()
    events = pl.DataFrame({"t_ns": _times(np.array([1.0])), "wall_count": [1], "static_count": [0]})
    episode = _episode({"robot_tracks": _robot_tracks(t_s, t_s, np.zeros_like(t_s), np.ones_like(t_s), np.zeros_like(t_s)), "collision_events": events})

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.obstacle_kind == "wall"
    assert record.v_close_robot_mps is None
    assert record.v_close_target_mps == pytest.approx(0.0, abs=1e-9)
    assert record.freeze_clearance_m is None


def test_a_coincident_pedestrian_leaves_the_closing_speeds_undefined() -> None:
    """A degenerate line of sight is no measurement, not a zero closing speed."""
    t_s = _grid(1.0)
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_1", zeros, zeros))
    robot_trajectory, _yaw = resolve_robot_trajectory(_calculator(), episode)
    track = extract_episode_ped_tracks(episode)["Pedestrian_1"]

    v_robot, v_target = compute_closing_speeds(ContactEvent(0, 0, "pedestrian", "Pedestrian_1", track=track), ContactContext(robot=robot_trajectory))

    assert np.isnan(v_robot)
    assert np.isnan(v_target)


def test_frozen_robot_clearance_is_not_applicable_to_another_robot() -> None:
    """Another robot reacts to a stopped robot, so it has no frozen-robot clearance."""
    context = ContactContext(robot=_still_trajectory(1.0))

    assert frozen_robot_clearance(ContactEvent(0, 0, "robot", "env_0_robot_2"), context) is None


def test_a_contact_slice_without_identity_raises() -> None:
    """An unknown collider kind is an input error, never a silent pedestrian default."""
    t_s = _grid(2.0)
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros)
    scope_slice = ScopeSlice(
        scope=MetricScope.CONTACT,
        slice_id="contact-0",
        parent_id=None,
        t_start_ns=0,
        t_end_ns=0,
        identity={"obstacle_id": "x"},
        window_bundle=episode,
        full_episode=episode,
    )

    with pytest.raises(MetricInputError):
        _calculator().calculate_for_scope(episode, {}, scope_slice)


# --- detection of contact intervals -------------------------------------------


@pytest.mark.parametrize(
    ("kind", "obstacle_id", "expected_kind"),
    [
        ("pedestrian", "env_0/Pedestrian_1", "pedestrian"),
        ("ped_human", "env_0/Pedestrian_1", "pedestrian"),
        ("human", "env_0/Pedestrian_1", "pedestrian"),
        ("wall", "wall_north", "wall"),
        ("static", "pillar_south", "static"),
    ],
)
def test_collision_contact_kind_is_normalized(kind: str, obstacle_id: str, expected_kind: str) -> None:
    """Every recorded contact kind is normalized before attribution."""
    t_s = _grid()
    episode = _scene(t_s, t_s, np.ones_like(t_s), ped=("Pedestrian_1", np.full_like(t_s, 2.0), np.zeros_like(t_s)), contacts=[_hit(SECOND_NS, obstacle_id, 2.0, kind)])

    records = _calculator().attribute_all_contacts(episode)

    assert len(records) == 1
    assert records[0].obstacle_kind == expected_kind


def test_a_pedestrian_contact_without_an_obstacle_id_uses_the_nearest_track() -> None:
    """An unresolved pedestrian id resolves to the nearest recorded track."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_1", np.full_like(t_s, 0.45), zeros), contacts=[_hit(SECOND_NS, "", 0.45)])

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.target_ped_id == "Pedestrian_1"
    assert record.target_ped_name == "Pedestrian_1"
    assert record.obstacle_id == "Pedestrian_1"


def test_a_numeric_obstacle_suffix_resolves_the_pedestrian_track() -> None:
    """``env_0/7`` resolves to ``Pedestrian_7`` by its numeric suffix."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_7", np.full_like(t_s, 1.0), zeros), contacts=[_hit(SECOND_NS, "env_0/7", 1.0)])

    assert _calculator().attribute_all_contacts(episode)[0].target_ped_id == "Pedestrian_7"


def test_bumper_events_read_wall_and_static_counts() -> None:
    """The collision_events wall and static counters become wall and static contacts."""
    t_s = _grid(4.0)
    zeros = np.zeros_like(t_s)
    events = pl.DataFrame({"t_ns": _times(np.array([1.0, 2.0, 3.0])), "wall_count": [1, 0, 0], "static_count": [0, 1, 0], "ped_count": [0, 0, 1]})
    episode = _episode({"robot_tracks": _robot_tracks(t_s, zeros, zeros, zeros, zeros), "collision_events": events})

    assert _calculator().calculate(episode, {})["contact_count"] == 2


@pytest.mark.parametrize(("gap_s", "expected_count", "expected_duration_s"), [(0.2, 1, 0.2), (0.3, 2, 0.0)])
def test_collision_contacts_merge_within_the_gap_and_split_beyond_it(gap_s: float, expected_count: int, expected_duration_s: float) -> None:
    """Contacts of one target merge within the declared gap and stay separate beyond it."""
    assert CONTACT_MERGE_GAP_S == pytest.approx(0.25)
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    second_ns = int(round((1.0 + gap_s) * SECOND_NS))
    episode = _scene(t_s, zeros, zeros, contacts=[_hit(SECOND_NS, "env_0/Pedestrian_1", 0.5), _hit(second_ns, "env_0/Pedestrian_1", 0.5)])

    records = _calculator().attribute_all_contacts(episode)

    assert len(records) == expected_count
    assert records[0].duration_s == pytest.approx(expected_duration_s, abs=1e-6)


def test_a_configured_contact_honours_the_target_agent_id_and_zero_windows() -> None:
    """An explicitly configured contact is attributed without detection."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_10", 2.0 - t_s, np.full_like(t_s, -1.0)))
    calc = _calculator(
        {
            "contact_t0_ns": int(round(1.4 * SECOND_NS)),
            "contact_t_end_ns": int(round(1.4 * SECOND_NS)),
            "obstacle_kind": "pedestrian",
            "target_agent_id": 10,
            "pre_contact_window_s": 0.0,
            "post_contact_window_s": 0.0,
        }
    )

    record = calc.attribute_all_contacts(episode)[0]

    assert record.target_ped_id == "Pedestrian_10"
    assert record.v_close_target_mps == pytest.approx(1.0, abs=1e-6)


def test_absent_or_empty_robot_tracks_leave_the_contact_count_null() -> None:
    """No robot trajectory is no measurement: the count is NULL, never zero."""
    for topics in (None, {"robot_tracks": pl.DataFrame(schema=_ROBOT_SCHEMA)}):
        out = _calculator().calculate(_episode(topics), {})

        assert out["contact_count"] is None
        assert out["contact_count_null_reason"] == "topic_missing:robot_tracks"


def test_single_contact_record_keeps_the_episode_relative_onset() -> None:
    """The record's onset is relative to the robot's first sample and its duration is non-negative."""
    record = attribute_single_contact(3, ContactEvent(int(1.5 * SECOND_NS), SECOND_NS, "wall", "wall_north", static_point=(2.0, 0.0)), ContactContext(robot=_still_trajectory(2.0)))

    assert isinstance(record, ContactAttributionRecord)
    assert record.contact_id == 3
    assert record.t0_s == pytest.approx(1.5)
    assert record.duration_s == 0.0
