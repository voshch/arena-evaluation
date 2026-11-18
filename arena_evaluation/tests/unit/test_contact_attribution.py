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
    ROBOT_COLLISION_CONTACTS_TOPIC,
    ContactContext,
    ContactEvent,
    closing_shares,
    compute_closing_speeds,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    Trajectory2D,
    extract_episode_ped_tracks,
    resolve_robot_trajectory,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

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


def _robot_tracks(
    t_s: np.ndarray, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> pl.DataFrame:
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


def _pedestrian_tracks(
    ped_id: str, t_s: np.ndarray, x: np.ndarray, vx: np.ndarray
) -> pl.DataFrame:
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


def _hit(
    t_ns: int, obstacle_id: str, x: float, kind: str = "pedestrian"
) -> dict[str, object]:
    """One recorded collision row at ``x`` metres on the x axis."""
    return {
        "t_ns": t_ns,
        "obstacle_id": obstacle_id,
        "kind": kind,
        "obstacle_x": x,
        "obstacle_y": 0.0,
    }


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
    topics: dict[str, pl.DataFrame] = {
        "robot_tracks": _robot_tracks(t_s, robot_x, zeros, robot_vx, zeros)
    }
    if ped is not None:
        ped_id, ped_x, ped_vx = ped
        topics["pedestrian_arena_peds"] = _pedestrian_tracks(ped_id, t_s, ped_x, ped_vx)
    if contacts is not None:
        topics[ROBOT_COLLISION_CONTACTS_TOPIC] = pl.DataFrame(
            contacts, schema=_CONTACT_SCHEMA
        )
    return _episode(topics)


def _episode(topics: Mapping[str, pl.DataFrame] | None) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[3.0, 0.0, 0.0],
        topics=None if topics is None else dict(topics),
    )


def _calculator(
    config: Mapping[str, object] | None = None,
) -> ContactAttributionCalculator:
    calc = ContactAttributionCalculator(make_robot_params(robot_radius=ROBOT_RADIUS_M))
    calc.metrics_config = dict(config or {})
    return calc


def _still_trajectory(duration_s: float) -> Trajectory2D:
    t_s = _grid(duration_s)
    zeros = np.zeros_like(t_s)
    return Trajectory2D(
        t_ns=_times(t_s), x=zeros, y=zeros, vx=zeros, vy=zeros, radius_m=ROBOT_RADIUS_M
    )


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




def test_golden_2_robot_drives_into_a_standing_pedestrian() -> None:
    """A robot driving into a stationary pedestrian: share 1 and a positive frozen-robot clearance."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(
        t_s,
        1.2 * t_s,
        np.full_like(t_s, 1.2),
        ped=("Pedestrian_5", np.full_like(t_s, 2.0), zeros),
        contacts=[_hit(SECOND_NS, "env_0/Pedestrian_5", 2.0)],
    )

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_robot_mps == pytest.approx(1.2, abs=1e-6)
    assert record.v_close_target_mps == pytest.approx(0.0, abs=1e-9)
    assert record.robot_closing_share == pytest.approx(1.0, abs=1e-9)
    assert record.target_closing_share == pytest.approx(0.0, abs=1e-9)
    assert record.freeze_clearance_m == pytest.approx(0.8, abs=1e-6)
    assert record.freeze_clearance_m > 0.0


def test_golden_4_a_wall_has_a_full_robot_share_and_no_freeze_clearance() -> None:
    """A wall: share 1 when the robot closes in, freeze clearance NULL not_applicable."""
    t_s = _grid()
    episode = _scene(t_s, t_s, np.ones_like(t_s))
    calc = _calculator(
        {
            "contact_t0_ns": SECOND_NS,
            "obstacle_kind": "wall",
            "static_point": [2.0, 0.0],
        }
    )

    out = calc.calculate_for_scope(
        episode,
        {},
        _contact_slice(
            episode,
            collider_kind="wall",
            obstacle_id="wall_north",
            t0_ns=SECOND_NS,
            static_point=(2.0, 0.0),
        ),
    )

    assert out["v_close_robot_mps"] == pytest.approx(1.0, abs=1e-6)
    assert out["v_close_target_mps"] == pytest.approx(0.0, abs=1e-9)
    assert out["freeze_clearance_m"] is None
    assert out["freeze_clearance_m_null_reason"] == "not_applicable"
    assert closing_shares(out["v_close_robot_mps"], out["v_close_target_mps"]) == (
        pytest.approx(1.0),
        pytest.approx(0.0),
    )


def test_golden_5_two_bodies_at_rest_have_zero_closing_speeds_and_no_share() -> None:
    """Two bodies at rest touching: closing speeds 0 and shares NULL zero_denominator."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(
        t_s,
        zeros,
        zeros,
        ped=("Pedestrian_2", np.full_like(t_s, 0.6), zeros),
        contacts=[_hit(SECOND_NS, "env_0/Pedestrian_2", 0.6)],
    )

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_robot_mps == pytest.approx(0.0, abs=1e-9)
    assert record.v_close_target_mps == pytest.approx(0.0, abs=1e-9)
    assert record.robot_closing_share is None
    assert record.target_closing_share is None


def test_golden_6_empty_recorded_collision_stream_is_zero_contacts_and_absent_is_null() -> (
    None
):
    """An empty recorded collision stream reports 0 contacts; no stream at all reports NULL."""
    t_s = _grid(2.0)
    zeros = np.zeros_like(t_s)
    ped = ("Pedestrian_9", np.full_like(t_s, 5.0), zeros)

    recorded_empty = _calculator().calculate(
        _scene(t_s, zeros, zeros, ped=ped, contacts=[]), {}
    )
    absent = _calculator().calculate(_scene(t_s, zeros, zeros, ped=ped), {})

    assert recorded_empty["contact_count"] == 0
    assert "contact_count_null_reason" not in recorded_empty
    assert absent["contact_count"] is None
    assert (
        absent["contact_count_null_reason"] == "topic_missing:robot_collision_contacts"
    )




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
def test_closing_shares_clip_the_speeds_and_leave_an_empty_ratio_null(
    v_robot: float | None,
    v_target: float | None,
    expected: tuple[float | None, float | None],
) -> None:
    """The two shares are a view of the stored speeds, clipped at 0 and NULL without a denominator."""
    robot_share, target_share = closing_shares(v_robot, v_target)

    if expected[0] is None:
        assert robot_share is None
        assert target_share is None
    else:
        assert robot_share == pytest.approx(expected[0])
        assert target_share == pytest.approx(expected[1])
        assert robot_share + target_share == pytest.approx(1.0)




def test_a_pedestrian_without_a_recorded_track_has_no_target_scalars() -> None:
    """A tracked body without a track has no measured speed: no zero stands in for it."""
    t_s = _grid()
    episode = _scene(
        t_s, t_s, np.ones_like(t_s), contacts=[_hit(SECOND_NS, "Missing_7", 1.5)]
    )

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_close_target_mps is None
    assert record.v_target_impact_mps is None
    assert record.relative_impact_speed_mps is None
    assert record.freeze_clearance_m is None
    assert record.robot_closing_share is None


def test_a_coincident_pedestrian_leaves_the_closing_speeds_undefined() -> None:
    """A degenerate line of sight is no measurement, not a zero closing speed."""
    t_s = _grid(1.0)
    zeros = np.zeros_like(t_s)
    episode = _scene(t_s, zeros, zeros, ped=("Pedestrian_1", zeros, zeros))
    robot_trajectory, _yaw = resolve_robot_trajectory(_calculator(), episode)
    track = extract_episode_ped_tracks(episode)["Pedestrian_1"]

    v_robot, v_target = compute_closing_speeds(
        ContactEvent(0, 0, "pedestrian", "Pedestrian_1", track=track),
        ContactContext(robot=robot_trajectory),
    )

    assert np.isnan(v_robot)
    assert np.isnan(v_target)


def test_episode_contact_counts_are_published_per_kind() -> None:
    """D-01: counts by collider kind are views of the same detector, never a stored judgement."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    contacts = [
        _hit(SECOND_NS, "Pedestrian_1", 2.0),
        _hit(SECOND_NS, "env_0_robot_2", 2.0, kind="robot"),
        _hit(SECOND_NS, "wall", 2.0, kind="wall"),
        _hit(2 * SECOND_NS, "env_0/box", 2.0, kind="static"),
    ]
    episode = _scene(
        t_s,
        t_s,
        np.ones_like(t_s),
        ped=("Pedestrian_1", np.full_like(t_s, 2.0), zeros),
        contacts=contacts,
    )

    out = _calculator().calculate(episode, {})

    assert out["contact_count"] == 4
    assert out["contact_count_pedestrian"] == 1
    assert out["contact_count_robot"] == 1
    assert out["contact_count_wall"] == 1
    assert out["contact_count_static"] == 1


def test_a_pedestrian_contact_without_a_recorded_radius_has_no_frozen_clearance() -> (
    None
):
    """A missing pedestrian radius is NULL with not_recorded, never replaced by an invented 0.3 m."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    peds = _pedestrian_tracks(
        "Pedestrian_5", t_s, np.full_like(t_s, 2.0), zeros
    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("radius_m"))
    episode = _episode(
        {
            "robot_tracks": _robot_tracks(t_s, t_s, zeros, np.ones_like(t_s), zeros),
            "pedestrian_arena_peds": peds,
            ROBOT_COLLISION_CONTACTS_TOPIC: pl.DataFrame(
                [_hit(SECOND_NS, "Pedestrian_5", 2.0)], schema=_CONTACT_SCHEMA
            ),
        }
    )

    out = _calculator().calculate_for_scope(
        episode,
        {},
        _contact_slice(
            episode,
            collider_kind="pedestrian",
            obstacle_id="Pedestrian_5",
            t0_ns=SECOND_NS,
        ),
    )

    assert out["freeze_clearance_m"] is None
    assert out["freeze_clearance_m_null_reason"] == "not_recorded"


def test_clearance_detection_refuses_a_pedestrian_without_a_recorded_radius() -> None:
    """Detecting a clearance contact needs the recorded radius; a silent 0.3 m default is not a measurement."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    peds = _pedestrian_tracks(
        "Pedestrian_6", t_s, np.full_like(t_s, 0.5), zeros
    ).with_columns(pl.lit(None, dtype=pl.Float64).alias("radius_m"))
    episode = _episode(
        {
            "robot_tracks": _robot_tracks(t_s, t_s, zeros, np.ones_like(t_s), zeros),
            "pedestrian_arena_peds": peds,
        }
    )

    with pytest.raises(MetricInputError):
        _calculator().calculate(episode, {})


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




def test_a_pedestrian_contact_without_an_obstacle_id_uses_the_nearest_track() -> None:
    """An unresolved pedestrian id resolves to the nearest recorded track."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(
        t_s,
        zeros,
        zeros,
        ped=("Pedestrian_1", np.full_like(t_s, 0.45), zeros),
        contacts=[_hit(SECOND_NS, "", 0.45)],
    )

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.target_ped_id == "Pedestrian_1"
    assert record.target_ped_name == "Pedestrian_1"
    assert record.obstacle_id == "Pedestrian_1"


def test_a_numeric_obstacle_suffix_resolves_the_pedestrian_track() -> None:
    """``env_0/7`` resolves to ``Pedestrian_7`` by its numeric suffix."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(
        t_s,
        zeros,
        zeros,
        ped=("Pedestrian_7", np.full_like(t_s, 1.0), zeros),
        contacts=[_hit(SECOND_NS, "env_0/7", 1.0)],
    )

    assert (
        _calculator().attribute_all_contacts(episode)[0].target_ped_id == "Pedestrian_7"
    )


def test_a_contact_after_the_pedestrian_track_has_no_target_speed() -> None:
    """A contact whose onset lies after the recorded pedestrian track has no target speed, not the track's last one."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _episode(
        {
            "robot_tracks": _robot_tracks(t_s, zeros, zeros, zeros, zeros),
            "pedestrian_arena_peds": _pedestrian_tracks(
                "Pedestrian_1", t_s[:10], np.full(10, 0.45), np.zeros(10)
            ),
            ROBOT_COLLISION_CONTACTS_TOPIC: pl.DataFrame(
                [_hit(2 * SECOND_NS, "Pedestrian_1", 0.45)], schema=_CONTACT_SCHEMA
            ),
        }
    )

    record = _calculator().attribute_all_contacts(episode)[0]

    assert record.v_target_impact_mps is None


def test_a_configured_contact_honours_the_target_agent_id_and_zero_windows() -> None:
    """An explicitly configured contact is attributed without detection."""
    t_s = _grid()
    zeros = np.zeros_like(t_s)
    episode = _scene(
        t_s, zeros, zeros, ped=("Pedestrian_10", 2.0 - t_s, np.full_like(t_s, -1.0))
    )
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


