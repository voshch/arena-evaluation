from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.hhi_contact import (
    ContactInteractionCalculator,
    bucket,
    pair_track,
    point_segment_distance,
    segment_crossings,
    smoothed_speed,
    stall_intervals,
    window_bounds,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def test_every_hhi_metric_is_episode_scoped() -> None:
    """The study has one interaction per episode, so the episode row carries every HHI output."""
    assert len(ContactInteractionCalculator.OUTPUTS) == 17
    assert ContactInteractionCalculator.effective_scopes() == frozenset(
        {MetricScope.EPISODE}
    )
    assert {
        scope
        for metric in ContactInteractionCalculator.OUTPUTS
        for scope in metric.scopes
    } == {MetricScope.EPISODE}
    assert ContactInteractionCalculator.REQUIRED_TOPICS == [
        "robot_tracks",
        "pedestrian_arena_peds",
        "pedestrian_interaction_events",
    ]


def test_stall_needs_the_minimum_duration_and_prior_motion() -> None:
    t = np.arange(0.0, 12.0, 0.1)
    speed = np.full_like(t, 0.5)
    speed[:10] = 0.0
    speed[30:35] = 0.0
    speed[40:50] = 0.0
    speed[60:95] = 0.0
    assert stall_intervals(t, speed) == [(pytest.approx(6.0), pytest.approx(9.4))]


def test_stops_at_the_goal_are_arrival() -> None:
    t = np.arange(0.0, 8.0, 0.1)
    speed = np.where(t < 3.0, 0.5, 0.0)
    far = np.full_like(t, 5.0)
    assert len(stall_intervals(t, speed, far)) == 1
    assert stall_intervals(t, speed, np.where(t < 3.0, 5.0, 0.2)) == []


def test_smoothed_speed_of_constant_motion() -> None:
    t = np.arange(0.0, 2.0, 0.02)
    v = smoothed_speed(t, 0.8 * t, np.zeros_like(t))
    assert np.median(v) == pytest.approx(0.8)


def test_windows_split_at_the_edges_and_clip() -> None:
    w = window_bounds(30.0, 8.0, 14.0)
    assert w == {
        "approach": (0.0, 8.0),
        "hold": (8.0, 14.0),
        "post": (14.0, 19.0),
        "rest": (19.0, 30.0),
    }
    assert window_bounds(16.0, 8.0, 14.0)["post"] == (14.0, 16.0)
    assert window_bounds(16.0, 8.0, 14.0)["rest"] == (16.0, 16.0)
    no_release = window_bounds(20.0, 8.0, None)
    assert no_release["hold"] == (8.0, 20.0) and no_release["post"] == (20.0, 20.0)
    assert window_bounds(20.0, None, None)["approach"] == (0.0, 20.0)


def test_bucket_by_onset_window() -> None:
    w = window_bounds(30.0, 8.0, 14.0)
    assert bucket([1.0, 8.0, 13.9, 14.0, 18.9, 19.0, 30.0], w) == {
        "approach": 1,
        "hold": 2,
        "post": 2,
        "rest": 2,
    }


def test_segment_crossing_detects_passing_between() -> None:
    robot = np.array([[0.0, -1.0], [0.0, 1.0], [0.0, 3.0]])
    a = np.array([[-0.6, 0.0]] * 3)
    b = np.array([[0.6, 0.0]] * 3)
    assert segment_crossings(robot, a, b).tolist() == [True, False]
    around = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert segment_crossings(around, a[:2], b[:2]).tolist() == [False]


def test_point_segment_distance() -> None:
    d = point_segment_distance(
        np.array([[0.0, 1.0], [2.0, 0.0]]),
        np.array([[-1.0, 0.0]] * 2),
        np.array([[1.0, 0.0]] * 2),
    )
    assert d.tolist() == pytest.approx([1.0, 1.0])


def test_pair_track_follows_the_pedestrian_id_at_each_robot_time() -> None:
    held = pl.DataFrame(
        {
            "t_ns": [0, SEC, SEC],
            "ped_id": ["a", "b", "a"],
            "x": [0.0, 5.0, 1.0],
            "y": [0.0, 5.0, 1.0],
        }
    )

    track = pair_track(held, np.array([0, SEC, 2 * SEC]), "a")

    assert track[0].tolist() == [0.0, 0.0]
    assert track[1].tolist() == [1.0, 1.0]
    assert np.isnan(track[2]).all()


def _episode(release: bool = True) -> AlignedEpisodeBundle:
    """Robot drives +x at 1 m/s, stalls 4 s right after the pair releases, and passes between them during the hold."""
    t = np.arange(0.0, 22.0, 0.05)
    x = np.empty_like(t)
    for i, ti in enumerate(t):
        x[i] = (
            min(ti, 12.0) if ti < 12.0 else (12.0 if ti < 16.0 else 12.0 + (ti - 16.0))
        )
    y = np.zeros_like(t)
    data = pl.DataFrame(
        {
            "t_ns": (t * SEC).astype(np.int64),
            "x": x,
            "y": y,
            "yaw": np.zeros_like(t),
        }
    )
    peds = flat_peds(
        [
            (
                int(ti * SEC),
                [
                    {"ped_id": "Pedestrian_1", "x": 6.0, "y": -0.6},
                    {"ped_id": "Pedestrian_2", "x": 6.0, "y": 0.6},
                ],
            )
            for ti in t
        ]
    )
    steps = [("ACTIVATED", 0, 4.0), ("HOLD_ONSET", 1, 5.0)] + (
        [("RELEASED", 2, 11.0)] if release else []
    )
    events = pl.DataFrame(
        [
            {
                "t_ns": int(s * SEC),
                "env_id": "env_0",
                "interaction_id": 3,
                "interaction_type": 10,
                "event": code,
                "event_name": name,
                "ped_id": ped,
            }
            for name, code, s in steps
            for ped in ("Pedestrian_1", "Pedestrian_2")
        ]
    )
    return AlignedEpisodeBundle(
        episode_id=0,
        data=data,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[30.0, 0.0],
        topics={"pedestrian_arena_peds": peds, "pedestrian_interaction_events": events},
    )


def test_calculator_on_a_synthetic_episode() -> None:
    episode = _episode()
    contact = ContactInteractionCalculator(make_robot_params())
    out = contact.calculate(episode, {})
    assert out["hhi_interaction"] == "hug"
    assert (out["hhi_hold_onset_s"], out["hhi_release_s"]) == (
        pytest.approx(5.0),
        pytest.approx(11.0),
    )
    assert out["hhi_stall_count"] == 1
    assert contact.series_table(episode).get_column("onset_s")[0] == pytest.approx(
        12.0, abs=0.2
    )
    assert (
        out["hhi_stalls_approach"],
        out["hhi_stalls_hold"],
        out["hhi_stalls_post"],
        out["hhi_stalls_rest"],
    ) == (0, 0, 1, 0)
    shares = [out[f"hhi_share_{w}"] for w in ("approach", "hold", "post", "rest")]
    assert sum(shares) == pytest.approx(1.0)
    assert out["hhi_share_post"] == pytest.approx(5.0 / out["hhi_episode_s"])
    assert out["hhi_passes_between"] == 1
    assert out["hhi_pair_gap_hold"] == pytest.approx(1.2)
    assert out["hhi_min_pair_clearance"] == pytest.approx(0.0, abs=0.06)
    assert set(out) == set(ContactInteractionCalculator.output_keys())


def test_calculator_without_a_release_holds_to_the_end() -> None:
    out = ContactInteractionCalculator(make_robot_params()).calculate(
        _episode(release=False), {}
    )
    assert out["hhi_release_s"] is None
    assert out["hhi_stalls_hold"] == 1 and out["hhi_stalls_post"] == 0


def test_calculator_without_contact_events_reports_unknown_windows() -> None:
    """Without the interaction stream there is no hold/release window: unknown, never zero."""
    ep = _episode()
    ep.topics = {}
    calc = ContactInteractionCalculator(make_robot_params())
    out = calc.calculate(ep, {})
    assert out["hhi_episode_s"] == pytest.approx(22.0, abs=0.1)
    for key in (
        "hhi_interaction",
        "hhi_hold_onset_s",
        "hhi_release_s",
        "hhi_stall_count",
        "hhi_stall_time_s",
        "hhi_passes_between",
        "hhi_min_pair_clearance",
        "hhi_pair_gap_hold",
    ):
        assert out[key] is None, key
    for window in ("approach", "hold", "post", "rest"):
        assert out[f"hhi_stalls_{window}"] is None
        assert out[f"hhi_share_{window}"] is None
    provenance = calc.missing_input_provenance(ep)
    assert (
        provenance["hhi_stall_count"]
        == "hhi_stall_count:topic_missing:pedestrian_interaction_events"
    )


def test_speed_of_a_single_sample_is_zero() -> None:
    assert smoothed_speed(
        np.array([0.0]), np.array([1.0]), np.array([2.0])
    ).tolist() == [0.0]


def test_no_stalls_without_samples_or_prior_motion() -> None:
    assert stall_intervals(np.zeros(0), np.zeros(0)) == []
    assert stall_intervals(np.arange(5.0), np.zeros(5)) == []


def test_crossings_need_a_robot_move() -> None:
    assert (
        segment_crossings(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))).tolist()
        == []
    )


def test_calculator_without_an_activation_edge_sees_no_interaction() -> None:
    ep = _episode()
    ep.topics = {
        **ep.topics,
        "pedestrian_interaction_events": ep.topics[
            "pedestrian_interaction_events"
        ].filter(pl.col("event_name") == "RELEASED"),
    }
    out = ContactInteractionCalculator(make_robot_params()).calculate(ep, {})
    assert (
        out["hhi_interaction"] is None
        and out["hhi_hold_onset_s"] is None
        and out["hhi_release_s"] is None
    )
    assert out["hhi_stall_count"] == 1


def test_calculator_on_too_short_or_unlocatable_data_is_all_none() -> None:
    empty = dict.fromkeys(ContactInteractionCalculator.output_keys())
    short = AlignedEpisodeBundle(
        episode_id=0,
        data=pl.DataFrame({"t_ns": [0]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0],
    )
    assert (
        ContactInteractionCalculator(make_robot_params()).calculate(short, {}) == empty
    )
    nan = float("nan")
    lost = AlignedEpisodeBundle(
        episode_id=0,
        data=pl.DataFrame(
            {"t_ns": [0, SEC], "x": [nan, nan], "y": [nan, nan], "yaw": [0.0, 0.0]}
        ),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0],
    )
    assert (
        ContactInteractionCalculator(make_robot_params()).calculate(lost, {}) == empty
    )
