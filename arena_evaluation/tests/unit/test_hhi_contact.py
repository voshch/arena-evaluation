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
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def test_stall_needs_the_minimum_duration_and_prior_motion() -> None:
    t = np.arange(0.0, 12.0, 0.1)
    speed = np.full_like(t, 0.5)
    speed[:10] = 0.0  # start-up latency: not a stall
    speed[30:35] = 0.0  # 0.5 s: too short
    speed[40:50] = 0.0  # 1 s: under the 3 s floor
    speed[60:95] = 0.0  # 3.5 s stall
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
    assert w == {"approach": (0.0, 8.0), "hold": (8.0, 14.0), "post": (14.0, 19.0), "rest": (19.0, 30.0)}
    assert window_bounds(16.0, 8.0, 14.0)["post"] == (14.0, 16.0)
    assert window_bounds(16.0, 8.0, 14.0)["rest"] == (16.0, 16.0)
    no_release = window_bounds(20.0, 8.0, None)
    assert no_release["hold"] == (8.0, 20.0) and no_release["post"] == (20.0, 20.0)
    assert window_bounds(20.0, None, None)["approach"] == (0.0, 20.0)


def test_bucket_by_onset_window() -> None:
    w = window_bounds(30.0, 8.0, 14.0)
    assert bucket([1.0, 8.0, 13.9, 14.0, 18.9, 19.0, 30.0], w) == {"approach": 1, "hold": 2, "post": 2, "rest": 2}


def test_segment_crossing_detects_passing_between() -> None:
    robot = np.array([[0.0, -1.0], [0.0, 1.0], [0.0, 3.0]])
    a = np.array([[-0.6, 0.0]] * 3)
    b = np.array([[0.6, 0.0]] * 3)
    assert segment_crossings(robot, a, b).tolist() == [True, False]
    around = np.array([[1.0, -1.0], [1.0, 1.0]])
    assert segment_crossings(around, a[:2], b[:2]).tolist() == [False]


def test_point_segment_distance() -> None:
    d = point_segment_distance(np.array([[0.0, 1.0], [2.0, 0.0]]), np.array([[-1.0, 0.0]] * 2), np.array([[1.0, 0.0]] * 2))
    assert d.tolist() == pytest.approx([1.0, 1.0])


def test_pair_track_follows_ids_not_order() -> None:
    ids = [[1, 2], [2, 1], [2]]
    pos = [[0.0, 0.0, 0.0, 5.0, 5.0, 0.0], [5.0, 5.0, 0.0, 1.0, 1.0, 0.0], [5.0, 5.0, 0.0]]
    track = pair_track(ids, pos, 1)
    assert track[:2].tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert np.isnan(track[2]).all()


def _episode(release: bool = True) -> AlignedEpisodeBundle:
    """Robot drives +x at 1 m/s, stalls 4 s right after the pair releases, and passes between them during the hold."""
    t = np.arange(0.0, 22.0, 0.05)
    x = np.empty_like(t)
    for i, ti in enumerate(t):
        x[i] = min(ti, 12.0) if ti < 12.0 else (12.0 if ti < 16.0 else 12.0 + (ti - 16.0))
    y = np.zeros_like(t)
    # the pair straddles y=0 at x=6 and the robot crosses their segment around t=6
    peds_ids = [[1, 2]] * len(t)
    peds_positions = [[6.0, -0.6, 0.0, 6.0, 0.6, 0.0]] * len(t)
    data = pl.DataFrame(
        {
            "time_ns": (t * SEC).astype(np.int64),
            "pos_x": x,
            "pos_y": y,
            "yaw": np.zeros_like(t),
            "peds_ids": peds_ids,
            "peds_positions": peds_positions,
        }
    )
    names = ["ACTIVATED", "HOLD_ONSET"] + (["RELEASED"] if release else [])
    times = [4.0, 5.0] + ([11.0] if release else [])
    events = pl.DataFrame(
        {
            "time_ns": [int(s * SEC) for s in times],
            "interaction_id": [3] * len(names),
            "interaction_type": [10] * len(names),
            "event_name": names,
            "participants": [[1, 2]] * len(names),
        }
    )
    return AlignedEpisodeBundle(episode_id=0, data=data, start_pos=[0.0, 0.0, 0.0], goal_pos=[30.0, 0.0], topics={"interaction_events": events})


def test_calculator_on_a_synthetic_episode() -> None:
    out = ContactInteractionCalculator(RobotParams()).calculate(_episode(), {})
    assert out["hhi_interaction"] == "hug"
    assert (out["hhi_hold_onset_s"], out["hhi_release_s"]) == (pytest.approx(5.0), pytest.approx(11.0))
    assert out["hhi_stall_count"] == 1
    assert out["hhi_stall_onsets_s"][0] == pytest.approx(12.0, abs=0.2)
    assert (out["hhi_stalls_approach"], out["hhi_stalls_hold"], out["hhi_stalls_post"], out["hhi_stalls_rest"]) == (0, 0, 1, 0)
    shares = [out[f"hhi_share_{w}"] for w in ("approach", "hold", "post", "rest")]
    assert sum(shares) == pytest.approx(1.0)
    assert out["hhi_share_post"] == pytest.approx(5.0 / out["hhi_episode_s"])
    assert out["hhi_passes_between"] == 1
    assert out["hhi_pair_gap_hold"] == pytest.approx(1.2)
    assert out["hhi_min_pair_clearance"] == pytest.approx(0.0, abs=0.06)
    assert set(out) == set(ContactInteractionCalculator.output_keys())


def test_calculator_without_a_release_holds_to_the_end() -> None:
    out = ContactInteractionCalculator(RobotParams()).calculate(_episode(release=False), {})
    assert out["hhi_release_s"] is None
    assert out["hhi_stalls_hold"] == 1 and out["hhi_stalls_post"] == 0


def test_calculator_without_contact_events() -> None:
    ep = _episode()
    ep.topics = {}
    out = ContactInteractionCalculator(RobotParams()).calculate(ep, {})
    assert out["hhi_interaction"] is None and out["hhi_hold_onset_s"] is None
    assert out["hhi_stalls_approach"] == out["hhi_stall_count"] == 1
    assert out["hhi_passes_between"] is None
