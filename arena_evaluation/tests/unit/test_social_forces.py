"""Scoped, vectorized social forces: Helbing magnitude, anisotropy, personal-space index and typed nulls."""

from __future__ import annotations

import math

import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.social_forces import (
    SFM_INTERACTION_RANGE_M,
    SFM_INTERACTION_STRENGTH_N,
    SocialForcesCalculator,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS_M = 0.3
ROBOT_TRACK_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "pose_source": pl.String,
    }
)


def _sfm(distance: float) -> float:
    """Helbing SFM magnitude for one pedestrian at center distance `distance`."""
    return SFM_INTERACTION_STRENGTH_N * math.exp(
        (RADIUS + PED_RADIUS_M - distance) / SFM_INTERACTION_RANGE_M
    )


def _episode(
    times_ns: list[int],
    peds_positions: list[list[float]],
    robot_xy: list[tuple[float, float]],
    yaw: list[float],
    headings: list[list[float]] | None = None,
    robot_times_ns: list[int] | None = None,
) -> AlignedEpisodeBundle:
    """Flat pedestrian frames; ``robot_times_ns`` may end before the pedestrian axis to leave it unresolved."""
    frames = []
    for i, (t, flat) in enumerate(zip(times_ns, peds_positions, strict=True)):
        pedestrians = []
        for j in range(len(flat) // 2):
            ped = {"x": flat[2 * j], "y": flat[2 * j + 1]}
            if headings is not None:
                ped["yaw"] = headings[i][j]
            pedestrians.append(ped)
        frames.append((t, pedestrians))
    peds = flat_peds(frames).with_columns(
        radius_m=pl.lit(PED_RADIUS_M, dtype=pl.Float64)
    )
    robot_times = times_ns if robot_times_ns is None else robot_times_ns
    robot = pl.DataFrame(
        {
            "t_ns": robot_times,
            "x": [p[0] for p in robot_xy[: len(robot_times)]],
            "y": [p[1] for p in robot_xy[: len(robot_times)]],
            "yaw": list(yaw[: len(robot_times)]),
            "pose_source": ["tf_gt"] * len(robot_times),
        },
        schema=ROBOT_TRACK_SCHEMA,
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(
            {
                "t_ns": times_ns,
                "x": [p[0] for p in robot_xy],
                "y": [p[1] for p in robot_xy],
                "yaw": list(yaw),
            }
        ),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics={"robot_tracks": robot, "pedestrian_arena_peds": peds},
        peds=peds,
    )


@pytest.fixture
def calc() -> SocialForcesCalculator:
    return SocialForcesCalculator(make_robot_params(robot_radius=RADIUS))


def test_an_unresolved_frame_is_null_not_zero(calc: SocialForcesCalculator):
    """A pedestrian frame further than the asof tolerance from the robot track has no force, never a fabricated zero."""
    episode = _episode(
        [0, SEC],
        [[1.0, 0.0], [1.0, 0.0]],
        [(0.0, 0.0), (0.0, 0.0)],
        [0.0, 0.0],
        robot_times_ns=[0],
    )
    table = calc.series_table(episode)
    forces = table.get_column("sfm_force_n").to_list()
    assert forces[0] == pytest.approx(_sfm(1.0))
    assert forces[1] is None
    results = calc.calculate(episode, {})
    # Only the resolved frame contributes to the summaries.
    assert results["sfm_peak_force"] == pytest.approx(_sfm(1.0))


def test_a_frame_without_robot_state_is_skipped_in_the_cumulative_force(
    calc: SocialForcesCalculator,
):
    """An unresolved frame in the middle is skipped row by row: its force and its duration stay aligned."""
    episode = _episode(
        [0, SEC, 2 * SEC],
        [[1.0, 0.0]] * 3,
        [(0.0, 0.0)] * 3,
        [0.0] * 3,
        robot_times_ns=[0, 2 * SEC],
    )
    frames = calc._geometry(episode).frames
    duration_s = dict(
        zip(frames.get_column("t_ns").to_list(), frames.get_column("dt_s").to_list())
    )
    expected = sum(
        _sfm(1.0) * duration_s[t] for t in (0, 2 * SEC) if duration_s[t] is not None
    )
    assert calc.calculate(episode, {})["sfm_cumulative_force"] == pytest.approx(
        expected
    )


def _slice(scope: MetricScope, bundle: AlignedEpisodeBundle) -> ScopeSlice:
    """Scope slice over one bundle, used to prove which population a scope measures."""
    return ScopeSlice(
        scope=scope,
        slice_id="slice",
        t_start_ns=0,
        t_end_ns=SEC,
        identity={},
        window_bundle=bundle,
        full_episode=bundle,
    )


def test_episode_scope_recomputes_and_unserved_scopes_do_not_masquerade(
    calc: SocialForcesCalculator,
):
    """The EPISODE scope is the calculator's own population; a scope it does not serve yields typed nulls."""
    episode = _episode([0, SEC], [[1.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    assert calc.calculate_for_scope(
        episode, {}, _slice(MetricScope.EPISODE, episode)
    ) == calc.calculate(episode, {})
    pedestrian_slice = _slice(MetricScope.PEDESTRIAN, episode)
    assert all(
        value is None
        for value in calc.calculate_for_scope(episode, {}, pedestrian_slice).values()
    )


