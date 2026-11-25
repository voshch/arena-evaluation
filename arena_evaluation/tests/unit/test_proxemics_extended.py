"""Scoped, vectorized proxemics_extended: zone dwell, events, closing time and typed missingness."""

from __future__ import annotations

import polars as pl
import pytest
from flat_peds import flat_peds

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.proxemics_extended import (
    ProxemicsExtendedCalculator,
)
from arena_evaluation.storage.exceptions import MetricInputError
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS_M = 0.3
D_COMBINED = RADIUS + PED_RADIUS_M
ROBOT_TRACK_SCHEMA = pl.Schema(
    {
        "t_ns": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "yaw": pl.Float64,
        "pose_source": pl.String,
    }
)


def _at_clearance(d_eff: float) -> list[float]:
    """One pedestrian straight ahead at the given edge-to-edge clearance."""
    return [d_eff + D_COMBINED, 0.0]


def _episode(
    times_ns: list[int],
    peds_positions: list[list[float]],
    robot_xy: list[tuple[float, float]] | None = None,
    twists: list[list[float]] | None = None,
    robot_times_ns: list[int] | None = None,
) -> AlignedEpisodeBundle:
    """Episode of flat pedestrian frames; ``robot_times_ns`` may end before the pedestrian axis to leave it unresolved."""
    n = len(times_ns)
    robot_xy = robot_xy or [(0.0, 0.0)] * n
    frames = []
    for i, (t, flat) in enumerate(zip(times_ns, peds_positions, strict=True)):
        pedestrians = []
        for j in range(len(flat) // 2):
            ped = {"x": flat[2 * j], "y": flat[2 * j + 1]}
            if twists is not None:
                ped["vx"], ped["vy"] = twists[i][3 * j], twists[i][3 * j + 1]
            pedestrians.append(ped)
        frames.append((t, pedestrians))
    peds = flat_peds(frames).with_columns(radius_m=pl.lit(PED_RADIUS_M, dtype=pl.Float64))
    robot_times = times_ns if robot_times_ns is None else robot_times_ns
    robot = pl.DataFrame(
        {
            "t_ns": robot_times,
            "x": [p[0] for p in robot_xy[: len(robot_times)]],
            "y": [p[1] for p in robot_xy[: len(robot_times)]],
            "yaw": [0.0] * len(robot_times),
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
                "yaw": [0.0] * n,
            }
        ),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics={"robot_tracks": robot, "pedestrian_arena_peds": peds},
        peds=peds,
    )


@pytest.fixture
def calc() -> ProxemicsExtendedCalculator:
    return ProxemicsExtendedCalculator(make_robot_params(robot_radius=RADIUS))


@pytest.mark.parametrize("radius_m", [None, 0.0, -0.3])
def test_a_pedestrian_without_a_positive_recorded_radius_is_refused(calc: ProxemicsExtendedCalculator, radius_m: float | None):
    """The calculator never invents a body size; a missing or non-positive recorded radius raises."""
    episode = _episode([0, SEC], [[1.0, 0.0], [1.0, 0.0]])
    episode.topics["pedestrian_arena_peds"] = episode.topics["pedestrian_arena_peds"].with_columns(radius_m=pl.lit(radius_m, dtype=pl.Float64))
    with pytest.raises(MetricInputError):
        calc.calculate(episode, {})


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
    calc: ProxemicsExtendedCalculator,
):
    """The EPISODE scope is the calculator's own population; a scope it does not serve yields typed nulls."""
    episode = _episode([0, SEC], [_at_clearance(0.8), _at_clearance(0.8)])
    assert calc.calculate_for_scope(episode, {}, _slice(MetricScope.EPISODE, episode)) == calc.calculate(episode, {})
    pedestrian_slice = _slice(MetricScope.PEDESTRIAN, episode)
    assert all(value is None for value in calc.calculate_for_scope(episode, {}, pedestrian_slice).values())
