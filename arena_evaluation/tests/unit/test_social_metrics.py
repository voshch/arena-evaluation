"""Episode-scope gaze and proxemics calculators: typed nulls, 1NF series tables, no list columns."""

from __future__ import annotations

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.proxemics import (
    PedestrianIntrusionCalculator,
    ProxemicsCalculator,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

EMPTY_EPISODE = AlignedEpisodeBundle(
    episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0
)


def _calc(cls):
    return cls(
        make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0)
    )


def test_proxemics_no_peds():
    results = _calc(ProxemicsCalculator).calculate(EMPTY_EPISODE, {})
    assert results["total_time_in_personal_space"] is None
    assert results["avg_velocity_in_personal_space"] is None
    assert results["num_pedestrians"] is None


WINDOW_RATE_NS = 100_000_000


def _window_bundle(*, ped_x_m: float, count: int = 3) -> AlignedEpisodeBundle:
    """A window slice's own telemetry: a robot at the origin and one stationary pedestrian ahead of it."""
    t_ns = np.arange(count, dtype=np.int64) * WINDOW_RATE_NS
    robot = pl.DataFrame(
        {
            "t_ns": t_ns,
            "x": np.arange(count, dtype=np.float64) * WINDOW_RATE_NS / 1e9,
            "y": np.zeros(count, dtype=np.float64),
            "yaw": np.zeros(count, dtype=np.float64),
            "vx": np.ones(count, dtype=np.float64),
            "vy": np.zeros(count, dtype=np.float64),
        },
        schema={
            "t_ns": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            "yaw": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
        },
    )
    peds = pl.DataFrame(
        {
            "t_ns": t_ns,
            "ped_id": ["Pedestrian_1"] * count,
            "ped_name": ["alice"] * count,
            "x": np.full(count, ped_x_m, dtype=np.float64),
            "y": np.zeros(count, dtype=np.float64),
            "yaw": np.zeros(count, dtype=np.float64),
            "vx": np.zeros(count, dtype=np.float64),
            "vy": np.zeros(count, dtype=np.float64),
            "radius_m": np.full(count, 0.3, dtype=np.float64),
        },
        schema={
            "t_ns": pl.Int64,
            "ped_id": pl.String,
            "ped_name": pl.String,
            "x": pl.Float64,
            "y": pl.Float64,
            "yaw": pl.Float64,
            "vx": pl.Float64,
            "vy": pl.Float64,
            "radius_m": pl.Float64,
        },
    )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[],
        goal_pos=[],
        topics={"robot_tracks": robot, "pedestrian_arena_peds": peds},
    )


def _window_slice(window: AlignedEpisodeBundle) -> ScopeSlice:
    """ENCOUNTER scope slice over one window bundle."""
    return ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc-001-jackal-000",
        t_start_ns=0,
        t_end_ns=2 * WINDOW_RATE_NS,
        identity={},
        window_bundle=window,
        full_episode=window,
    )


def test_intrusion_count_is_recomputed_not_taken_from_a_prior_total():
    """An episode without tracks has a NULL intrusion count, even when prior_results carries a total."""
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    prior = {"total_intrusions_intimate": 99, "total_intrusions_personal": 99}
    results = _calc(PedestrianIntrusionCalculator).calculate(empty, prior)
    assert results["total_intrusions_intimate"] is None


