"""Episode-scope gaze and proxemics calculators: typed nulls, 1NF series tables, no list columns."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.gaze import (
    SERIES_SCHEMA as GAZE_SERIES_SCHEMA,
    GazeMetricsCalculator,
)
from arena_evaluation.processing.metrics.social.interaction_common import (
    extract_episode_ped_tracks,
)
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import (
    PedestrianDisturbanceCalculator,
)
from arena_evaluation.processing.metrics.social.proxemics import (
    PERSONAL_SPACE_SERIES_SCHEMA,
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
    return cls(make_robot_params(robot_radius=0.2, laser_min_range=0.0, laser_max_range=10.0))


def test_proxemics_no_peds():
    results = _calc(ProxemicsCalculator).calculate(EMPTY_EPISODE, {})
    assert results["total_time_in_personal_space"] is None
    assert results["avg_velocity_in_personal_space"] is None
    assert results["num_pedestrians"] is None


def test_proxemics_series_table_is_flat():
    table = _calc(ProxemicsCalculator).series_table(EMPTY_EPISODE)
    assert_frame_equal(table, pl.DataFrame(schema=PERSONAL_SPACE_SERIES_SCHEMA))


def test_proxemics_has_no_nested_series_key():
    assert "time_in_personal_space" not in ProxemicsCalculator.output_keys()


def test_gaze_no_peds():
    results = _calc(GazeMetricsCalculator).calculate(EMPTY_EPISODE, {})
    assert results["total_time_looking_at_pedestrians"] is None
    assert results["total_time_looked_at_by_pedestrians"] is None


def test_gaze_series_table_is_flat():
    table = _calc(GazeMetricsCalculator).series_table(EMPTY_EPISODE)
    assert_frame_equal(table, pl.DataFrame(schema=GAZE_SERIES_SCHEMA))


def test_gaze_has_no_nested_series_key():
    assert not any(
        key.startswith("time_looking_") or key.startswith("time_looked_")
        for key in GazeMetricsCalculator.output_keys()
    )


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
        topics={"robot_tracks": robot, "pedestrian_tracks": peds},
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


def test_gaze_encounter_scope_measures_the_window_samples():
    """The gaze dwell is the window's own sample dwell, not an episode total."""
    window = _window_bundle(ped_x_m=2.0)
    results = _calc(GazeMetricsCalculator).calculate_for_scope(
        window, {}, _window_slice(window)
    )
    assert results["total_time_looking_at_pedestrians"] == pytest.approx(0.2)
    assert results["total_time_looked_at_by_pedestrians"] == pytest.approx(0.0)


def test_proxemics_encounter_scope_measures_the_window_samples():
    """The personal-space dwell is the window's own sample dwell and the window's own pedestrian set."""
    window = _window_bundle(ped_x_m=1.0)
    results = _calc(ProxemicsCalculator).calculate_for_scope(
        window, {}, _window_slice(window)
    )
    assert results["num_pedestrians"] == 1
    assert results["total_time_in_personal_space"] == pytest.approx(0.2)
    assert results["avg_velocity_in_personal_space"] == pytest.approx(1.0)


def test_intrusion_count_is_recomputed_not_taken_from_a_prior_total():
    """An episode without tracks has a NULL intrusion count, even when prior_results carries a total."""
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    prior = {"total_intrusions_intimate": 99, "total_intrusions_personal": 99}
    results = _calc(PedestrianIntrusionCalculator).calculate(empty, prior)
    assert results["total_intrusions_intimate"] is None


def test_personal_space_velocity_comes_from_pose_in_the_encounter_scope_too():
    """A zero recorded twist does not publish 0 m/s: both scopes read the ground-truth position difference."""
    window = _window_bundle(ped_x_m=1.0)
    window.topics["robot_tracks"] = window.topics["robot_tracks"].with_columns(
        pl.lit(0.0).alias("vx")
    )
    results = _calc(ProxemicsCalculator).calculate_for_scope(
        window, {}, _window_slice(window)
    )
    assert results["avg_velocity_in_personal_space"] == pytest.approx(1.0)


def test_velocity_delay_ratio_uses_the_recorded_desired_speed():
    """The delay is relative to the pedestrian's own desired_speed (0.8 m/s), not the nominal 1.2 m/s."""
    count = 21
    t_ns = np.arange(count, dtype=np.int64) * 100_000_000
    frame = pl.DataFrame(
        {
            "t_ns": t_ns,
            "ped_id": ["Pedestrian_2"] * count,
            "ped_name": ["bob"] * count,
            "x": np.arange(count, dtype=np.float64) * 0.04,
            "y": np.zeros(count),
            "vx": np.full(count, 0.4),
            "vy": np.zeros(count),
            "radius_m": np.full(count, 0.3),
            "desired_speed": np.full(count, 0.8),
        }
    )
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0, 0.0],
        topics={"pedestrian_tracks": frame},
    )

    results = _calc(PedestrianDisturbanceCalculator).calculate(episode, {})

    # 0.8 m travelled over 2 s is 0.4 m/s, half of the 0.8 m/s the pedestrian asked for.
    assert results["ped_velocity_delay_ratio"] == pytest.approx(0.5, abs=1e-3)


def test_a_pedestrian_without_recorded_twist_has_no_first_velocity_sample():
    """The first velocity of a pedestrian with no recorded twist is NaN, not a copy of the second sample."""
    episode = AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame(),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0, 0.0],
        topics={
            "pedestrian_tracks": pl.DataFrame(
                {
                    "ped_id": ["Pedestrian_1"] * 4,
                    "ped_name": ["Pedestrian_1"] * 4,
                    "t_ns": np.arange(4, dtype=np.int64) * 100_000_000,
                    "x": np.arange(4, dtype=np.float64) * 0.1,
                    "y": np.zeros(4),
                    "vx": np.zeros(4),
                    "vy": np.zeros(4),
                    "radius_m": [0.3] * 4,
                    "desired_speed": [1.0] * 4,
                }
            )
        },
    )
    track = extract_episode_ped_tracks(episode)["Pedestrian_1"]
    assert np.isnan(track.trajectory.vx[0])
    assert track.trajectory.vx[1] == pytest.approx(1.0)
