"""Scoped, vectorized social forces: Helbing magnitude, anisotropy, personal-space index and typed nulls."""

from __future__ import annotations

import math

import polars as pl
import pytest
from flat_peds import flat_peds
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import ScopeSlice
from arena_evaluation.processing.metrics.social.social_forces import (
    CI_SIGMA_X_M,
    ESFM_ANISOTROPY,
    SERIES_SCHEMA,
    SFM_CUTOFF_M,
    SFM_INTERACTION_RANGE_M,
    SFM_INTERACTION_STRENGTH_N,
    SFM_MAX_FORCE_N,
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
    return SFM_INTERACTION_STRENGTH_N * math.exp((RADIUS + PED_RADIUS_M - distance) / SFM_INTERACTION_RANGE_M)


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
    peds = flat_peds(frames).with_columns(radius_m=pl.lit(PED_RADIUS_M, dtype=pl.Float64))
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
                "pos_x_gt": [p[0] for p in robot_xy],
                "pos_y_gt": [p[1] for p in robot_xy],
                "yaw_gt": list(yaw),
            }
        ),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics={"robot_tracks": robot, "pedestrian_tracks": peds},
        peds=peds,
    )


@pytest.fixture
def calc() -> SocialForcesCalculator:
    return SocialForcesCalculator(make_robot_params(robot_radius=RADIUS))


def test_no_nested_series_column_is_an_output_key():
    """The per-frame force and index series are a 1NF table, never list-valued metrics."""
    assert not any(key.startswith("timeseries_") for key in SocialForcesCalculator.output_keys())
    assert set(SERIES_SCHEMA) == {
        "run_id",
        "episode_id",
        "robot_id",
        "t_ns",
        "time_s",
        "sfm_force_n",
        "ci",
    }


def test_no_topics_yields_all_none(calc: SocialForcesCalculator):
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)
    results = calc.calculate(episode, {})
    assert set(results) == set(calc.output_keys())
    assert all(value is None for value in results.values())


def test_sfm_magnitude_matches_helbing(calc: SocialForcesCalculator):
    d = 1.0
    episode = _episode([0, SEC], [[d, 0.0], [d, 0.0]], [(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert results["sfm_mean_force"] == pytest.approx(_sfm(d), rel=1e-9)
    # One second of exposure, the final frame carries a negligible dt.
    assert results["sfm_cumulative_force"] == pytest.approx(_sfm(d), rel=1e-3)


def test_esfm_de_weights_pedestrians_behind_the_robot(calc: SocialForcesCalculator):
    """A ped ahead is fully weighted; the same ped behind is de-weighted to lambda."""
    d = 1.0
    ahead = _episode([0, SEC], [[d, 0.0], [d, 0.0]], [(0.0, 0.0)] * 2, [0.0, 0.0])
    behind = _episode([0, SEC], [[-d, 0.0], [-d, 0.0]], [(0.0, 0.0)] * 2, [0.0, 0.0])

    r_ahead = calc.calculate(ahead, {})
    r_behind = calc.calculate(behind, {})

    assert r_ahead["sfm_peak_force"] == pytest.approx(r_behind["sfm_peak_force"])
    assert r_ahead["esfm_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert r_behind["esfm_peak_force"] == pytest.approx(_sfm(d) * ESFM_ANISOTROPY, rel=1e-9)


def test_esfm_ped_variant_uses_pedestrian_heading(calc: SocialForcesCalculator):
    """The ped-heading variant weights by where the pedestrian faces, not the robot."""
    d = 1.0
    facing_robot = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0], headings=[[math.pi]] * 2)
    facing_away = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0], headings=[[0.0]] * 2)

    assert calc.calculate(facing_robot, {})["esfm_ped_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert calc.calculate(facing_away, {})["esfm_ped_peak_force"] == pytest.approx(_sfm(d) * ESFM_ANISOTROPY, rel=1e-9)


def test_pedestrians_beyond_cutoff_contribute_no_force_but_still_score_ci(
    calc: SocialForcesCalculator,
):
    far = SFM_CUTOFF_M + 1.0
    episode = _episode([0, SEC], [[far, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == 0.0
    assert results["esfm_peak_force"] == 0.0
    assert results["ci_max"] == pytest.approx(math.exp(-(far**2) / (2.0 * CI_SIGMA_X_M**2)), rel=1e-9)


def test_ci_is_the_gaussian_personal_space_index(calc: SocialForcesCalculator):
    d = 0.28  # one sigma
    episode = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    expected = math.exp(-(d**2) / (2.0 * CI_SIGMA_X_M**2))
    assert results["ci_max"] == pytest.approx(expected, rel=1e-9)
    assert results["ci_mean"] == pytest.approx(expected, rel=1e-9)


def test_ci_peaks_at_contact(calc: SocialForcesCalculator):
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    assert calc.calculate(episode, {})["ci_max"] == pytest.approx(1.0)


def test_frames_without_pedestrians_score_zero(calc: SocialForcesCalculator):
    far = SFM_CUTOFF_M + 1.0
    episode = _episode([0, SEC], [[], [far, 0.0]], [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == 0.0


def test_force_at_contact_stays_finite(calc: SocialForcesCalculator):
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == pytest.approx(_sfm(0.0), rel=1e-9)
    assert math.isfinite(results["sfm_cumulative_force"])


def test_force_is_clamped_for_a_large_footprint():
    """A wide robot at contact would blow up the exponential without the clamp."""
    calc = SocialForcesCalculator(make_robot_params(robot_radius=1.5))
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == pytest.approx(SFM_MAX_FORCE_N)


def test_series_table_has_one_flat_row_per_pedestrian_frame(calc: SocialForcesCalculator):
    """The 1NF series replaces the dropped list keys with one row per frame."""
    times = [0, SEC, 2 * SEC, 3 * SEC]
    episode = _episode(times, [[1.0, 0.0]] * 4, [(0.0, 0.0)] * 4, [0.0] * 4)
    table = calc.series_table(episode, run_id="run-1", robot_id="jackal_0")
    assert table.schema == SERIES_SCHEMA
    assert table.height == len(times)
    assert table.get_column("t_ns").to_list() == times
    assert table.get_column("time_s").to_list() == [0.0, 1.0, 2.0, 3.0]
    assert table.get_column("sfm_force_n").to_list() == pytest.approx([_sfm(1.0)] * 4)


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


def test_series_table_is_empty_with_schema_without_pedestrians(calc: SocialForcesCalculator):
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)
    assert_frame_equal(calc.series_table(episode), pl.DataFrame(schema=SERIES_SCHEMA))


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
    assert calc.calculate_for_scope(episode, {}, _slice(MetricScope.EPISODE, episode)) == calc.calculate(episode, {})
    pedestrian_slice = _slice(MetricScope.PEDESTRIAN, episode)
    assert all(value is None for value in calc.calculate_for_scope(episode, {}, pedestrian_slice).values())


def test_encounter_scope_recomputes_the_window_telemetry(
    calc: SocialForcesCalculator,
):
    """An ENCOUNTER slice measures its own window bundle, never the episode bundle passed alongside it."""
    episode = _episode(
        [0, SEC, 2 * SEC, 3 * SEC],
        [[1.0, 0.0]] * 4,
        [(0.0, 0.0)] * 4,
        [0.0] * 4,
    )
    window = _episode([0, SEC], [[1.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    encounter_slice = _slice(MetricScope.ENCOUNTER, window)
    window_metrics = calc.calculate_for_scope(episode, {}, encounter_slice)
    assert window_metrics == calc.calculate(window, {})
    assert window_metrics["sfm_peak_force"] == pytest.approx(_sfm(1.0), rel=1e-9)
    assert window_metrics["sfm_cumulative_force"] == pytest.approx(_sfm(1.0), rel=1e-4)
