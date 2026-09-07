import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.social_forces import SocialForcesCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
RADIUS = 0.25
PED_RADIUS = SocialForcesCalculator._PED_RADIUS
A = SocialForcesCalculator._A
B = SocialForcesCalculator._B


def _sfm(distance: float) -> float:
    """Helbing SFM magnitude for one pedestrian at center distance `distance`."""
    return A * math.exp((RADIUS + PED_RADIUS - distance) / B)


def _episode(times_ns, peds_positions, robot_xy, yaw, headings=None):
    peds = {
        "time_ns": times_ns,
        "peds_positions": peds_positions,
        "num_pedestrians": [len(p) // 2 for p in peds_positions],
    }
    if headings is not None:
        peds["peds_headings"] = headings
    topics = {
        "peds": pl.DataFrame(peds),
        "tf_gt": pl.DataFrame(
            {
                "time_ns": times_ns,
                "pos_x_gt": [p[0] for p in robot_xy],
                "pos_y_gt": [p[1] for p in robot_xy],
                "yaw_gt": yaw,
            }
        ),
    }
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": times_ns}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        num_pedestrians=1,
        topics=topics,
    )


@pytest.fixture
def calc():
    return SocialForcesCalculator(RobotParams(robot_radius=RADIUS))


def test_no_topics_yields_all_none(calc):
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)
    results = calc.calculate(episode, {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_sfm_magnitude_matches_helbing(calc):
    d = 1.0
    episode = _episode([0, SEC], [[d, 0.0], [d, 0.0]], [(0.0, 0.0), (0.0, 0.0)], [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert results["sfm_mean_force"] == pytest.approx(_sfm(d), rel=1e-9)
    # One second of exposure, the final frame carries a negligible dt.
    assert results["sfm_cumulative_force"] == pytest.approx(_sfm(d), rel=1e-3)


def test_esfm_de_weights_pedestrians_behind_the_robot(calc):
    """A ped ahead is fully weighted; the same ped behind is de-weighted to lambda."""
    d = 1.0
    ahead = _episode([0, SEC], [[d, 0.0], [d, 0.0]], [(0.0, 0.0)] * 2, [0.0, 0.0])
    behind = _episode([0, SEC], [[-d, 0.0], [-d, 0.0]], [(0.0, 0.0)] * 2, [0.0, 0.0])

    r_ahead = calc.calculate(ahead, {})
    r_behind = calc.calculate(behind, {})

    assert r_ahead["sfm_peak_force"] == pytest.approx(r_behind["sfm_peak_force"])
    assert r_ahead["esfm_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert r_behind["esfm_peak_force"] == pytest.approx(_sfm(d) * SocialForcesCalculator._LAMBDA_ESFM, rel=1e-9)


def test_esfm_ped_variant_uses_pedestrian_heading(calc):
    """The ped-heading variant weights by where the pedestrian faces, not the robot."""
    d = 1.0
    facing_robot = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0], headings=[[math.pi]] * 2)
    facing_away = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0], headings=[[0.0]] * 2)
    lam = SocialForcesCalculator._LAMBDA_ESFM

    assert calc.calculate(facing_robot, {})["esfm_ped_peak_force"] == pytest.approx(_sfm(d), rel=1e-9)
    assert calc.calculate(facing_away, {})["esfm_ped_peak_force"] == pytest.approx(_sfm(d) * lam, rel=1e-9)


def test_pedestrians_beyond_cutoff_contribute_no_force_but_still_score_ci(calc):
    far = SocialForcesCalculator._CUTOFF + 1.0
    episode = _episode([0, SEC], [[far, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == 0.0
    assert results["esfm_peak_force"] == 0.0
    assert results["ci_max"] == pytest.approx(0.0, abs=1e-12)


def test_ci_is_the_gaussian_personal_space_index(calc):
    d = 0.28  # one sigma
    episode = _episode([0, SEC], [[d, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    expected = math.exp(-(d**2) / (2.0 * SocialForcesCalculator._SIGMA_PX0**2))
    assert results["ci_max"] == pytest.approx(expected, rel=1e-9)
    assert results["ci_mean"] == pytest.approx(expected, rel=1e-9)


def test_ci_peaks_at_contact(calc):
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    assert calc.calculate(episode, {})["ci_max"] == pytest.approx(1.0)


def test_frames_without_pedestrians_score_zero(calc):
    episode = _episode([0, SEC], [[], []], [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == 0.0
    assert results["ci_max"] == 0.0
    assert results["timeseries_sfm_force"] == [0.0, 0.0]


def test_force_at_contact_stays_finite(calc):
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == pytest.approx(_sfm(0.0), rel=1e-9)
    assert np.isfinite(results["sfm_cumulative_force"])


def test_force_is_clamped_for_a_large_footprint():
    """A wide robot at contact would blow up the exponential without the clamp."""
    calc = SocialForcesCalculator(RobotParams(robot_radius=1.5))
    episode = _episode([0, SEC], [[0.0, 0.0]] * 2, [(0.0, 0.0)] * 2, [0.0, 0.0])
    results = calc.calculate(episode, {})
    assert results["sfm_peak_force"] == SocialForcesCalculator._MAX_FORCE


def test_timeseries_length_matches_the_native_peds_axis(calc):
    times = [0, SEC, 2 * SEC, 3 * SEC]
    episode = _episode(times, [[1.0, 0.0]] * 4, [(0.0, 0.0)] * 4, [0.0] * 4)
    results = calc.calculate(episode, {})
    assert len(results["timeseries_sfm_force"]) == len(times)
    assert len(results["timeseries_ci"]) == len(times)
