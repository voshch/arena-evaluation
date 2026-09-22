import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import (
    ProxemicsExtendedCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
RADIUS = 0.25
D_COMBINED = RADIUS + ProxemicsExtendedCalculator._PED_RADIUS


def _at_clearance(d_eff: float) -> list[float]:
    """One pedestrian straight ahead at the given edge-to-edge clearance."""
    return [d_eff + D_COMBINED, 0.0]


def _episode(times_ns, peds_positions, robot_xy=None, twists=None):
    robot_xy = robot_xy or [(0.0, 0.0)] * len(times_ns)
    peds = {
        "time_ns": times_ns,
        "peds_positions": peds_positions,
        "num_pedestrians": [len(p) // 2 for p in peds_positions],
    }
    if twists is not None:
        peds["peds_twists"] = twists
    topics = {
        "peds": pl.DataFrame(peds),
        "tf_gt": pl.DataFrame(
            {
                "time_ns": times_ns,
                "pos_x_gt": [p[0] for p in robot_xy],
                "pos_y_gt": [p[1] for p in robot_xy],
                "yaw_gt": [0.0] * len(times_ns),
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
    return ProxemicsExtendedCalculator(RobotParams(robot_radius=RADIUS))


def test_output_keys_do_not_collide_with_legacy_proxemics():
    """Both calculators land in the same results dict, so keys must be disjoint."""
    overlap = set(ProxemicsCalculator.output_keys()) & set(ProxemicsExtendedCalculator.output_keys())
    assert overlap == set()


def test_no_topics_yields_all_none(calc):
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[], num_pedestrians=0)
    results = calc.calculate(episode, {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_time_is_attributed_to_the_hall_zone_of_the_edge_distance(calc):
    """One second in each zone, classified on d_eff rather than center distance."""
    times = [0, SEC, 2 * SEC, 3 * SEC]
    peds = [
        _at_clearance(0.2),  # intimate
        _at_clearance(0.8),  # personal
        _at_clearance(2.0),  # social
        _at_clearance(5.0),  # public, final frame carries a negligible dt
    ]
    results = calc.calculate(_episode(times, peds), {})

    assert results["time_in_intimate_zone"] == pytest.approx(1.0)
    assert results["time_in_personal_zone"] == pytest.approx(1.0)
    assert results["time_in_social_zone"] == pytest.approx(1.0)
    assert results["time_in_public_zone"] == pytest.approx(0.0, abs=1e-3)


def test_zone_boundaries_use_edge_to_edge_not_center_distance(calc):
    """A ped 1.0 m from the robot center is intimate once radii are removed."""
    center_distance = 1.0
    results = calc.calculate(_episode([0, SEC], [[center_distance, 0.0]] * 2), {})
    assert results["time_in_intimate_zone"] > 0.0
    assert results["time_in_personal_zone"] == 0.0
    assert results["timeseries_min_ped_clearance"][0] == pytest.approx(center_distance - D_COMBINED)


def test_psii_integrates_clearance_inside_personal_space(calc):
    times = [0, SEC, 2 * SEC]
    peds = [_at_clearance(0.2), _at_clearance(0.8), _at_clearance(3.0)]
    results = calc.calculate(_episode(times, peds), {})
    # Penetration depth: (1.2 - 0.2) * 1s + (1.2 - 0.8) * 1s = 1.0 + 0.4 = 1.4 m·s; social frame is outside the band.
    assert results["personal_space_intrusion_integral"] == pytest.approx(1.4)


def test_frames_without_pedestrians_report_no_clearance(calc):
    results = calc.calculate(_episode([0, SEC], [[], []]), {})
    assert results["timeseries_min_ped_clearance"] == [None, None]
    assert results["movement_towards_peds_ratio"] is None
    assert results["time_in_intimate_zone"] == 0.0


def test_movement_towards_peds_ratio_counts_approaching_frames(calc):
    """Robot drives +x with the ped ahead, so every moving frame is approaching."""
    times = [0, SEC, 2 * SEC]
    robot = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    peds = [[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]
    results = calc.calculate(_episode(times, peds, robot_xy=robot), {})
    assert results["movement_towards_peds_ratio"] == pytest.approx(2 / 3)


def test_tti_is_none_when_nothing_is_closing(calc):
    times = [0, SEC]
    peds = [_at_clearance(2.0), _at_clearance(2.0)]
    results = calc.calculate(_episode(times, peds), {})
    assert results["tti_min"] is None
    assert results["tti_mean"] is None


def test_tti_uses_relative_velocity_of_an_approaching_pedestrian(calc):
    """Ped 2 m clear closing at 1 m/s on a stationary robot gives TTI = 2 s."""
    times = [0, SEC]
    peds = [_at_clearance(2.0), _at_clearance(2.0)]
    twists = [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    results = calc.calculate(_episode(times, peds, twists=twists), {})
    assert results["tti_min"] == pytest.approx(2.0, rel=1e-6)


def test_max_speed_is_recorded_per_zone(calc):
    """Speed is only credited to the zone the robot was in at that frame."""
    times = [0, SEC, 2 * SEC]
    robot = [(0.0, 0.0), (0.0, 0.0), (2.0, 0.0)]
    peds = [
        [_at_clearance(0.2)[0], 0.0],
        [_at_clearance(0.2)[0], 0.0],
        [2.0 + _at_clearance(3.0)[0], 0.0],
    ]
    results = calc.calculate(_episode(times, peds, robot_xy=robot), {})
    assert results["max_speed_intimate_zone"] == pytest.approx(0.0)
    assert results["max_speed_social_zone"] == pytest.approx(2.0)


def test_psi_events_merge_within_the_gap_and_split_beyond_it(calc):
    """Intrusions closer than the 2 s gap are one event, further apart are two."""
    near, far = _at_clearance(0.2), _at_clearance(3.0)

    close_times = [0, SEC // 2, SEC]
    close = calc.calculate(_episode(close_times, [near, far, near]), {})
    assert close["psi_intimate_events"] == 1

    spread_times = [0, 5 * SEC // 2, 5 * SEC]
    spread = calc.calculate(_episode(spread_times, [near, far, near]), {})
    assert spread["psi_intimate_events"] == 2


def test_psi_events_are_zero_without_intrusions(calc):
    results = calc.calculate(_episode([0, SEC], [_at_clearance(3.0)] * 2), {})
    assert results["psi_intimate_events"] == 0
    assert results["psi_personal_events"] == 0
