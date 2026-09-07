import polars as pl
import pytest

from arena_evaluation.processing.metrics.ecological.energy_extended import (
    EnergyExtendedCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

MASS = 18.431
G = 9.81


def _episode():
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": [0, 1_000_000_000]}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0],
    )


@pytest.fixture
def calc():
    return EnergyExtendedCalculator(RobotParams(robot_radius=0.25, base_mass=MASS))


def test_cost_of_transport_is_unreported_without_a_declared_mass():
    """An undeclared mass (0 kg) skips CoT rather than inventing a number."""
    calc = EnergyExtendedCalculator(RobotParams(robot_radius=0.25))
    results = calc.calculate(_episode(), {"energy_total_wh": 1.0, "path_length": 10.0})
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] == pytest.approx(0.1)


def test_without_episode_data_all_outputs_are_none(calc):
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    results = calc.calculate(empty, {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_cost_of_transport_is_energy_over_weight_times_distance(calc):
    results = calc.calculate(_episode(), {"energy_total_wh": 1.0, "path_length": 10.0})
    assert results["specific_cost_of_transport"] == pytest.approx(3600.0 / (MASS * G * 10.0), rel=1e-9)


def test_energy_per_meter(calc):
    results = calc.calculate(_episode(), {"energy_total_wh": 2.5, "path_length": 10.0})
    assert results["energy_per_meter"] == pytest.approx(0.25)


def test_negligible_path_length_is_rejected(calc):
    results = calc.calculate(_episode(), {"energy_total_wh": 1.0, "path_length": 0.05})
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] is None


def test_missing_priors_yield_none_not_a_crash(calc):
    results = calc.calculate(_episode(), {})
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] is None
    assert results["peak_to_mean_power_ratio"] is None
    assert results["standstill_energy_penalty_wh"] is None


def test_peak_to_mean_power_ratio_is_median_filtered(calc):
    """A single-frame spike is filtered down instead of dominating the ratio."""
    results = calc.calculate(_episode(), {"timeseries_power_total_w": [10.0, 10.0, 10.0, 30.0]})
    # Filtered series is [10, 10, 10, 20]: peak 20 over mean 12.5.
    assert results["peak_to_mean_power_ratio"] == pytest.approx(20.0 / 12.5)


def test_flat_power_gives_unit_ratio(calc):
    results = calc.calculate(_episode(), {"timeseries_power_total_w": [12.0] * 5})
    assert results["peak_to_mean_power_ratio"] == pytest.approx(1.0)


def test_zero_power_leaves_the_ratio_undefined(calc):
    results = calc.calculate(_episode(), {"timeseries_power_total_w": [0.0, 0.0, 0.0]})
    assert results["peak_to_mean_power_ratio"] is None


def test_standstill_penalty_only_bills_stationary_frames(calc):
    prior = {
        "velocity": [0.0, 0.0, 1.0, 1.0],
        "timeseries_power_total_w": [10.0, 10.0, 50.0, 50.0],
        "timeseries_time_s": [0.0, 1.0, 2.0, 3.0],
    }
    results = calc.calculate(_episode(), prior)
    # One stationary second at 10 W; the moving frames are not charged.
    assert results["standstill_energy_penalty_wh"] == pytest.approx(10.0 / 3600.0)


def test_short_standstill_blocks_are_ignored(calc):
    prior = {
        "velocity": [1.0, 0.0, 1.0, 1.0],
        "timeseries_power_total_w": [10.0, 10.0, 10.0, 10.0],
        "timeseries_time_s": [0.0, 0.1, 0.2, 0.3],
    }
    results = calc.calculate(_episode(), prior)
    assert results["standstill_energy_penalty_wh"] == pytest.approx(0.0)


def test_a_robot_that_never_moves_is_billed_for_the_whole_run(calc):
    prior = {
        "velocity": [0.0, 0.0, 0.0],
        "timeseries_power_total_w": [20.0, 20.0, 20.0],
        "timeseries_time_s": [0.0, 1.0, 2.0],
    }
    results = calc.calculate(_episode(), prior)
    assert results["standstill_energy_penalty_wh"] == pytest.approx(40.0 / 3600.0)


def test_timeseries_of_unequal_length_are_truncated(calc):
    prior = {
        "velocity": [0.0, 0.0, 0.0, 0.0],
        "timeseries_power_total_w": [10.0, 10.0],
        "timeseries_time_s": [0.0, 1.0, 2.0],
    }
    results = calc.calculate(_episode(), prior)
    assert results["standstill_energy_penalty_wh"] == pytest.approx(10.0 / 3600.0)
