import polars as pl
import pytest

from arena_evaluation.processing.metrics.ecological.energy import EnergyMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _episode(**columns):
    n = len(next(iter(columns.values())))
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": [i * SEC for i in range(n)], **columns}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[1.0, 0.0],
    )


@pytest.fixture
def calc():
    return EnergyMetricCalculator(RobotParams(robot_radius=0.25))


def test_missing_power_and_battery_columns_give_null_metrics(calc):
    results = calc.calculate(_episode(vel_linear=[0.1, 0.2, 0.3]), {})
    for key in (
        "energy_static_wh",
        "energy_mechanical_wh",
        "energy_thermal_wh",
        "energy_total_wh",
        "power_peak_w",
        "battery_soc_final",
        "battery_soc_drop_pct",
        "timeseries_power_total_w",
        "timeseries_battery_soc",
    ):
        assert results[key] is None, key
    assert results["timeseries_velocity_linear"] == pytest.approx([0.1, 0.2, 0.3])


def test_all_null_power_column_gives_null_metrics(calc):
    nulls = pl.Series([None, None, None], dtype=pl.Float64)
    results = calc.calculate(_episode(total_power_w=nulls, battery_soc_percent=nulls), {})
    assert results["energy_total_wh"] is None
    assert results["power_peak_w"] is None
    assert results["battery_soc_final"] is None
    assert results["timeseries_power_total_w"] is None


def test_leading_power_nulls_are_back_filled(calc):
    power = pl.Series([None, 36.0, 72.0], dtype=pl.Float64)
    results = calc.calculate(_episode(total_power_w=power), {})
    assert results["timeseries_power_total_w"] == pytest.approx([36.0, 36.0, 72.0])
    assert results["energy_total_wh"] == pytest.approx((36.0 + 72.0) / 3600.0)
    assert results["power_peak_w"] == pytest.approx(72.0)
    assert results["energy_static_wh"] is None


def test_cumulative_energy_topic_wins_without_power(calc):
    results = calc.calculate(_episode(total_energy_consumed_wh=[1.0, 1.5, 2.25]), {})
    assert results["energy_total_wh"] == pytest.approx(1.25)
    assert results["power_peak_w"] is None
