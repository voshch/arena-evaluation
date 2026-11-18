import dataclasses

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.energy import EnergyCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000
MASS = 18.431
GRAVITY = 9.81


def _episode_at(
    times_ns: list[int], **columns: list[float | None] | pl.Series
) -> AlignedEpisodeBundle:
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": times_ns, **columns}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[10.0, 0.0],
    )


def _episode(**columns: list[float | None] | pl.Series) -> AlignedEpisodeBundle:
    count = len(next(iter(columns.values())))
    return _episode_at([index * SEC for index in range(count)], **columns)


@pytest.fixture
def calc() -> EnergyCalculator:
    return EnergyCalculator(RobotParams(robot_radius=0.25, base_mass=MASS))


def test_missing_power_and_battery_give_null_energy(calc):
    results = calc.calculate(_episode(vel_linear=[0.1, 0.2, 0.3]), {})
    assert set(results) == set(calc.output_keys())
    assert results["energy_total_wh"] is None
    assert results["power_peak_w"] is None
    assert results["battery_soc_final"] is None


def test_all_null_power_gives_null_energy_and_peak(calc):
    nulls = pl.Series([None, None, None], dtype=pl.Float64)
    results = calc.calculate(
        _episode(total_power_w=nulls, battery_soc_percent=nulls), {}
    )
    assert results["energy_total_wh"] is None
    assert results["power_peak_w"] is None
    assert results["battery_soc_final"] is None


def test_leading_power_nulls_are_back_filled(calc):
    results = calc.calculate(
        _episode(total_power_w=pl.Series([None, 36.0, 72.0], dtype=pl.Float64)), {}
    )
    assert results["energy_total_wh"] == pytest.approx((36.0 + 72.0) / 3600.0)
    assert results["power_peak_w"] == pytest.approx(72.0)
    assert results["energy_static_wh"] is None


def test_total_is_the_integral_of_power_not_the_mean_times_duration(calc):
    results = calc.calculate(_episode(total_power_w=[0.0, 0.0, 7200.0]), {})
    # 0, 0, then 7200 W over one-second steps is 7200 J, i.e. 2 Wh. Mean power times duration would give 1.33 Wh.
    assert results["energy_total_wh"] == pytest.approx(2.0)
    assert results["energy_total_wh"] == pytest.approx(2.0)


def test_a_disagreeing_device_counter_does_not_change_the_total(calc):
    results = calc.calculate(
        _episode(
            total_power_w=[0.0, 3600.0, 3600.0],
            total_energy_consumed_wh=[1.0, 1.5, 2.25],
        ),
        {},
    )
    assert results["energy_total_wh"] == pytest.approx(2.0)


def test_the_integral_agrees_with_a_consistent_device_counter(calc):
    counter_wh = [0.0, 1.0, 2.0]
    results = calc.calculate(
        _episode(
            total_power_w=[0.0, 3600.0, 3600.0], total_energy_consumed_wh=counter_wh
        ),
        {},
    )
    counter_difference_wh = (
        counter_wh[-1] - counter_wh[0]
    )  # cross-check only, never published
    assert results["energy_total_wh"] == pytest.approx(counter_difference_wh)


def test_mean_power_is_energy_over_the_episode_span(calc):
    results = calc.calculate(
        _episode(total_power_w=[36.0, 36.0, 36.0]), {"time_diff": 2 * SEC}
    )
    # 36 W for 2 s is 72 J, i.e. 0.02 Wh over 2 s.
    assert results["mean_power_w"] == pytest.approx(36.0)


def test_without_episode_data_every_output_is_none(calc):
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    results = calc.calculate(empty, {})
    assert set(results) == set(calc.output_keys())
    assert all(value is None for value in results.values())


def test_cost_of_transport_is_energy_over_weight_times_distance(calc):
    results = calc.calculate(
        _episode(total_power_w=[0.0, 3600.0]), {"path_length_m": 10.0}
    )
    assert results["specific_cost_of_transport"] == pytest.approx(
        3600.0 / (MASS * GRAVITY * 10.0), rel=1e-9
    )


def test_cost_of_transport_is_unreported_without_a_declared_mass():
    undeclared = EnergyCalculator(RobotParams(robot_radius=0.25))
    results = undeclared.calculate(
        _episode(total_power_w=[0.0, 3600.0]), {"path_length_m": 10.0}
    )
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] == pytest.approx(0.1)


def test_energy_per_meter_divides_by_the_path(calc):
    results = calc.calculate(
        _episode(total_power_w=[0.0, 9000.0]), {"path_length_m": 10.0}
    )
    assert results["energy_per_meter"] == pytest.approx(0.25)


def test_negligible_path_length_is_rejected(calc):
    results = calc.calculate(
        _episode(total_power_w=[0.0, 3600.0]), {"path_length_m": 0.05}
    )
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] is None


def test_peak_to_mean_power_ratio_is_median_filtered(calc):
    results = calc.calculate(_episode(total_power_w=[10.0, 10.0, 10.0, 30.0]), {})
    # The median-filtered series [10, 10, 10, 20] has peak 20 over mean 12.5.
    assert results["peak_to_mean_power_ratio"] == pytest.approx(20.0 / 12.5)


def test_flat_power_gives_unit_peak_to_mean_ratio(calc):
    results = calc.calculate(_episode(total_power_w=[12.0] * 5), {})
    assert results["peak_to_mean_power_ratio"] == pytest.approx(1.0)


def test_zero_power_leaves_the_peak_to_mean_ratio_undefined(calc):
    results = calc.calculate(_episode(total_power_w=[0.0, 0.0, 0.0]), {})
    assert results["peak_to_mean_power_ratio"] is None


def test_standstill_penalty_only_bills_stationary_samples(calc):
    results = calc.calculate(
        _episode(
            total_power_w=[10.0, 10.0, 50.0, 50.0], vel_linear=[0.0, 0.0, 1.0, 1.0]
        ),
        {},
    )
    # One stationary second at 10 W; the moving samples are not charged.
    assert results["standstill_energy_penalty_wh"] == pytest.approx(10.0 / 3600.0)


def test_short_standstill_blocks_are_ignored(calc):
    episode = _episode_at(
        [0, 100_000_000, 200_000_000, 300_000_000],
        total_power_w=[10.0, 10.0, 10.0, 10.0],
        vel_linear=[1.0, 0.0, 1.0, 1.0],
    )
    assert calc.calculate(episode, {})["standstill_energy_penalty_wh"] == pytest.approx(
        0.0
    )


def test_a_robot_that_never_moves_is_billed_for_the_whole_run(calc):
    results = calc.calculate(
        _episode(total_power_w=[20.0, 20.0, 20.0], vel_linear=[0.0, 0.0, 0.0]), {}
    )
    assert results["standstill_energy_penalty_wh"] == pytest.approx(40.0 / 3600.0)


def test_kinetic_energy_demand_stays_unknown_until_a_yaw_rate_is_recorded(calc):
    results = calc.calculate(
        _episode(
            total_power_w=[10.0, 10.0, 10.0, 10.0], vel_linear=[0.0, 1.0, 1.0, 0.0]
        ),
        {},
    )
    assert results["kinetic_energy_demand_j"] is None


def test_series_table_is_flat_one_row_per_sample(calc):
    power = pl.Series([None, 36.0, 72.0], dtype=pl.Float64)
    table = calc.series_table(
        _episode(total_power_w=power, vel_linear=[0.1, 0.2, 0.3]),
        run_id="run",
        robot_id="robot",
    )
    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 3,
            "episode_id": [1] * 3,
            "robot_id": ["robot"] * 3,
            "t_ns": [0, SEC, 2 * SEC],
            "time_s": [0.0, 1.0, 2.0],
            "power_total_w": [36.0, 36.0, 72.0],
            "power_static_w": [None, None, None],
            "power_mechanical_w": [None, None, None],
            "power_thermal_w": [None, None, None],
            "battery_soc_pct": [None, None, None],
            "velocity_linear_mps": [0.1, 0.2, 0.3],
        },
        schema={
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "time_s": pl.Float64,
            "power_total_w": pl.Float64,
            "power_static_w": pl.Float64,
            "power_mechanical_w": pl.Float64,
            "power_thermal_w": pl.Float64,
            "battery_soc_pct": pl.Float64,
            "velocity_linear_mps": pl.Float64,
        },
    )
    assert_frame_equal(table, expected, check_dtypes=True, rel_tol=1e-12)


def test_series_table_without_data_keeps_the_schema(calc):
    empty = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    assert len(calc.series_table(empty)) == 0
    assert calc.series_table(empty).schema == pl.Schema(
        {
            "run_id": pl.String,
            "episode_id": pl.Int64,
            "robot_id": pl.String,
            "t_ns": pl.Int64,
            "time_s": pl.Float64,
            "power_total_w": pl.Float64,
            "power_static_w": pl.Float64,
            "power_mechanical_w": pl.Float64,
            "power_thermal_w": pl.Float64,
            "battery_soc_pct": pl.Float64,
            "velocity_linear_mps": pl.Float64,
        }
    )


# --- ENCOUNTER scope ---------------------------------------------------------


def _encounter_episode(
    power_w: float = 360.0, track: bool = True
) -> AlignedEpisodeBundle:
    times = [0, SEC, 2 * SEC]
    topics = {
        "robot_power": pl.DataFrame({"t_ns": times, "total_power_w": [power_w] * 3})
    }
    if track:
        topics["robot_tracks"] = pl.DataFrame(
            {
                "t_ns": times,
                "x": [0.0, 1.0, 2.0],
                "y": [0.0] * 3,
                "speed_mps": [1.0] * 3,
            }
        )
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": times}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[2.0, 0.0],
        topics=topics,
    )


def _encounter_slice(
    window: AlignedEpisodeBundle, full: AlignedEpisodeBundle
) -> ScopeSlice:
    return ScopeSlice(
        scope=MetricScope.ENCOUNTER,
        slice_id="enc-001",
        parent_id=None,
        t_start_ns=0,
        t_end_ns=2 * SEC,
        identity={},
        window_bundle=dataclasses.replace(window, data=pl.DataFrame()),
        full_episode=full,
    )


def test_encounter_scope_emits_only_its_own_outputs(calc):
    episode = _encounter_episode(power_w=36.0)
    window = _encounter_slice(episode, episode)
    result = calc.calculate_for_scope(window.window_bundle, {}, window)
    assert set(calc.output_keys_for_scope(MetricScope.ENCOUNTER)) <= set(result)
    assert result["energy_total_wh"] == pytest.approx(36.0 * 2.0 / 3600.0)
    assert result["power_peak_w"] == pytest.approx(36.0)
    assert result["energy_static_wh"] is None


def test_encounter_scope_divides_by_the_path_it_is_given(calc):
    episode = _encounter_episode()
    window = _encounter_slice(episode, episode)
    result = calc.calculate_for_scope(
        window.window_bundle, {"path_length_m": 2.0}, window
    )
    assert result["energy_per_meter"] == pytest.approx(0.2 / 2.0)
    assert result["specific_cost_of_transport"] == pytest.approx(
        0.2 * 3600.0 / (MASS * GRAVITY * 2.0)
    )
    assert result["peak_to_mean_power_ratio"] == pytest.approx(1.0)


def test_encounter_scope_honours_an_explicit_path_prior(calc):
    episode = _encounter_episode()
    window = _encounter_slice(episode, episode)
    result = calc.calculate_for_scope(
        window.window_bundle, {"path_length_m": 10.0}, window
    )
    assert result["energy_per_meter"] == pytest.approx(0.2 / 10.0)


def test_encounter_surge_ratio_compares_the_window_with_the_whole_episode(calc):
    window_episode = _encounter_episode(power_w=360.0)
    full_episode = _encounter_episode(power_w=120.0)
    window = _encounter_slice(window_episode, full_episode)
    result = calc.calculate_for_scope(
        window.window_bundle, {"mean_power_w": 120.0}, window
    )
    assert result["energy_surge_ratio"] == pytest.approx(3.0)


def test_encounter_surge_ratio_prefers_the_robot_scope_prior(calc):
    """The ROBOT-scope mean power is the one definition; a differing episode mean is not recomputed."""
    window_episode = _encounter_episode(power_w=360.0)
    full_episode = _encounter_episode(power_w=60.0)
    window = _encounter_slice(window_episode, full_episode)
    result = calc.calculate_for_scope(
        window.window_bundle, {"mean_power_w": 120.0}, window
    )
    assert result["energy_surge_ratio"] == pytest.approx(3.0)


def test_a_power_gap_longer_than_the_fill_bound_leaves_energy_undefined(calc):
    """Nine missing samples are not invented from the last value: the integral is NULL, not a guess."""
    times = [0, SEC, 2 * SEC, 3 * SEC]
    gaps = pl.Series([10.0, None, None, 10.0], dtype=pl.Float64)
    results = calc.calculate(_episode_at(times, total_power_w=gaps), {})
    assert results["energy_total_wh"] is None
    assert results["power_peak_w"] == pytest.approx(10.0)


def test_a_short_power_gap_is_filled_from_the_bracketing_sample(calc):
    """A gap of at most the fill bound is filled; its duration is charged at the measured power."""
    times = [0, 100_000_000, 200_000_000]
    gaps = pl.Series([10.0, None, 10.0], dtype=pl.Float64)
    results = calc.calculate(_episode_at(times, total_power_w=gaps), {})
    assert results["energy_total_wh"] == pytest.approx(2.0 / 3600.0)


def test_mechanical_energy_reads_the_aligned_mechanical_power_column(calc):
    power_w = [30.0, 30.0, 30.0]
    episode = _episode(
        total_power_w=power_w,
        total_mech_power_w=power_w,
        total_thermal_power_w=[0.0] * 3,
        static_power_w=[0.0] * 3,
        battery_soc_percent=[90.0] * 3,
        vel_linear=[0.0] * 3,
        vel_angular=[0.0] * 3,
    )
    results = calc.calculate(episode, {})
    assert results["energy_mechanical_wh"] is not None
    assert results["energy_mechanical_wh"] == pytest.approx(results["energy_total_wh"])


def test_kinetic_energy_demand_uses_the_recorded_yaw_rate(calc):
    episode = _episode(
        total_power_w=[10.0] * 3,
        total_mech_power_w=[10.0] * 3,
        vel_linear=[0.0] * 3,
        vel_angular=[0.0, 1.0, 0.0],
    )
    results = calc.calculate(episode, {})
    spin_up_j = 0.5 * calc.robot_params.mass * calc.robot_params.robot_radius**2 * 0.5
    assert results["kinetic_energy_demand_j"] == pytest.approx(2 * spin_up_j)
