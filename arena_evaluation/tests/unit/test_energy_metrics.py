import dataclasses

import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.ecological.energy import EnergyCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000
MASS = 18.431
GRAVITY = 9.81


def _episode_at(times_ns: list[int], **columns: list[float | None] | pl.Series) -> AlignedEpisodeBundle:
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
    return EnergyCalculator(make_robot_params(robot_radius=0.25, base_mass=MASS))


def test_leading_power_nulls_are_back_filled(calc):
    results = calc.calculate(_episode(total_power_w=pl.Series([None, 36.0, 72.0], dtype=pl.Float64)), {})
    assert results["energy_total_wh"] == pytest.approx((36.0 + 72.0) / 3600.0)
    assert results["power_peak_w"] == pytest.approx(72.0)
    assert results["energy_static_wh"] is None


def test_mean_power_is_energy_over_the_episode_span(calc):
    results = calc.calculate(_episode(total_power_w=[36.0, 36.0, 36.0]), {"time_diff": 2 * SEC})
    # 36 W for 2 s is 72 J, i.e. 0.02 Wh over 2 s.
    assert results["mean_power_w"] == pytest.approx(36.0)


def test_cost_of_transport_is_unreported_without_a_declared_mass():
    undeclared = EnergyCalculator(make_robot_params(robot_radius=0.25))
    results = undeclared.calculate(_episode(total_power_w=[0.0, 3600.0]), {"path_length_m": 10.0})
    assert results["specific_cost_of_transport"] is None
    assert results["energy_per_meter"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("power_w", "expected"),
    [
        # The median-filtered series [10, 10, 10, 20] has peak 20 over mean 12.5.
        ([10.0, 10.0, 10.0, 30.0], 20.0 / 12.5),
        ([12.0] * 5, 1.0),
        ([0.0, 0.0, 0.0], None),
    ],
    ids=["median-filtered", "flat-power", "zero-power"],
)
def test_peak_to_mean_power_ratio_is_median_filtered(calc, power_w: list[float], expected: float | None) -> None:
    ratio = calc.calculate(_episode(total_power_w=power_w), {})["peak_to_mean_power_ratio"]
    if expected is None:
        assert ratio is None
    else:
        assert ratio == pytest.approx(expected)


def test_short_standstill_blocks_are_ignored(calc):
    episode = _episode_at(
        [0, 100_000_000, 200_000_000, 300_000_000],
        total_power_w=[10.0, 10.0, 10.0, 10.0],
        speed_mps=[1.0, 0.0, 1.0, 1.0],
    )
    assert calc.calculate(episode, {})["standstill_energy_penalty_wh"] == pytest.approx(0.0)


def _encounter_episode(power_w: float = 360.0, track: bool = True) -> AlignedEpisodeBundle:
    times = [0, SEC, 2 * SEC]
    topics = {"robot_power": pl.DataFrame({"t_ns": times, "total_power_w": [power_w] * 3})}
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


def _encounter_slice(window: AlignedEpisodeBundle, full: AlignedEpisodeBundle) -> ScopeSlice:
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
    result = calc.calculate_for_scope(window.window_bundle, {"path_length_m": 2.0}, window)
    assert result["energy_per_meter"] == pytest.approx(0.2 / 2.0)
    assert result["specific_cost_of_transport"] == pytest.approx(0.2 * 3600.0 / (MASS * GRAVITY * 2.0))
    assert result["peak_to_mean_power_ratio"] == pytest.approx(1.0)


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
        speed_mps=[0.0] * 3,
        omega_z=[0.0] * 3,
    )
    results = calc.calculate(episode, {})
    assert results["energy_mechanical_wh"] is not None
    assert results["energy_mechanical_wh"] == pytest.approx(results["energy_total_wh"])
