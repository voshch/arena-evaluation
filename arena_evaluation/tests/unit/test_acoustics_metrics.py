import math

import polars as pl
import pytest

from arena_evaluation.processing.metrics.ecological.acoustics import AcousticsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _episode(levels_dba, dt_s=1.0):
    times = [int(i * dt_s * SEC) for i in range(len(levels_dba))]
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"time_ns": times}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        topics={"acoustics": pl.DataFrame({"time_ns": times, "total_level_af_dba": levels_dba})},
    )


@pytest.fixture
def calc():
    return AcousticsCalculator(RobotParams(robot_radius=0.25))


def test_without_the_acoustics_topic_all_outputs_are_none(calc):
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame(), start_pos=[], goal_pos=[])
    results = calc.calculate(episode, {})
    assert set(results) == set(calc.output_keys())
    assert all(v is None for v in results.values())


def test_single_sample_is_rejected(calc):
    results = calc.calculate(_episode([60.0]), {})
    assert results["l_aeq_t"] is None


def test_constant_level_gives_that_level_as_laeq(calc):
    results = calc.calculate(_episode([60.0] * 4), {})
    assert results["l_aeq_t"] == pytest.approx(60.0, abs=1e-9)
    assert results["l_a10"] == pytest.approx(60.0)
    assert results["l_a50"] == pytest.approx(60.0)
    assert results["l_a90"] == pytest.approx(60.0)


def test_laeq_is_energy_averaged_not_arithmetic(calc):
    """Half the time at 60 dBA and half at 70 dBA is louder than the mean of 65."""
    results = calc.calculate(_episode([60.0, 60.0, 70.0, 70.0]), {})
    expected = 10.0 * math.log10((10**6.0 + 10**6.0 + 10**7.0 + 10**7.0) / 4.0)
    assert results["l_aeq_t"] == pytest.approx(expected, abs=1e-9)
    assert results["l_aeq_t"] > 65.0


def test_statistical_levels_follow_the_l_an_convention(calc):
    """L_A10 is the loud tail, L_A90 the quiet one."""
    results = calc.calculate(_episode([40.0, 50.0, 60.0, 70.0, 80.0]), {})
    assert results["l_a10"] > results["l_a50"] > results["l_a90"]
    assert results["l_a50"] == pytest.approx(60.0)


def test_startle_factor_is_the_gated_rise_rate(calc):
    results = calc.calculate(_episode([50.0, 50.0, 80.0, 80.0, 80.0]), {})
    assert results["acoustic_startle_factor"] == pytest.approx(30.0)


def test_rises_below_the_level_gate_do_not_count(calc):
    """The same 30 dBA rise below the gate is not a transient."""
    results = calc.calculate(_episode([20.0, 20.0, 50.0, 50.0, 50.0]), {})
    assert results["acoustic_startle_factor"] == 0.0


def test_falling_level_has_no_startle(calc):
    results = calc.calculate(_episode([80.0, 80.0, 50.0, 50.0, 50.0]), {})
    assert results["acoustic_startle_factor"] == 0.0


def test_surge_index_is_zero_for_a_steady_source(calc):
    results = calc.calculate(_episode([60.0] * 6), {})
    assert results["acoustic_surge_index"] == pytest.approx(0.0, abs=1e-9)


def test_surge_index_grows_with_level_swings(calc):
    steady = calc.calculate(_episode([60.0] * 8), {})
    swinging = calc.calculate(_episode([40.0, 80.0] * 4), {})
    assert swinging["acoustic_surge_index"] > steady["acoustic_surge_index"]


def test_acoustic_cost_is_energy_per_meter(calc):
    levels = [60.0] * 4
    results = calc.calculate(_episode(levels), {"path_length": 10.0})
    # 4 samples of 10^6 for 1 s each, over 10 m.
    assert results["acoustic_cost"] == pytest.approx(4 * 10**6.0 / 10.0, rel=1e-9)


def test_acoustic_cost_needs_a_path_length(calc):
    assert calc.calculate(_episode([60.0] * 4), {})["acoustic_cost"] is None
    assert calc.calculate(_episode([60.0] * 4), {"path_length": 0.01})["acoustic_cost"] is None


def test_null_levels_are_forward_filled(calc):
    results = calc.calculate(_episode([60.0, None, 60.0, 60.0]), {})
    assert results["l_aeq_t"] == pytest.approx(60.0, abs=1e-9)
