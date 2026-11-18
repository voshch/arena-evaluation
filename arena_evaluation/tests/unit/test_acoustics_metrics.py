
import polars as pl
import pytest

from arena_evaluation.processing.metrics.ecological.acoustics import (
    ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2,
    AcousticsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _episode(levels_dba, dt_s=1.0):
    times = [int(i * dt_s * SEC) for i in range(len(levels_dba))]
    return AlignedEpisodeBundle(
        episode_id=1,
        data=pl.DataFrame({"t_ns": times}),
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[5.0, 0.0],
        topics={
            "robot_acoustics": pl.DataFrame(
                {"t_ns": times, "total_level_af_dba": levels_dba}
            )
        },
    )


@pytest.fixture
def calc():
    return AcousticsCalculator(make_robot_params(robot_radius=0.25))


def test_single_sample_is_rejected(calc):
    results = calc.calculate(_episode([60.0]), {})
    assert results["l_aeq_t"] is None


def test_surge_index_grows_with_level_swings(calc):
    steady = calc.calculate(_episode([60.0] * 8), {})
    swinging = calc.calculate(_episode([40.0, 80.0] * 4), {})
    assert swinging["acoustic_surge_index"] > steady["acoustic_surge_index"]


def test_acoustic_cost_is_energy_per_meter(calc):
    levels = [60.0] * 4
    results = calc.calculate(_episode(levels), {"path_length_m": 10.0})
    # 4 samples of p_0^2 * 10^6 for 1 s each, over 10 m.
    expected = ACOUSTIC_REFERENCE_PRESSURE_SQ_PA2 * 4 * 10**6.0 / 10.0
    assert results["acoustic_cost"] == pytest.approx(expected, rel=1e-9)


