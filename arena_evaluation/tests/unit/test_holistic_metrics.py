import polars as pl
import pytest

from arena_evaluation.processing.metrics.holistic.holistic_metrics import HolisticMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

S = 1_000_000_000


def _calc(prior: dict, data: pl.DataFrame | None = None) -> dict:
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame() if data is None else data, start_pos=[], goal_pos=[])
    return HolisticMetricsCalculator(RobotParams()).calculate(episode, prior)


def _times(*seconds: float) -> pl.DataFrame:
    return pl.DataFrame({"time_ns": [int(s * S) for s in seconds]})


def test_output_keys():
    assert HolisticMetricsCalculator.output_keys() == ["aeps_linear_s", "e_cot"]


def test_no_inputs_yield_none():
    assert _calc({}) == {"aeps_linear_s": None, "e_cot": None}


def test_aeps_integrates_flat_exposure_over_intruded_frames():
    prior = {"timeseries_min_ped_clearance": [2.0, 1.0, 0.5, None], "timeseries_acoustic_exposure_dba": [60.0, 70.0, 80.0, 90.0]}
    result = _calc(prior, _times(0, 1, 3, 4))
    assert result["aeps_linear_s"] == pytest.approx(1e7 * 1.0 + 1e8 * 2.0, rel=1e-12)
    assert result["e_cot"] is None


def test_aeps_first_frame_carries_zero_dt():
    prior = {"timeseries_min_ped_clearance": [0.5, 2.0], "timeseries_acoustic_exposure_dba": [90.0, 90.0]}
    assert _calc(prior, _times(0, 1))["aeps_linear_s"] == 0.0


def test_aeps_nested_exposure_takes_loudest_pedestrian_and_45_dba_for_empty_frames():
    prior = {"timeseries_min_ped_clearance": [1.0, 1.0, 1.0], "timeseries_acoustic_exposure_dba": [[50.0, 70.0], [], [60.0]]}
    expected = 1e7 * 0.0 + 10.0**4.5 * 1.0 + 1e6 * 1.0
    assert _calc(prior, _times(0, 1, 2))["aeps_linear_s"] == pytest.approx(expected, rel=1e-12)


def test_aeps_falls_back_to_total_level_timeseries_when_exposure_is_empty():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5], "timeseries_acoustic_exposure_dba": [], "timeseries_total_level_dba": [80.0, 80.0]}
    assert _calc(prior, _times(0, 1))["aeps_linear_s"] == pytest.approx(1e8, rel=1e-12)


def test_aeps_falls_back_to_total_level_timeseries_when_exposure_is_missing():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5], "timeseries_total_level_dba": [60.0, 70.0]}
    assert _calc(prior, _times(0, 3))["aeps_linear_s"] == pytest.approx(1e7 * 3.0, rel=1e-12)


def test_aeps_falls_back_to_aligned_frame_level_column():
    data = pl.DataFrame({"time_ns": [0, S, 2 * S], "total_level_af_dba": [50.0, 60.0, 70.0]})
    prior = {"timeseries_min_ped_clearance": [0.5, 2.0, 0.5]}
    assert _calc(prior, data)["aeps_linear_s"] == pytest.approx(1e7, rel=1e-12)


def test_aeps_defaults_to_45_dba_without_any_level_source():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5]}
    assert _calc(prior, _times(0, 2))["aeps_linear_s"] == pytest.approx(10.0**4.5 * 2.0, rel=1e-12)


def test_aeps_truncates_to_shorter_series():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5, 0.5], "timeseries_acoustic_exposure_dba": [60.0, 60.0]}
    assert _calc(prior, _times(0, 1, 2))["aeps_linear_s"] == pytest.approx(1e6, rel=1e-12)


def test_aeps_clamps_backwards_timestamps_to_zero_dt():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5, 0.5], "timeseries_acoustic_exposure_dba": [60.0, 60.0, 60.0]}
    assert _calc(prior, _times(0, 2, 1))["aeps_linear_s"] == pytest.approx(2e6, rel=1e-12)


def test_aeps_is_zero_without_intrusions():
    prior = {"timeseries_min_ped_clearance": [1.2, 3.0, None], "timeseries_acoustic_exposure_dba": [90.0, 90.0, 90.0]}
    assert _calc(prior, _times(0, 1, 2))["aeps_linear_s"] == 0.0


def test_aeps_is_zero_without_timestamps():
    prior = {"timeseries_min_ped_clearance": [0.5, 0.5], "timeseries_acoustic_exposure_dba": [90.0, 90.0]}
    assert _calc(prior, pl.DataFrame({"x": [0.0, 1.0]}))["aeps_linear_s"] == 0.0


def test_aeps_is_zero_when_aligned_frame_is_missing():
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])
    result = HolisticMetricsCalculator(RobotParams()).calculate(episode, {"timeseries_min_ped_clearance": [0.5]})
    assert result == {"aeps_linear_s": 0.0, "e_cot": None}


def test_aeps_is_none_for_empty_clearance_series():
    assert _calc({"timeseries_min_ped_clearance": []}, _times())["aeps_linear_s"] is None


@pytest.mark.parametrize("cot", [None, 0.0, -1.0])
def test_e_cot_is_none_without_positive_cost_of_transport(cot: float | None):
    assert _calc({"specific_cost_of_transport": cot, "collision_amount": 3, "time_to_goal": 10.0})["e_cot"] is None


def test_e_cot_penalizes_collisions_and_intrusion_rate():
    prior = {"specific_cost_of_transport": 1.5, "collision_amount": 2, "personal_space_intrusion_integral": 4.0, "time_to_goal": 10.0}
    assert _calc(prior)["e_cot"] == pytest.approx(1.5 * (1.0 + 4.0 + 0.2), abs=1e-9)


def test_e_cot_treats_missing_penalty_inputs_as_zero():
    assert _calc({"specific_cost_of_transport": 3.0})["e_cot"] == 3.0


def test_e_cot_treats_none_penalty_inputs_as_zero():
    prior = {"specific_cost_of_transport": 2.0, "collision_amount": None, "personal_space_intrusion_integral": None, "time_to_goal": 10.0}
    assert _calc(prior)["e_cot"] == 2.0


@pytest.mark.parametrize("time_to_goal", [0.0, None, -5.0, 0.001])
def test_e_cot_floors_episode_time_at_10_ms(time_to_goal: float | None):
    prior = {"specific_cost_of_transport": 1.0, "personal_space_intrusion_integral": 0.02, "time_to_goal": time_to_goal}
    assert _calc(prior)["e_cot"] == pytest.approx(2.0, abs=1e-9)


def test_e_cot_truncates_fractional_collision_count():
    prior = {"specific_cost_of_transport": 1.0, "collision_amount": 1.9, "time_to_goal": 10.0}
    assert _calc(prior)["e_cot"] == 3.0


def test_e_cot_rounds_to_six_decimals():
    assert _calc({"specific_cost_of_transport": 1.0 / 3.0, "time_to_goal": 10.0})["e_cot"] == 0.333333
