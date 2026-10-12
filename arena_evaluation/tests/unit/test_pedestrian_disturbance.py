import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

S = 1_000_000_000
ALL_NONE = {"ped_path_deflection_m": None, "ped_velocity_delay_ratio": None, "ped_round_trips_completed": None}
deflection = PedestrianDisturbanceCalculator._compute_trajectory_deflection


def _frame(rows: list, seconds: list[float] | None = None) -> pl.DataFrame:
    columns: dict = {"peds_positions": pl.Series(rows, dtype=pl.List(pl.Float64))}
    if seconds is not None:
        columns["time_ns"] = pl.Series([int(s * S) for s in seconds], dtype=pl.Int64)
    return pl.DataFrame(columns)


def _calc(data: pl.DataFrame | None) -> dict:
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])
    return PedestrianDisturbanceCalculator(RobotParams()).calculate(episode, {})


def _walk(*points: tuple[float, float]) -> list[list[float]]:
    """One single-pedestrian x,y,z row per point."""
    return [[x, y, 0.0] for x, y in points]


def test_output_keys():
    assert PedestrianDisturbanceCalculator.output_keys() == ["ped_path_deflection_m", "ped_velocity_delay_ratio", "ped_round_trips_completed"]


def test_deflection_of_empty_input_is_zero():
    assert deflection(np.empty((0, 2)), np.array([[0.0, 0.0]])) == 0.0
    assert deflection(np.array([[0.0, 0.0]]), np.empty((0, 2))) == 0.0


def test_deflection_of_all_nan_rows_is_zero():
    assert deflection(np.array([[np.nan, 0.0], [1.0, np.nan]]), np.array([[0.0, 0.0]])) == 0.0
    assert deflection(np.array([[0.0, 0.0]]), np.array([[np.nan, np.nan]])) == 0.0


def test_deflection_is_mean_nearest_reference_distance_ignoring_nan_rows():
    actual = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 0.0], [1.0, 3.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert deflection(actual, reference) == pytest.approx((1.0 + 1.0 + 3.0) / 3.0, abs=1e-12)


def test_deflection_ignores_z():
    assert deflection(np.array([[0.0, 0.0, 5.0]]), np.array([[3.0, 4.0, -2.0]])) == pytest.approx(5.0, abs=1e-12)


def test_deflection_with_infinite_reference_falls_back_to_dense_distances():
    actual = np.array([[0.0, 2.0], [4.0, 0.0]])
    reference = np.array([[0.0, 0.0], [np.inf, 0.0], [4.0, 1.0]])
    assert deflection(actual, reference) == pytest.approx((2.0 + 1.0) / 2.0, abs=1e-12)


def test_missing_frame_yields_none():
    assert _calc(None) == ALL_NONE


def test_frame_without_peds_column_yields_none():
    assert _calc(pl.DataFrame({"time_ns": [0, S]})) == ALL_NONE


def test_empty_frame_yields_none():
    assert _calc(_frame([], [])) == ALL_NONE


def test_rows_without_valid_points_yield_zeros():
    rows = [None, [], [math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0], [0.0, -math.inf, 0.0], [2e6, 0.0, 0.0], [0.0, 1e6, 0.0]]
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5, 6])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0}


def test_agents_under_five_samples_count_only_toward_speed():
    rows = _walk((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0))
    assert _calc(_frame(rows, [0, 2, 4, 6])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": round(1.0 - (3.0 * math.sqrt(2.0) / 6.0) / 1.2, 3),
        "ped_round_trips_completed": 0.0,
    }


def test_back_and_forth_walk_counts_round_trips():
    rows = _walk((0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (0.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": round(1.0 - 1.0 / 1.2, 3),
        "ped_round_trips_completed": 2.0,
    }


def test_straight_walk_at_half_speed_has_no_deflection_and_no_round_trip():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": 0.583,
        "ped_round_trips_completed": 0.0,
    }


def test_walk_faster_than_desired_speed_clamps_delay_to_zero():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 0.5, 1, 1.5, 2]))["ped_velocity_delay_ratio"] == 0.0


def test_deflection_and_delay_average_over_all_agents():
    detour = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0)]
    rows = [[x, y, 0.0, 10.0, 10.0, 0.0] for x, y in detour]
    result = _calc(_frame(rows, [0, 1, 2, 3, 4]))
    detour_length = 2.0 + 2.0 * math.sqrt(2.0)
    assert result["ped_path_deflection_m"] == round(0.6 / 2.0, 3)
    assert result["ped_velocity_delay_ratio"] == round(1.0 - (detour_length / 2.0 / 4.0) / 1.2, 3)


def test_stationary_pedestrians_report_zero_delay():
    rows = [[1.0, 1.0, 0.0]] * 5
    assert _calc(_frame(rows, [0, 1, 2, 3, 4])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_delay_is_zero_without_timestamps():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows))["ped_velocity_delay_ratio"] == 0.0


def test_delay_is_zero_for_single_sample():
    assert _calc(_frame([[1.0, 1.0, 0.0]], [0])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_delay_is_zero_for_zero_duration():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 0, 0, 0, 0]))["ped_velocity_delay_ratio"] == 0.0


def test_delay_is_zero_when_every_agent_has_one_sample():
    assert _calc(_frame([[1.0, 1.0, 0.0], []], [0, 1]))["ped_velocity_delay_ratio"] == 0.0


def test_invalid_sample_does_not_shift_pedestrian_identity():
    rows = [[0.0, 0.0, 0.0, 10.0, 0.0, 0.0] for _ in range(6)]
    rows[2] = [math.nan, math.nan, 0.0, 10.0, 0.0, 0.0]
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_one_way_detour_completes_no_round_trip():
    rows = _walk((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4]))["ped_round_trips_completed"] == 0.0


def test_return_short_of_start_completes_no_round_trip():
    rows = _walk((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (3.0, 0.0), (2.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4]))["ped_round_trips_completed"] == 0.0


def test_round_trip_counts_return_near_start_after_reaching_far_end():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (4.0, 0.0), (3.9, 0.0), (0.9, 0.0), (0.5, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5, 6]))["ped_round_trips_completed"] == 1.0
