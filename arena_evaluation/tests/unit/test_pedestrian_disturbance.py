import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from flat_peds import flat_peds
from robot_params_factory import make_robot_params

S = 1_000_000_000
ALL_NONE = {"ped_path_deflection_m": None, "ped_velocity_delay_ratio": None}
deflection = PedestrianDisturbanceCalculator._compute_trajectory_deflection


def _frame(rows: list, seconds: list[float] | None = None) -> pl.DataFrame:
    """Each row is one frame: its pedestrians as flat (x, y, z) triples, stamped at seconds[i] (or i)."""
    times = seconds if seconds is not None else list(range(len(rows)))
    frames = []
    for t, row in zip(times, rows, strict=True):
        values = row or []
        pedestrians = [{"ped_id": f"Pedestrian_{j}", "x": values[3 * j], "y": values[3 * j + 1]} for j in range(len(values) // 3)]
        frames.append((int(t * S), pedestrians))
    return flat_peds(frames)


def _calc(peds: pl.DataFrame | None) -> dict:
    """The PEDESTRIAN slice's bundle: one pedestrian's track, on its own frame times."""
    topics = {} if peds is None else {"pedestrian_tracks": peds}
    data = None if peds is None else pl.DataFrame({"t_ns": peds["t_ns"].unique().sort()})
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], peds=peds, topics=topics)
    return PedestrianDisturbanceCalculator(make_robot_params()).calculate(episode, {})


def _walk(*points: tuple[float, float]) -> list[list[float]]:
    """One single-pedestrian x,y,z row per point."""
    return [[x, y, 0.0] for x, y in points]


def test_output_keys():
    assert PedestrianDisturbanceCalculator.output_keys() == ["ped_path_deflection_m", "ped_velocity_delay_ratio"]


def test_deflection_of_empty_input_is_null():
    assert math.isnan(deflection(np.empty((0, 2)), np.array([[0.0, 0.0]])))
    assert math.isnan(deflection(np.array([[0.0, 0.0]]), np.empty((0, 2))))


def test_deflection_of_all_nan_rows_is_null():
    assert math.isnan(deflection(np.array([[np.nan, 0.0], [1.0, np.nan]]), np.array([[0.0, 0.0]])))
    assert math.isnan(deflection(np.array([[0.0, 0.0]]), np.array([[np.nan, np.nan]])))


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


def test_missing_pedestrian_table_yields_none():
    assert _calc(None) == ALL_NONE


def test_empty_frame_yields_none():
    assert _calc(_frame([], [])) == ALL_NONE


def test_rows_without_valid_points_yield_null_deflection_and_delay():
    rows = [None, [], [math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0], [0.0, -math.inf, 0.0], [2e6, 0.0, 0.0], [0.0, 1e6, 0.0]]
    result = _calc(_frame(rows, [0, 1, 2, 3, 4, 5, 6]))
    assert math.isnan(result["ped_path_deflection_m"])
    assert math.isnan(result["ped_velocity_delay_ratio"])


def test_fewer_than_five_samples_leave_the_deflection_null_but_count_toward_speed():
    rows = _walk((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0))
    assert _calc(_frame(rows, [0, 2, 4, 6])) == {
        "ped_path_deflection_m": None,
        "ped_velocity_delay_ratio": round(1.0 - (3.0 * math.sqrt(2.0) / 6.0) / 1.2, 3),
    }


def test_a_walk_that_returns_to_its_start_has_no_chord_and_no_deflection():
    rows = _walk((0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (0.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": None,
        "ped_velocity_delay_ratio": round(1.0 - 1.0 / 1.2, 3),
    }


def test_straight_walk_at_half_speed_has_no_deflection():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": 0.583,
    }


def test_walk_faster_than_desired_speed_clamps_delay_to_zero():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 0.5, 1, 1.5, 2]))["ped_velocity_delay_ratio"] == 0.0


def test_a_detour_has_its_mean_lateral_deviation_and_its_delay():
    detour = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0)]
    result = _calc(_frame(_walk(*detour), [0, 1, 2, 3, 4]))
    detour_length = 2.0 + 2.0 * math.sqrt(2.0)
    assert result["ped_path_deflection_m"] == pytest.approx(0.6, abs=1e-3)
    assert detour_length > 0.0
    assert result["ped_velocity_delay_ratio"] == 0.0


def test_stationary_pedestrian_has_no_deflection_and_null_delay():
    rows = [[1.0, 1.0, 0.0]] * 5
    result = _calc(_frame(rows, [0, 1, 2, 3, 4]))
    assert result["ped_path_deflection_m"] is None
    assert math.isnan(result["ped_velocity_delay_ratio"])


def test_delay_is_null_for_single_sample():
    result = _calc(_frame([[1.0, 1.0, 0.0]], [0]))
    assert result["ped_path_deflection_m"] is None
    assert math.isnan(result["ped_velocity_delay_ratio"])


def test_delay_is_null_for_zero_duration():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert math.isnan(_calc(_frame(rows, [0, 0, 0, 0, 0]))["ped_velocity_delay_ratio"])
