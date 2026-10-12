import math

import polars as pl
import pytest

from arena_evaluation.processing.metrics.performance.pedestrian_path_metrics import PedestrianPathMetricsCalculator
from arena_evaluation.processing.metrics.social.mutual_accommodation import MutualAccommodationCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

M = MutualAccommodationCalculator
ALL_NONE = {"mar": None, "pfi": None, "robot_path_deviation_m": None, "relative_throughput": None, "kcsc_wh": None}
PED_SHIFTED = [[[0.0, 1.0], [1.0, 1.0]]]
PED_REFERENCE = [[[0.0, 0.0], [1.0, 0.0]]]


def _reconcile(dynamic: dict, ref_robot: dict | None = None, ref_ped: dict | None = None) -> dict:
    return M.reconcile_stage_references(dynamic_row=dynamic, ref_robot_row=ref_robot, ref_ped_row=ref_ped)


def test_output_keys():
    assert M.output_keys() == ["mar", "pfi", "robot_path_deviation_m", "relative_throughput", "kcsc_wh"]


def test_calculate_defers_everything_to_reconciliation():
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame({"x": [1.0]}), start_pos=[], goal_pos=[])
    assert M(RobotParams()).calculate(episode, {"path_length": 3.0}) == ALL_NONE


@pytest.mark.parametrize(("pts", "ref"), [(None, [[0.0, 0.0], [1.0, 0.0]]), ([[0.0, 1.0]], None), ([], [[0.0, 0.0], [1.0, 0.0]]), ([[0.0, 1.0]], [[0.0, 0.0]])])
def test_orthogonal_deviation_of_degenerate_input_is_zero(pts: list | None, ref: list | None):
    assert M.compute_orthogonal_deviation(pts, ref) == (0.0, 0.0)


def test_orthogonal_deviation_projects_onto_segment():
    assert M.compute_orthogonal_deviation([[5.0, 1.0], [2.0, -0.5]], [[0.0, 0.0], [10.0, 0.0]]) == pytest.approx((1.0, 0.75), abs=1e-12)


@pytest.mark.parametrize("point", [[-3.0, 4.0], [13.0, 4.0]])
def test_orthogonal_deviation_clamps_to_segment_endpoints(point: list[float]):
    assert M.compute_orthogonal_deviation([point], [[0.0, 0.0], [10.0, 0.0]]) == pytest.approx((5.0, 5.0), abs=1e-12)


def test_orthogonal_deviation_of_zero_length_segment_is_point_distance():
    assert M.compute_orthogonal_deviation([[3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]) == pytest.approx((5.0, 5.0), abs=1e-12)


def test_orthogonal_deviation_takes_nearest_segment():
    assert M.compute_orthogonal_deviation([[5.0, 2.0]], [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]]) == pytest.approx((2.0, 2.0), abs=1e-12)
    assert M.compute_orthogonal_deviation([[1.0, 1.0]], [[0.0, 0.0], [2.0, 0.0], [20.0, 10.0]]) == pytest.approx((1.0, 1.0), abs=1e-12)


def test_orthogonal_deviation_unwraps_batched_input_and_ignores_z():
    assert M.compute_orthogonal_deviation([[[5.0, 1.0, 7.0]]], [[[0.0, 0.0, 0.0], [10.0, 0.0, 3.0]]]) == pytest.approx((1.0, 1.0), abs=1e-12)


@pytest.mark.parametrize(("actual", "ref"), [([], PED_REFERENCE), (PED_SHIFTED, []), (None, PED_REFERENCE)])
def test_pfi_without_paths_is_none(actual: list | None, ref: list):
    assert M.compute_pfi_dtw(actual, ref) is None


def test_pfi_of_identical_paths_is_zero():
    assert M.compute_pfi_dtw(PED_REFERENCE, PED_REFERENCE) == 0.0


def test_pfi_of_parallel_offset_is_offset():
    assert M.compute_pfi_dtw(PED_SHIFTED, PED_REFERENCE) == pytest.approx(1.0, abs=1e-12)


def test_pfi_normalizes_warped_cost_by_longer_path():
    actual = [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
    ref = [[[0.0, 1.0], [2.0, 1.0]]]
    assert M.compute_pfi_dtw(actual, ref) == pytest.approx((2.0 + math.sqrt(2.0)) / 3.0, abs=1e-12)


def test_pfi_skips_unusable_pairs_and_unmatched_pedestrians():
    actual = [None, [[0.0, 0.0]], [0.0, 1.0, 2.0], [[0.0], [1.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]], [[9.0, 9.0], [9.0, 9.0]]]
    ref = [PED_REFERENCE[0], PED_REFERENCE[0], PED_REFERENCE[0], PED_REFERENCE[0], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [3.0, 2.0]]]
    assert M.compute_pfi_dtw(actual, ref) == pytest.approx((1.0 + 1.0) / 2.0, abs=1e-12)


def test_pfi_is_none_when_every_pair_is_unusable():
    assert M.compute_pfi_dtw([[[0.0, 0.0]], None], [PED_REFERENCE[0], PED_REFERENCE[0]]) is None


def test_pfi_skips_pairs_left_with_one_finite_sample():
    nan = [math.nan, math.nan]
    assert M.compute_pfi_dtw([[nan, nan, [0.0, 0.0]]], [PED_REFERENCE[0]]) is None
    assert M.compute_pfi_dtw([PED_REFERENCE[0]], [[nan, [0.0, 0.0], nan]]) is None


def test_pfi_drops_non_finite_samples_before_alignment():
    nan = [math.nan, math.nan]
    assert M.compute_pfi_dtw([[nan, *PED_SHIFTED[0]]], [[*PED_REFERENCE[0], nan]]) == pytest.approx(1.0, abs=1e-12)


def test_pfi_downsamples_long_actual_path_to_300_samples():
    actual = [[0.0, 0.0]] * 301
    actual[299] = [0.0, 5.0]
    assert M.compute_pfi_dtw([actual], [[[0.0, 0.0], [0.0, 0.0]]]) == 0.0


def test_pfi_downsamples_long_reference_path_to_300_samples():
    ref = [[0.0, 0.0]] * 301
    ref[299] = [0.0, 5.0]
    assert M.compute_pfi_dtw([[[0.0, 0.0], [0.0, 0.0]]], [ref]) == 0.0


def test_reconcile_without_references_is_all_none():
    assert _reconcile({"pedestrian_path": PED_SHIFTED, "path": [[0.0, 0.0], [1.0, 0.0]], "time_to_goal": 5.0}) == ALL_NONE


@pytest.mark.parametrize(("dynamic", "ref_ped"), [({}, {"pedestrian_path": PED_REFERENCE}), ({"pedestrian_path": PED_SHIFTED}, {"pedestrian_path": []}), ({"pedestrian_path": PED_SHIFTED}, {}), ({"pedestrian_path": []}, {"pedestrian_path": PED_REFERENCE})])
def test_reconcile_skips_flow_terms_without_both_ped_paths(dynamic: dict, ref_ped: dict):
    assert _reconcile(dynamic, None, ref_ped) == ALL_NONE


def test_reconcile_pedestrian_flow_reports_pfi_and_deflection():
    assert _reconcile({"pedestrian_path": PED_SHIFTED}, None, {"pedestrian_path": PED_REFERENCE}) == {**ALL_NONE, "pfi": 1.0, "ped_path_deflection_m": 1.0}


def test_reconcile_reports_deflection_when_pfi_is_undefined():
    assert _reconcile({"pedestrian_path": [[[0.0, 1.0]]]}, None, {"pedestrian_path": [[[0.0, 0.0]]]}) == {**ALL_NONE, "ped_path_deflection_m": 1.0}


def test_reconcile_deflection_pools_samples_and_skips_empty_paths():
    dynamic = {"pedestrian_path": [[], [[0.0, 1.0], [1.0, 1.0]], [[5.0, 3.0], [6.0, 3.0]]]}
    ref_ped = {"pedestrian_path": [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[5.0, 0.0], [6.0, 0.0]]]}
    assert _reconcile(dynamic, None, ref_ped) == {**ALL_NONE, "pfi": 2.0, "ped_path_deflection_m": 2.0}


def test_reconcile_omits_deflection_for_mixed_dimension_paths():
    dynamic = {"pedestrian_path": [[[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]}
    ref_ped = {"pedestrian_path": [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]}
    assert _reconcile(dynamic, None, ref_ped) == {**ALL_NONE, "pfi": 1.0}


def test_reconcile_omits_deflection_for_non_list_path_containers():
    dynamic = {"pedestrian_path": (((0.0, 1.0), (1.0, 1.0)),)}
    ref_ped = {"pedestrian_path": (((0.0, 0.0), (1.0, 0.0)),)}
    assert _reconcile(dynamic, None, ref_ped) == {**ALL_NONE, "pfi": 1.0}


def test_reconcile_omits_deflection_for_flat_paths():
    assert _reconcile({"pedestrian_path": [[1.0, 2.0]]}, None, {"pedestrian_path": [[3.0, 4.0]]}) == ALL_NONE


def test_reconcile_full_stage_comparison():
    dynamic = {"path": [[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]], "path_length": 10.2, "time_to_goal": 20.0, "energy_total_wh": 5.0, "pedestrian_path": [[[5.0, -0.5], [5.0, 0.0]]]}
    ref_robot = {"path": [[0.0, 0.0], [10.0, 0.0]], "path_length": 10.0, "time_to_goal": 10.0, "energy_total_wh": 3.0}
    ref_ped = {"pedestrian_path": [[[5.0, 0.0], [5.0, 0.0]]]}
    assert _reconcile(dynamic, ref_robot, ref_ped) == {
        "mar": 0.25,
        "pfi": 0.25,
        "robot_path_deviation_m": 1.0,
        "relative_throughput": 0.5,
        "kcsc_wh": 2.0,
        "ped_path_deflection_m": 0.25,
    }


def test_reconcile_mar_floors_robot_detour_at_5_cm():
    dynamic = {"path": [[0.0, 0.0], [5.0, 0.02], [10.0, 0.0]], "pedestrian_path": PED_SHIFTED}
    result = _reconcile(dynamic, {"path": [[0.0, 0.0], [10.0, 0.0]]}, {"pedestrian_path": PED_REFERENCE})
    assert result["robot_path_deviation_m"] == 0.02
    assert result["mar"] == 20.0


@pytest.mark.parametrize(("dynamic_length", "expected_mar"), [(12.0, 0.5), (9.0, 100.0), (10.0, 100.0)])
def test_reconcile_mar_falls_back_to_path_length_excess_without_lateral_detour(dynamic_length: float, expected_mar: float):
    path = [[0.0, 0.0], [10.0, 0.0]]
    dynamic = {"path": path, "path_length": dynamic_length, "pedestrian_path": PED_SHIFTED}
    result = _reconcile(dynamic, {"path": path, "path_length": 10.0}, {"pedestrian_path": PED_REFERENCE})
    assert result["robot_path_deviation_m"] == 0.0
    assert result["mar"] == expected_mar


@pytest.mark.parametrize(("dynamic", "ref_robot"), [({"path_length": 12.0}, {"path": [[0.0, 0.0], [10.0, 0.0]], "path_length": 10.0}), ({"path": [[0.0, 0.0], [10.0, 0.0]], "path_length": 12.0}, {"path_length": 10.0})])
def test_reconcile_mar_uses_path_length_when_a_robot_path_is_missing(dynamic: dict, ref_robot: dict):
    result = _reconcile({**dynamic, "pedestrian_path": PED_SHIFTED}, ref_robot, {"pedestrian_path": PED_REFERENCE})
    assert result == {**ALL_NONE, "pfi": 1.0, "ped_path_deflection_m": 1.0, "mar": 0.5}


def test_reconcile_ragged_robot_path_leaves_deviation_unset_and_uses_path_length():
    dynamic = {"path": [[0.0, 0.0], [1.0]], "path_length": 11.0, "pedestrian_path": PED_SHIFTED}
    result = _reconcile(dynamic, {"path": [[0.0, 0.0], [10.0, 0.0]], "path_length": 10.0}, {"pedestrian_path": PED_REFERENCE})
    assert result["robot_path_deviation_m"] is None
    assert result["mar"] == 1.0


def test_reconcile_mar_is_none_without_detour_or_path_lengths():
    path = [[0.0, 0.0], [10.0, 0.0]]
    result = _reconcile({"path": path, "pedestrian_path": PED_SHIFTED}, {"path": path, "path_length": 10.0}, {"pedestrian_path": PED_REFERENCE})
    assert result == {**ALL_NONE, "pfi": 1.0, "ped_path_deflection_m": 1.0, "robot_path_deviation_m": 0.0}


def test_reconcile_mar_is_none_without_pfi():
    result = _reconcile({"path": [[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]], "path_length": 12.0}, {"path": [[0.0, 0.0], [10.0, 0.0]], "path_length": 10.0})
    assert result == {**ALL_NONE, "robot_path_deviation_m": 1.0}


@pytest.mark.parametrize(("ref_time", "ep_time"), [(10.0, 0.0), (10.0, -1.0), (None, 20.0), (10.0, None)])
def test_reconcile_relative_throughput_needs_both_times_and_positive_episode_time(ref_time: float | None, ep_time: float | None):
    assert _reconcile({"time_to_goal": ep_time}, {"time_to_goal": ref_time})["relative_throughput"] is None


def test_reconcile_relative_throughput_rounds_to_four_decimals():
    assert _reconcile({"time_to_goal": 30.0}, {"time_to_goal": 10.0})["relative_throughput"] == 0.3333


@pytest.mark.parametrize(("ep_energy", "ref_energy", "expected"), [(2.0, 3.0, 0.0), (None, 3.0, None), (5.0, None, None), (3.5, 1.25, 2.25)])
def test_reconcile_kcsc_is_clamped_energy_excess(ep_energy: float | None, ref_energy: float | None, expected: float | None):
    assert _reconcile({"energy_total_wh": ep_energy}, {"energy_total_wh": ref_energy})["kcsc_wh"] == expected


def test_reconcile_empty_robot_reference_is_ignored():
    assert _reconcile({"time_to_goal": 20.0, "energy_total_wh": 5.0, "path": [[0.0, 0.0], [1.0, 1.0]]}, {}) == ALL_NONE


def test_reconcile_pfi_ignores_samples_before_first_pedestrian_message():
    calc = PedestrianPathMetricsCalculator(RobotParams())

    def ped_paths(rows: list) -> list:
        df = pl.DataFrame({"peds_positions": rows}, schema={"peds_positions": pl.List(pl.Float64)})
        return calc.calculate(AlignedEpisodeBundle(episode_id=1, data=df, start_pos=[], goal_pos=[]), {})["pedestrian_path"]

    dynamic = {"pedestrian_path": ped_paths([None, [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])}
    ref_ped = {"pedestrian_path": ped_paths([None, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    assert _reconcile(dynamic, None, ref_ped)["pfi"] == 1.0
