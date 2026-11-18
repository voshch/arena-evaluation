"""Path geometry: scalar aggregates in the metric row, per-sample path in the 1NF series table."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.base import MetricScope, ScopeSlice
from arena_evaluation.processing.metrics.performance.path_metrics import SERIES_SCHEMA, PathMetricsCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

SEC = 1_000_000_000


def _episode(poses: list[tuple[float, float, float]], *, odom_dx: float = 0.0, episode_id: int = 1) -> AlignedEpisodeBundle:
    """Episode frame with ground-truth poses and the wheel odometry of the same track shifted by ``odom_dx``."""
    xs = [pose[0] for pose in poses]
    ys = [pose[1] for pose in poses]
    yaws = [pose[2] for pose in poses]
    data = pl.DataFrame(
        {
            "t_ns": pl.Series("t_ns", [index * SEC for index in range(len(poses))], dtype=pl.Int64),
            "pos_x_gt": pl.Series("pos_x_gt", xs, dtype=pl.Float64),
            "pos_y_gt": pl.Series("pos_y_gt", ys, dtype=pl.Float64),
            "yaw_gt": pl.Series("yaw_gt", yaws, dtype=pl.Float64),
            "pos_x": pl.Series("pos_x", [x + odom_dx for x in xs], dtype=pl.Float64),
            "pos_y": pl.Series("pos_y", ys, dtype=pl.Float64),
            "yaw": pl.Series("yaw", yaws, dtype=pl.Float64),
        }
    )
    return AlignedEpisodeBundle(episode_id=episode_id, data=data, start_pos=[], goal_pos=[])


def _series(poses: list[tuple[float, float, float]], *, odom_dx: float = 0.0) -> pl.DataFrame:
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    return calculator.series_table(_episode(poses, odom_dx=odom_dx), run_id="run", robot_id="jackal_0")


def test_output_keys_keep_only_the_aggregates():
    expected = ["path_length_m", "curvature_mean", "curvature_max", "roughness_mean", "roughness_max", "angle_over_length"]

    assert PathMetricsCalculator.output_keys() == expected
    assert {metric.name for metric in PathMetricsCalculator.OUTPUTS} == set(expected)


def test_series_table_is_flat_with_one_row_per_resolved_sample():
    table = _series([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)], odom_dx=0.5)

    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 4,
            "episode_id": [1] * 4,
            "robot_id": ["jackal_0"] * 4,
            "t_ns": [0, SEC, 2 * SEC, 3 * SEC],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "yaw_rad": [0.0, 0.0, 0.0, 0.0],
            "odom_x_m": [0.5, 1.5, 2.5, 3.5],
            "odom_y_m": [0.0, 0.0, 0.0, 0.0],
            "odom_yaw_rad": [0.0, 0.0, 0.0, 0.0],
            "step_length_m": [None, 1.0, 1.0, 1.0],
            "curvature_inv_m": [None, 0.0, 0.0, None],
            "normalized_curvature": [None, 0.0, 0.0, None],
            "roughness": [None, 0.0, 0.0, None],
        },
        schema=SERIES_SCHEMA,
    )

    assert_frame_equal(table, expected, check_dtypes=True, rel_tol=1e-9, abs_tol=1e-12)


def test_curvature_lands_on_the_middle_sample_of_its_triple():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = _episode([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, math.pi / 2)])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    expected_curvature = math.sqrt(2.0)
    expected_roughness = 0.5
    assert_series_equal(table["curvature_inv_m"], pl.Series("curvature_inv_m", [None, expected_curvature, None], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert_series_equal(table["normalized_curvature"], pl.Series("normalized_curvature", [None, 2.0 * math.sqrt(2.0), None], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert_series_equal(table["roughness"], pl.Series("roughness", [None, expected_roughness, None], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert results["curvature_mean"] == pytest.approx(expected_curvature)
    assert results["curvature_max"] == pytest.approx(expected_curvature)
    assert results["roughness_mean"] == pytest.approx(expected_roughness)
    assert results["path_length_m"] == pytest.approx(2.0)
    assert results["angle_over_length"] == pytest.approx((math.pi / 2) / 2.0)


def test_stationary_track_has_zero_length_and_no_curvature():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = _episode([(0.0, 0.0, 0.0)] * 4)

    results = calculator.calculate(episode, {})
    table = calculator.series_table(episode)

    assert results["path_length_m"] == 0.0
    assert results["angle_over_length"] is None
    assert results["curvature_mean"] is None
    assert results["roughness_mean"] is None
    assert_series_equal(table["step_length_m"], pl.Series("step_length_m", [None, 0.0, 0.0, 0.0], dtype=pl.Float64), check_names=False)
    assert table["curvature_inv_m"].null_count() == table.height


def test_null_ground_truth_rows_are_dropped_before_the_geometry():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    data = pl.DataFrame(
        {
            "t_ns": pl.Series("t_ns", [0, SEC, 2 * SEC, 3 * SEC], dtype=pl.Int64),
            "pos_x_gt": pl.Series("pos_x_gt", [0.0, None, 2.0, 3.0], dtype=pl.Float64),
            "pos_y_gt": pl.Series("pos_y_gt", [0.0, 0.0, 0.0, 0.0], dtype=pl.Float64),
            "yaw_gt": pl.Series("yaw_gt", [0.0, 0.0, 0.0, 0.0], dtype=pl.Float64),
        }
    )

    results = calculator.calculate(AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[]), {})
    table = calculator.series_table(AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[]))

    assert table.height == 3
    assert results["path_length_m"] == pytest.approx(3.0)
    assert_series_equal(table["step_length_m"], pl.Series("step_length_m", [None, 2.0, 1.0], dtype=pl.Float64), check_names=False)
    assert table["x_m"].null_count() == 0
    assert table["odom_x_m"].is_null().all()


def test_missing_ground_truth_yields_null_aggregates_and_an_empty_series():
    data = pl.DataFrame(
        {
            "t_ns": pl.Series("t_ns", [0, SEC], dtype=pl.Int64),
            "pos_x": pl.Series("pos_x", [0.0, 1.0], dtype=pl.Float64),
            "pos_y": pl.Series("pos_y", [0.0, 0.0], dtype=pl.Float64),
            "yaw": pl.Series("yaw", [0.0, 0.0], dtype=pl.Float64),
        }
    )
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(PathMetricsCalculator.output_keys())
    assert_frame_equal(calculator.series_table(episode), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)


def test_without_an_aligned_frame_the_aggregates_and_the_series_are_absent():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert calculator.calculate(episode, {}) == dict.fromkeys(PathMetricsCalculator.output_keys())
    assert_frame_equal(calculator.series_table(episode), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)


def test_a_single_sample_has_no_step_and_no_curvature():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = _episode([(2.0, 3.0, 0.5)])

    table = calculator.series_table(episode)
    results = calculator.calculate(episode, {})

    assert table.height == 1
    assert table["step_length_m"].is_null().all()
    assert table["curvature_inv_m"].is_null().all()
    assert results["path_length_m"] == 0.0
    assert results["curvature_max"] is None
    assert results["roughness_max"] is None


def test_path_length_is_the_single_metric_on_an_encounter_or_robot_slice():
    calculator = PathMetricsCalculator(RobotParams(robot_radius=0.2))
    episode = _episode([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])

    for scope in (MetricScope.ENCOUNTER, MetricScope.ROBOT):
        scope_slice = ScopeSlice(
            scope=scope,
            slice_id=f"{scope.value}-1",
            parent_id=None,
            t_start_ns=0,
            t_end_ns=2 * SEC,
            identity={},
            window_bundle=episode,
            full_episode=episode,
        )
        result = calculator.calculate_for_scope(episode, {}, scope_slice)
        assert set(calculator.output_keys_for_scope(scope)) <= set(result)
        assert result["path_length_m"] == pytest.approx(2.0)
