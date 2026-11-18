"""Pedestrian trajectories: one row per (t_ns, ped_id) in the 1NF pedestrian path series table."""

from __future__ import annotations

import polars as pl
import pytest
from flat_peds import flat_peds
from polars.testing import assert_frame_equal, assert_series_equal

from arena_evaluation.processing.metrics.performance.pedestrian_path_metrics import (
    SERIES_SCHEMA,
    PedestrianPathMetricsCalculator,
)
from arena_evaluation.storage.schemas import AlignedEpisodeBundle
from robot_params_factory import make_robot_params

SEC = 1_000_000_000


def _episode(times: list[int], frames: list[tuple[int, list[dict]]] | None = None) -> AlignedEpisodeBundle:
    """Robot samples at the given times, with the pedestrian table of the given frames."""
    data = pl.DataFrame({"t_ns": times}, schema={"t_ns": pl.Int64})
    if frames is None:
        return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])
    peds = flat_peds(frames)
    return AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], peds=peds, topics={"pedestrian_tracks": peds})


def _series(times: list[int], frames: list[tuple[int, list[dict]]]) -> pl.DataFrame:
    calculator = PedestrianPathMetricsCalculator(make_robot_params())
    return calculator.series_table(_episode(times, frames), run_id="run", robot_id="jackal_0")


def _at(x: float, y: float) -> dict:
    return {"x": x, "y": y}


def test_the_legacy_list_key_has_no_metric_row_output():
    assert PedestrianPathMetricsCalculator.output_keys() == []
    assert PedestrianPathMetricsCalculator.OUTPUTS == ()
    assert PedestrianPathMetricsCalculator(make_robot_params()).calculate(AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[]), {}) == {}


def test_missing_frame_yields_a_schema_typed_empty_table():
    calculator = PedestrianPathMetricsCalculator(make_robot_params())
    episode = AlignedEpisodeBundle(episode_id=1, data=None, start_pos=[], goal_pos=[])

    assert_frame_equal(calculator.series_table(episode), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)


def test_robot_samples_without_a_pedestrian_table_yield_a_schema_typed_empty_table():
    calculator = PedestrianPathMetricsCalculator(make_robot_params())

    assert_frame_equal(calculator.series_table(_episode([0, SEC])), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)


def test_a_pedestrian_table_without_rows_yields_a_schema_typed_empty_table():
    calculator = PedestrianPathMetricsCalculator(make_robot_params())

    assert_frame_equal(calculator.series_table(_episode([0, SEC], [])), pl.DataFrame(schema=SERIES_SCHEMA), check_dtypes=True)


def test_two_pedestrians_are_one_flat_row_each_per_frame():
    frames = [(0, [{"ped_id": "a", **_at(0.0, 0.0)}, {"ped_id": "b", **_at(5.0, 5.0)}]), (SEC, [{"ped_id": "a", **_at(1.0, 0.0)}, {"ped_id": "b", **_at(5.0, 6.0)}])]

    expected = pl.DataFrame(
        {
            "run_id": ["run"] * 4,
            "episode_id": [1] * 4,
            "robot_id": ["jackal_0"] * 4,
            "t_ns": [0, 0, SEC, SEC],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "ped_id": ["a", "b", "a", "b"],
            "x_m": [0.0, 5.0, 1.0, 5.0],
            "y_m": [0.0, 5.0, 0.0, 6.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "yaw_rad": [None, None, None, None],
            "vx_mps": [0.0, 0.0, 0.0, 0.0],
            "vy_mps": [0.0, 0.0, 0.0, 0.0],
            "speed_mps": [0.0, 0.0, 0.0, 0.0],
        },
        schema=SERIES_SCHEMA,
    )

    assert_frame_equal(_series([0, SEC], frames), expected, check_dtypes=True, rel_tol=1e-9, abs_tol=1e-12)


def test_recorded_velocity_reaches_the_series():
    frames = [(0, [{"ped_id": "a", "x": 1.0, "y": 2.0, "vx": 1.5, "vy": 0.0}])]

    table = _series([0], frames)

    assert table.height == 1
    assert_series_equal(table["vx_mps"], pl.Series("vx_mps", [1.5], dtype=pl.Float64), check_names=False, rel_tol=1e-9)
    assert_series_equal(table["vy_mps"], pl.Series("vy_mps", [0.0], dtype=pl.Float64), check_names=False)
    assert_series_equal(table["speed_mps"], pl.Series("speed_mps", [1.5], dtype=pl.Float64), check_names=False, rel_tol=1e-9)


def test_a_pedestrian_without_a_valid_position_is_dropped():
    frames = [
        (0, [{"ped_id": "a", **_at(0.0, 0.0)}, {"ped_id": "b", **_at(float("nan"), 1.0)}]),
        (SEC, [{"ped_id": "a", **_at(1.0, 0.0)}, {"ped_id": "b", **_at(float("nan"), 2.0)}]),
    ]

    table = _series([0, SEC], frames)

    assert table.filter(pl.col("ped_id") != "a").is_empty()
    assert_series_equal(table["x_m"], pl.Series("x_m", [0.0, 1.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)


def test_leading_robot_samples_before_the_first_message_add_no_pedestrian_rows():
    table = _series([0, SEC], [(SEC, [{"ped_id": "a", **_at(1.0, 2.0)}])])

    assert table.height == 1
    assert_series_equal(table["t_ns"], pl.Series("t_ns", [SEC], dtype=pl.Int64), check_names=False)
    assert_series_equal(table["x_m"], pl.Series("x_m", [1.0], dtype=pl.Float64), check_names=False, rel_tol=1e-9)


def test_a_pedestrian_table_without_a_position_column_is_rejected():
    calculator = PedestrianPathMetricsCalculator(make_robot_params())
    tracks = pl.DataFrame({"t_ns": [0], "ped_id": ["a"], "x": [0.0]}, schema={"t_ns": pl.Int64, "ped_id": pl.String, "x": pl.Float64})
    episode = AlignedEpisodeBundle(episode_id=1, data=pl.DataFrame({"t_ns": [0]}, schema={"t_ns": pl.Int64}), start_pos=[], goal_pos=[], topics={"pedestrian_tracks": tracks})

    with pytest.raises(ValueError, match=r"pedestrian_tracks lacks \['y'\]"):
        calculator.series_table(episode)
