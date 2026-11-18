"""The series tables: eleven declared kinds of per-sample rows, their names, keys, schemas and the writer that conforms them."""

from __future__ import annotations

import pathlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.scoped_evaluator import write_series_tables
from arena_evaluation.storage.contracts import ContractViolationError
from arena_evaluation.storage.registry import TableKind, registered_tables
from arena_evaluation.storage.registry.series import SERIES_TABLES, SeriesClearance
from arena_evaluation.storage.schemas import RobotParams

_ROBOT_KEY = ("run_id", "episode_id", "robot_id", "t_ns")
_PEDESTRIAN_KEY = ("run_id", "episode_id", "robot_id", "t_ns", "ped_id")
_IDENTITY = (
    ("run_id", pl.String),
    ("episode_id", pl.Int64),
    ("robot_id", pl.String),
    ("t_ns", pl.Int64),
)

_EXPECTED_COLUMNS: dict[str, tuple[tuple[str, pl.DataType], ...]] = {
    "series_characterization": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("power_total_w", pl.Float64),
        ("power_mech_w", pl.Float64),
        ("dba", pl.Float64),
        ("vx_achieved_mps", pl.Float64),
        ("vy_achieved_mps", pl.Float64),
        ("wz_achieved_radps", pl.Float64),
        ("speed_achieved_mps", pl.Float64),
        ("accel_achieved_mps2", pl.Float64),
        ("accel_target_mps2", pl.Float64),
        ("efficiency", pl.Float64),
        ("phase_kind", pl.String),
        ("vx_target_mps", pl.Float64),
        ("vy_target_mps", pl.Float64),
        ("wz_target_radps", pl.Float64),
        ("speed_target_mps", pl.Float64),
        ("turn_radius_m", pl.Float64),
        ("energy_intensity_j_m", pl.Float64),
        ("energy_per_rad_j_rad", pl.Float64),
        ("leq_power_w", pl.Float64),
    ),
    "series_clearance": (
        *_IDENTITY,
        ("obstacle_clearance_m", pl.Float64),
        ("pedestrian_clearance_m", pl.Float64),
        ("combined_clearance_m", pl.Float64),
    ),
    "series_energy": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("power_total_w", pl.Float64),
        ("power_static_w", pl.Float64),
        ("power_mechanical_w", pl.Float64),
        ("power_thermal_w", pl.Float64),
        ("battery_soc_pct", pl.Float64),
        ("velocity_linear_mps", pl.Float64),
    ),
    "series_gaze": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("looking_at_peds", pl.Int64),
        ("looked_at_by_peds", pl.Int64),
    ),
    "series_hhi_contact": (
        *_IDENTITY,
        ("onset_s", pl.Float64),
        ("duration_s", pl.Float64),
        ("window", pl.String),
    ),
    "series_motion_metrics": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("speed_mps", pl.Float64),
        ("acceleration_mps2", pl.Float64),
        ("jerk_mps3", pl.Float64),
    ),
    "series_path_metrics": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("x_m", pl.Float64),
        ("y_m", pl.Float64),
        ("yaw_rad", pl.Float64),
        ("odom_x_m", pl.Float64),
        ("odom_y_m", pl.Float64),
        ("odom_yaw_rad", pl.Float64),
        ("step_length_m", pl.Float64),
        ("curvature_inv_m", pl.Float64),
        ("normalized_curvature", pl.Float64),
        ("roughness", pl.Float64),
    ),
    "series_pedestrian_path_metrics": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("ped_id", pl.String),
        ("x_m", pl.Float64),
        ("y_m", pl.Float64),
        ("z_m", pl.Float64),
        ("yaw_rad", pl.Float64),
        ("vx_mps", pl.Float64),
        ("vy_mps", pl.Float64),
        ("speed_mps", pl.Float64),
    ),
    "series_proxemics": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("peds_in_personal_space", pl.Int64),
    ),
    "series_proxemics_extended": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("min_ped_clearance_m", pl.Float64),
    ),
    "series_social_forces": (
        *_IDENTITY,
        ("time_s", pl.Float64),
        ("sfm_force_n", pl.Float64),
        ("ci", pl.Float64),
    ),
}

_EXPECTED_KEYS: dict[str, tuple[str, ...]] = {
    name: _PEDESTRIAN_KEY if name == "series_pedestrian_path_metrics" else _ROBOT_KEY
    for name in _EXPECTED_COLUMNS
}


def _series_classes() -> dict[str, type]:
    return {
        table.table_name: table
        for table in registered_tables()
        if table.table_kind is TableKind.SERIES
    }


def test_the_eleven_series_tables_are_declared_as_their_own_kind() -> None:
    assert set(_series_classes()) == set(_EXPECTED_COLUMNS)
    assert {table.table_name for table in SERIES_TABLES.values()} == set(
        _EXPECTED_COLUMNS
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED_COLUMNS))
def test_series_schema_is_pinned_in_name_order_and_dtype(name: str) -> None:
    assert _series_classes()[name].polars_schema() == pl.Schema(_EXPECTED_COLUMNS[name])


@pytest.mark.parametrize("name", sorted(_EXPECTED_COLUMNS))
def test_series_row_key_is_pinned(name: str) -> None:
    assert _series_classes()[name].key == _EXPECTED_KEYS[name]


@pytest.mark.parametrize("name", sorted(_EXPECTED_COLUMNS))
def test_series_file_lives_in_the_metrics_folder_of_its_episode(name: str) -> None:
    assert (
        _series_classes()[name].relative_path().as_posix() == f"metrics/{name}.parquet"
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED_COLUMNS))
def test_series_carry_no_null_reason_column(name: str) -> None:
    assert "null_reason" not in _series_classes()[name].polars_schema()


def test_every_series_table_has_a_producing_calculator_and_every_producer_a_table() -> (
    None
):
    registry = MetricRegistry(RobotParams(robot_radius=0.25))
    produced = {stem for stem, _ in registry.series_calculators()}

    assert produced == set(SERIES_TABLES)


def test_a_series_frame_is_written_and_read_back_with_its_declared_class(
    tmp_path: pathlib.Path,
) -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["bench", "bench"],
            "episode_id": [0, 0],
            "robot_id": ["jackal", "jackal"],
            "t_ns": [0, 100_000_000],
            "obstacle_clearance_m": [0.5, None],
            "pedestrian_clearance_m": [None, 1.25],
            "combined_clearance_m": [0.5, 1.25],
        },
        schema=SeriesClearance.polars_schema(),
    )
    metrics = tmp_path / "episodes" / "episode_000" / "metrics"

    write_series_tables({"clearance": frame}, metrics)

    assert_frame_equal(
        SeriesClearance.scan(tmp_path / "episodes" / "episode_000").collect(), frame
    )
    assert (metrics / "series_clearance.parquet").is_file()


def _clearance(**overrides: pl.Series) -> pl.DataFrame:
    columns = {
        "run_id": ["bench"],
        "episode_id": [0],
        "robot_id": ["jackal"],
        "t_ns": [0],
        "obstacle_clearance_m": [0.5],
        "pedestrian_clearance_m": [None],
        "combined_clearance_m": [0.5],
    }
    return pl.DataFrame({**columns, **overrides})


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(_clearance().drop("combined_clearance_m"), id="missing-column"),
        pytest.param(
            _clearance().with_columns(extra_m=pl.lit(1.0)), id="undeclared-column"
        ),
    ],
)
def test_a_series_frame_that_drifts_from_its_declaration_is_refused(
    tmp_path: pathlib.Path, frame: pl.DataFrame
) -> None:
    with pytest.raises(ContractViolationError, match="does not match its declaration"):
        write_series_tables({"clearance": frame}, tmp_path / "metrics")


def test_a_series_stem_without_a_declaration_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ContractViolationError, match="'planted' is not declared"):
        write_series_tables({"planted": _clearance()}, tmp_path / "metrics")
