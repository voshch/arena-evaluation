"""RobotParams from the recorded fleet rows: descriptor model plus resolved parameters, never a default."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.storage.exceptions import ContractViolationError, MetricInputError
from arena_evaluation.storage.schemas import robot_params_from_recorded

DESCRIPTOR = {
    "env_id": ["env_0"],
    "robot_id": ["jackal"],
    "t_ns": [0],
    "model": ["jackal"],
    "name": ["jackal"],
    "ns": ["jackal"],
    "frame": ["odom"],
    "base_frame": ["base_link"],
}


def _params(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "env_id": ["env_0"] * len(pairs),
            "robot_id": ["jackal"] * len(pairs),
            "t_ns": [0] * len(pairs),
            "param_key": [key for key, _ in pairs],
            "param_value": [value for _, value in pairs],
        }
    )


def _rows(*pairs: tuple[str, str]) -> pl.DataFrame:
    """The recorded fleet rows: the descriptor row and one row per resolved parameter."""
    return pl.concat([pl.DataFrame(DESCRIPTOR), _params(*pairs)], how="diagonal")


def test_the_descriptor_model_and_the_recorded_parameters_are_used():
    params = robot_params_from_recorded(
        _rows(
            ("robot_radius", "0.43"),
            ("robot_mass_kg", "18.4"),
        ),
        "jackal",
    )

    assert params.model == "jackal"
    assert params.robot_radius == pytest.approx(0.43)
    assert params.base_mass == pytest.approx(18.4)


def test_a_robot_without_a_descriptor_row_is_refused():
    rows = _rows(("robot_radius", "0.43"))

    with pytest.raises(MetricInputError, match="has no descriptor row"):
        robot_params_from_recorded(rows, "ghost")


def test_a_descriptor_row_without_a_model_is_not_a_descriptor():
    rows = _rows(
        ("robot_radius", "0.43"),
        ("robot_mass_kg", "18.4"),
    ).with_columns(model=pl.lit(None, dtype=pl.String))

    with pytest.raises(MetricInputError, match="has no descriptor row"):
        robot_params_from_recorded(rows, "jackal")


def test_a_missing_required_parameter_is_refused_not_defaulted():
    rows = _rows(("robot_radius", "0.43"))

    with pytest.raises(MetricInputError, match="robot_mass_kg"):
        robot_params_from_recorded(rows, "jackal")


def test_an_unparsable_parameter_is_refused():
    rows = _rows(
        ("robot_radius", "wide"),
        ("robot_mass_kg", "18.4"),
    )

    with pytest.raises(ContractViolationError, match="not a number"):
        robot_params_from_recorded(rows, "jackal")


def test_a_laser_range_that_was_not_recorded_stays_none():
    params = robot_params_from_recorded(
        _rows(("robot_radius", "0.43"), ("robot_mass_kg", "18.4")),
        "jackal",
    )

    assert params.laser_min_range is None
    assert params.laser_max_range is None
    assert params.component_masses == {}


def test_a_wide_parameter_column_is_not_a_parameter_row():
    rows = pl.DataFrame(
        {
            **DESCRIPTOR,
            "robot_radius": [0.43],
            "base_mass": [18.4],
        }
    )

    with pytest.raises(MetricInputError, match="robot_params"):
        robot_params_from_recorded(rows, "jackal")
