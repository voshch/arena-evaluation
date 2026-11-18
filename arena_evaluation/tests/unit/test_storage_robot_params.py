"""robot_params_from_recorded derives every parameter from the declared registry tables only."""

from __future__ import annotations

import polars as pl
import pytest

from arena_evaluation.storage.exceptions import ContractViolationError, MetricInputError
from arena_evaluation.storage.schemas import robot_params_from_recorded

_DESCRIPTOR_FIELDS = {
    "env_id": "env_0",
    "robot_id": "jackal",
    "name": "jackal",
    "ns": "jackal",
    "frame": "odom",
    "base_frame": "base_link",
}


def _descriptors(*models: str) -> pl.DataFrame:
    count = len(models)
    return pl.DataFrame(
        {
            **{field: [value] * count for field, value in _DESCRIPTOR_FIELDS.items()},
            "t_ns": list(range(count)),
            "model": list(models),
        }
    )


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
    """The recorded fleet rows: the descriptor rows and one row per resolved parameter."""
    return pl.concat([_descriptors("jackal"), _params(*pairs)], how="diagonal")


def test_the_declared_rows_fill_every_recorded_field():
    params = robot_params_from_recorded(
        _rows(
            ("robot_radius", "0.43"),
            ("robot_mass_kg", "18.4"),
            ("laser_min_range", "0.05"),
            ("laser_max_range", "12.0"),
        ),
        "jackal",
    )

    assert params.model == "jackal"
    assert params.robot_radius == pytest.approx(0.43)
    assert params.base_mass == pytest.approx(18.4)
    assert params.laser_min_range == pytest.approx(0.05)
    assert params.laser_max_range == pytest.approx(12.0)
    assert params.component_masses == {}


def test_a_laser_range_the_recording_omits_is_none():
    params = robot_params_from_recorded(
        _rows(("robot_radius", "0.43"), ("robot_mass_kg", "18.4")),
        "jackal",
    )

    assert params.laser_min_range is None
    assert params.laser_max_range is None
    assert params.component_masses == {}


def test_a_wide_column_is_not_a_parameter_row():
    rows = _descriptors("jackal").with_columns(
        robot_radius=pl.lit(0.43),
        base_mass=pl.lit(18.4),
    )

    with pytest.raises(MetricInputError, match="robot_params"):
        robot_params_from_recorded(rows, "jackal")


def test_conflicting_values_of_one_parameter_are_refused():
    rows = _rows(
        ("robot_radius", "0.43"),
        ("robot_mass_kg", "18.4"),
        ("base_mass", "19.0"),
    )

    with pytest.raises(ContractViolationError, match="conflicting"):
        robot_params_from_recorded(rows, "jackal")


def test_a_required_parameter_is_refused_not_defaulted():
    rows = _rows(("robot_radius", "0.43"))

    with pytest.raises(MetricInputError, match="robot_mass_kg"):
        robot_params_from_recorded(rows, "jackal")


def test_a_parameter_that_is_not_a_number_is_refused():
    rows = _rows(("robot_radius", "wide"), ("robot_mass_kg", "18.4"))

    with pytest.raises(ContractViolationError, match="not a number"):
        robot_params_from_recorded(rows, "jackal")


def test_conflicting_descriptor_models_are_refused():
    rows = pl.concat(
        [
            _descriptors("jackal", "husky"),
            _params(("robot_radius", "0.43"), ("robot_mass_kg", "18.4")),
        ],
        how="diagonal",
    )

    with pytest.raises(ContractViolationError, match="conflicting"):
        robot_params_from_recorded(rows, "jackal")
