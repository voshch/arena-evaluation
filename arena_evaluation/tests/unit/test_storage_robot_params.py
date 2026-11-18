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


def test_a_descriptor_row_without_a_model_is_not_a_descriptor():
    rows = _rows(
        ("robot_radius", "0.43"),
        ("robot_mass_kg", "18.4"),
    ).with_columns(model=pl.lit(None, dtype=pl.String))

    with pytest.raises(MetricInputError, match="has no descriptor row"):
        robot_params_from_recorded(rows, "jackal")
