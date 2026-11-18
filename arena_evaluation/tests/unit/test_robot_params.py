"""RobotParams holds recorded robot parameters: every field is required and nothing is defaulted."""

from __future__ import annotations

import dataclasses

import pytest

from arena_evaluation.storage.schemas import RobotParams

_PARAMETERS = {
    "model": "jackal",
    "robot_radius": 0.43,
    "laser_min_range": None,
    "laser_max_range": None,
    "base_mass": 18.431,
    "component_masses": {},
}


def test_total_mass_is_base_plus_components():
    """Mass sums the way power does: platform base, then each component."""
    params = RobotParams(
        model="jackal",
        robot_radius=0.43,
        laser_min_range=None,
        laser_max_range=None,
        base_mass=18.431,
        component_masses={"lidar/x": 0.13, "camera/y": 0.07},
    )

    assert params.mass == pytest.approx(18.631)


def test_components_without_a_declared_mass_contribute_nothing():
    params = RobotParams(**_PARAMETERS)

    assert params.mass == pytest.approx(18.431)


def test_a_robot_without_parameters_is_refused():
    with pytest.raises(TypeError):
        RobotParams()


@pytest.mark.parametrize("missing", list(_PARAMETERS))
def test_every_parameter_field_is_required(missing):
    provided = {key: value for key, value in _PARAMETERS.items() if key != missing}

    with pytest.raises(TypeError):
        RobotParams(**provided)


def test_the_dataclass_declares_no_defaults():
    fields = dataclasses.fields(RobotParams)

    assert all(field.default is dataclasses.MISSING for field in fields)
    assert all(field.default_factory is dataclasses.MISSING for field in fields)


def test_every_parameter_is_keyword_only():
    positional = tuple(_PARAMETERS.values())

    with pytest.raises(TypeError):
        RobotParams(*positional)


def test_the_loader_is_gone():
    assert "load" not in vars(RobotParams)
