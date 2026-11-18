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
    "instance_params": {},
}


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
