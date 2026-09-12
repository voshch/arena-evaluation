import pytest
import yaml

from arena_evaluation.storage.schemas import RobotParams

pytest.importorskip("ament_index_python", reason="arena_robots share dir needs ROS")


def _robot_yaml(model: str, *parts: str) -> dict:
    import os

    from ament_index_python.packages import get_package_share_directory

    path = os.path.join(get_package_share_directory("arena_robots"), "robots", model, *parts)
    with open(path) as f:
        return yaml.safe_load(f)


def test_unknown_model_declares_no_mass():
    """An unknown robot reports 0 kg, the sentinel for undeclared."""
    params = RobotParams.load("no_such_robot")
    assert params.model == "no_such_robot"
    assert params.base_mass == 0.0
    assert params.component_masses == {}
    assert params.mass == 0.0


def test_base_mass_and_radius_come_from_the_robot_definition():
    params = RobotParams.load("jackal")
    assert params.base_mass == pytest.approx(float(_robot_yaml("jackal", "model_params.yaml")["mass"]["base_kg"]))
    assert params.robot_radius == pytest.approx(float(_robot_yaml("jackal", "caps", "mobile.yaml")["radius"]))


def test_total_mass_is_base_plus_components():
    """Mass sums the way power does: platform base, then each component."""
    params = RobotParams(base_mass=18.431, component_masses={"lidar/x": 0.13, "camera/y": 0.07})
    assert params.mass == pytest.approx(18.631)


def test_components_without_a_declared_mass_contribute_nothing():
    params = RobotParams.load("jackal")
    assert params.mass == pytest.approx(params.base_mass + sum(params.component_masses.values()))


def test_mass_is_robot_specific():
    """A heavy base and a light one must not share a default."""
    light = RobotParams.load("turtlebot")
    heavy = RobotParams.load("husky")
    assert 0 < light.mass < heavy.mass
