import pytest
import yaml
from arena_evaluation_mcp.eval_bridge import EvalBridge
from arena_evaluation_mcp.tools import _dispatch


@pytest.fixture
def bridge():
    return EvalBridge()


def test_validate_scenario_valid(bridge):
    valid_yaml = """
format: arena-scenario/2
robots:
  - start: [10.0, 2.0, 1.57]
    phases:
      - goto: [10.0, 30.0, 1.57]
humans:
  - name: ped_1
    model: arenian
    pose: [10.0, 28.0, -1.57]
    velocity: 1.2
    waypoints: [[10.0, 4.0, -1.57]]
objects: []
regions: {}
"""
    res = _dispatch(
        "validate_scenario",
        {"yaml_content": valid_yaml, "map_name": "hospital_1"},
        bridge,
    )
    assert res.get("valid") is True
    assert res.get("n_robots") == 1
    assert res.get("n_dynamic_peds") == 1
    assert len(res.get("warnings", [])) == 0


def test_validate_scenario_missing_robot_start(bridge):
    invalid_yaml = """
format: arena-scenario/2
robots:
  - phases: [{goto: [10.0, 30.0, 1.57]}]
humans: []
"""
    res = _dispatch(
        "validate_scenario",
        {"yaml_content": invalid_yaml},
        bridge,
    )
    assert res.get("valid") is False
    assert "missing required 'start'" in res.get("error", "")


def test_validate_scenario_format1_rejected(bridge):
    legacy_yaml = """
robots:
  - start: [10.0, 2.0, 1.57]
    phases: [{goto: [10.0, 30.0, 1.57]}]
dynamic: []
"""
    res = _dispatch(
        "validate_scenario",
        {"yaml_content": legacy_yaml},
        bridge,
    )
    assert res.get("valid") is False
    assert "arena-scenario/2" in res.get("error", "")


def test_validate_scenario_out_of_bounds_warning(bridge):
    oob_yaml = """
format: arena-scenario/2
robots:
  - start: [1000.0, 2000.0, 1.57]
    phases:
      - goto: [10.0, 30.0, 1.57]
humans:
  - name: ped_1
    model: arenian
    pose: [10.0, 28.0, -1.57]
"""
    res = _dispatch(
        "validate_scenario",
        {"yaml_content": oob_yaml, "map_name": "hospital_1"},
        bridge,
    )
    assert res.get("valid") is True
    assert len(res.get("warnings", [])) > 0
    assert "outside map bounds" in res["warnings"][0]


def test_create_scenario_execution(bridge, tmp_path, monkeypatch):
    test_yaml = """
format: arena-scenario/2
robots:
  - start: [10.0, 2.0, 1.57]
    phases:
      - goto: [10.0, 30.0, 1.57]
humans:
  - name: ped_1
    model: arenian
    pose: [10.0, 28.0, -1.57]
    velocity: 1.2
    waypoints: [[10.0, 4.0, -1.57]]
"""
    target_file = tmp_path / "scenarios" / "s_test" / "scenario.yaml"
    monkeypatch.setattr(
        bridge,
        "scenario_write_targets",
        lambda map_name, scenario_name, location="both": [target_file],
    )

    res = _dispatch(
        "create_scenario",
        {
            "map_name": "hospital_1",
            "scenario_name": "s_test",
            "yaml_content": test_yaml,
        },
        bridge,
    )
    assert res.get("scenario_valid") is True
    assert target_file.is_file()
    loaded = yaml.safe_load(target_file.read_text())
    assert loaded["format"] == "arena-scenario/2"
    assert loaded["robots"][0]["start"] == [10.0, 2.0, 1.57]
