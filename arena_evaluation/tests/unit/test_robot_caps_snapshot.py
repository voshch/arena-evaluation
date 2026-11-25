import copy
import json
import pathlib

import polars as pl
import pytest
import yaml

pytest.importorskip("ament_index_python", reason="arena_robots share dir needs ROS")
pytest.importorskip("arena_robots", reason="the snapshot is resolved by arena_robots")

from arena_robots.fleet import instance_caps
from robot_params_factory import make_robot_params

from arena_evaluation.processing.metrics.ecological.characterization import CharacterizationCalculator
from arena_evaluation.processing.metrics.ecological.energy import EnergyCalculator
from arena_evaluation.processing.pipeline import _robot_identity
from arena_evaluation.storage.schemas import TopicBundle, caps_snapshot_path

_SCALED = {"cap:mobile.limits": json.dumps({"scale": 0.5})}


def _snapshot(params: dict[str, str] | None = None) -> dict:
    return instance_caps("jackal", params or {}).snapshot()


def _bundle(snapshot_ns: str) -> TopicBundle:
    descriptor = pl.DataFrame(
        {
            "t_ns": [0],
            "env_id": ["env_0"],
            "robot_id": ["jackal"],
            "name": ["jackal"],
            "model": ["jackal"],
            "ns": [snapshot_ns],
            "frame": ["odom"],
            "base_frame": ["base_link"],
        }
    )
    params = pl.DataFrame(
        {
            "t_ns": [0, 0],
            "env_id": ["env_0", "env_0"],
            "robot_id": ["jackal", "jackal"],
            "param_key": ["robot_radius", "robot_mass_kg"],
            "param_value": ["0.43", "18.4"],
        }
    )
    return TopicBundle(robot_descriptor=descriptor, robot_params=params)


def test_the_snapshot_blocks_are_the_effective_caps():
    snapshot = _snapshot(_SCALED)
    params = make_robot_params(snapshot=snapshot)
    assert params.mobile == snapshot["effective"]["mobile"]
    assert params.power == snapshot["effective"].get("power")
    assert params.acoustics == snapshot["effective"].get("acoustics")


def test_a_robot_without_a_snapshot_has_no_blocks():
    params = make_robot_params()
    assert params.mobile is None
    assert params.power is None
    assert params.acoustics is None


def test_the_pipeline_reads_the_snapshot_recorded_under_the_robot_namespace(tmp_path: pathlib.Path):
    path = caps_snapshot_path(tmp_path, "/env_0/jackal")
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(_snapshot(_SCALED), sort_keys=False))

    _, params = _robot_identity(_bundle("/env_0/jackal"), tmp_path)

    assert params.snapshot is not None
    assert params.snapshot["hash"] == _snapshot(_SCALED)["hash"]
    assert params.robot_radius == pytest.approx(0.43)


def test_the_pipeline_records_no_snapshot_for_a_run_without_one(tmp_path: pathlib.Path):
    assert _robot_identity(_bundle("jackal"), tmp_path)[1].snapshot is None


def test_energy_reads_the_rolling_resistance_of_the_snapshot_power_block():
    snapshot = copy.deepcopy(_snapshot())
    snapshot["effective"].setdefault("power", {}).setdefault("power_system", {})["rolling_resistance_crr"] = 0.123
    assert EnergyCalculator(make_robot_params(snapshot=snapshot))._rolling_resistance == pytest.approx(0.123)


def test_energy_has_no_rolling_resistance_without_a_power_block():
    snapshot = copy.deepcopy(_snapshot())
    snapshot["effective"].pop("power", None)
    assert EnergyCalculator(make_robot_params(snapshot=snapshot))._rolling_resistance is None


def test_characterization_reads_the_mobile_cap_of_the_snapshot():
    mobile = CharacterizationCalculator(make_robot_params(snapshot=_snapshot({"cap:mobile.radius": "0.5"})))._mobile()
    assert mobile is not None
    assert mobile.radius == pytest.approx(0.5)


def test_characterization_of_a_snapshot_without_a_mobile_cap_has_no_mobile():
    snapshot = copy.deepcopy(_snapshot())
    snapshot["effective"].pop("mobile")
    assert CharacterizationCalculator(make_robot_params(snapshot=snapshot))._mobile() is None
