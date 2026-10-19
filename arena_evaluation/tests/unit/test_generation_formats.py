from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from arena_evaluation.generation.formats import SIDECAR, first_level, read_scenario, write_scenario

legacy = pytest.importorskip("arena_simulation_setup.tree.World.legacy")
scenario_format = pytest.importorskip("arena_simulation_setup.tree.World.Scenario")

_LEVEL = {
    "zones": [
        {
            "name": "lobby",
            "corners": [{"x": 0.0, "y": 0.0}, {"x": 6.0, "y": 0.0}, {"x": 6.0, "y": 4.0}, {"x": 0.0, "y": 4.0}],
            "entities": {"static": [{"name": "bench", "model": "Hospital/Object/Waiting_Room_Bench", "pose": {"position": {"x": 2.0, "y": 1.0, "z": 0.0}, "orientation": {"w": math.cos(0.35), "x": 0.0, "y": 0.0, "z": math.sin(0.35)}}}]},
        },
        {
            "name": "ward",
            "corners": [{"x": 6.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 4.0}, {"x": 6.0, "y": 4.0}],
            "entities": {"static": [{"name": "bed", "model": "Hospital/Object/SM_HospitalBed_02d", "pose": {"position": {"x": 8.0, "y": 2.0, "z": 0.0}, "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}}]},
        },
    ],
}
_MAP = {"image": "map.png", "resolution": 0.05, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "free_thresh": 0.2, "negate": 0}


def _shape(level: dict) -> list[tuple]:
    out = []
    for zone in level["zones"]:
        statics = []
        for ent in zone["entities"]["static"]:
            statics.append((ent["name"], ent["model"], round(ent["pose"][0], 6), round(ent["pose"][1], 6)))
        out.append((zone["name"], [(c["x"], c["y"]) for c in zone["corners"]], statics))
    return out


def test_a_format_2_world_reads_like_the_format_1_world_it_was_migrated_from(tmp_path: Path) -> None:
    v1 = tmp_path / "v1"
    (v1 / "0").mkdir(parents=True)
    (v1 / "0" / "world.yaml").write_text(yaml.safe_dump(_LEVEL))
    (v1 / "0" / "map.yaml").write_text(yaml.safe_dump(_MAP))
    world, _kg, _warnings = legacy.upgrade({"0": _LEVEL})
    v2 = tmp_path / "v2"
    (v2 / "map").mkdir(parents=True)
    (v2 / "world.yaml").write_text(yaml.safe_dump(world))
    (v2 / "map" / "0.yaml").write_text(yaml.safe_dump(_MAP))

    assert _shape(first_level(v2)) == _shape(first_level(v1))


def test_a_scenario_is_written_as_format_2_with_its_block_beside_it(tmp_path: Path) -> None:
    doc = {
        "robots": [{"start": [1.0, 1.0, 0.0], "phases": [{"goto": [8.0, 2.0]}]}],
        "dynamic": [{"name": "talker_1", "model": "female_adult_medical_01", "pose": "lobby", "agent": {"agent_type": "./talker.yaml", "desired_velocity": 1.1}, "waypoints": ["lobby", "ward"], "waypoint_mode": "reverse"}],
        "edge_case": {"steps": ["A:evacuation"], "effects": [{"type": "stimulus", "label": "A:evacuation", "scope": ["talker_1"], "need": "alarm", "when": {"at": 8.0}}]},
    }
    path = write_scenario(tmp_path / "s", doc, "# header\n")

    raw = yaml.safe_load(path.read_text())
    # zone-name poses resolve only against a world (the integration test loads one); the keys are what format 2 checks first
    assert raw["format"] == scenario_format.SCENARIO_FORMAT and set(raw) <= scenario_format.SCENARIO_KEYS and raw["humans"][0]["pose"] == "lobby"
    assert yaml.safe_load((tmp_path / "s" / SIDECAR).read_text()) == doc["edge_case"]
    assert read_scenario(path) == {"robots": doc["robots"], "dynamic": doc["dynamic"], "edge_case": doc["edge_case"]}


def test_a_key_format_2_would_drop_refuses_to_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="would lose"):
        write_scenario(tmp_path / "s", {"dynamic": [], "knobs": {"speed": "high"}})
