from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_evaluation.generation.formats import first_level
from arena_evaluation.generation.scenario.occupancy import world_statics
from arena_evaluation.generation.scenario.zone_screen import _stamp_statics

pytest.importorskip("arena_simulation_setup.tree.World.view")

_WORLD = {
    "format": "arena-world/2",
    "levels": [
        {
            "id": "0",
            "zones": [{"name": "lobby", "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]], "floor": False}],
            "objects": [
                {"name": "counter", "pose": [4.0, 2.8, 0.0], "model": "Hospital/TestDesk"},
                {"name": "plant", "pose": [8.0, 5.0, 0.0], "model": "Hospital/TestPlant"},
                {"name": "kiosk", "pose": [2.0, 1.0, 0.0], "model": "Hospital/WorldKiosk"},
            ],
        }
    ],
}


@pytest.fixture
def world(tmp_path: Path) -> Path:
    from arena_simulation_setup.tree import DynamicPaths

    root = tmp_path / "world"
    root.mkdir()
    (root / "world.yaml").write_text(yaml.safe_dump(_WORLD))
    desk = tmp_path / "assets" / "Hospital" / "Object" / "TestDesk"
    desk.mkdir(parents=True)
    (desk / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[0.0, 2.42], [-2.76, 0.10], [0.0, 1.0]]}))
    (tmp_path / "assets" / "Hospital" / "Object" / "TestPlant").mkdir(parents=True)
    kiosk = root / "assets" / "Hospital" / "Object" / "WorldKiosk"
    kiosk.mkdir(parents=True)
    (kiosk / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]}))
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path / "assets"
    yield root
    DynamicPaths.ARENA.path = previous


def test_an_annotated_object_is_its_rotated_bounds(world: Path) -> None:
    statics = {s["name"]: s for s in world_statics(world)}
    xs, ys = zip(*statics["counter"]["polygon"], strict=True)
    assert (min(xs), round(max(xs), 2), round(min(ys), 2), round(max(ys), 2)) == (4.0, 6.42, 0.04, 2.9)
    assert statics["counter"]["radius"] == pytest.approx(math.hypot(2.42, 2.76))


def test_an_unannotated_object_is_the_runtime_box(world: Path) -> None:
    plant = next(s for s in world_statics(world) if s["name"] == "plant")
    xs, ys = zip(*plant["polygon"], strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == (7.0, 9.0, 4.0, 6.0)


def test_an_asset_shipped_with_the_world_resolves(world: Path) -> None:
    kiosk = next(s for s in world_statics(world) if s["name"] == "kiosk")
    xs, ys = zip(*kiosk["polygon"], strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == (1.5, 2.5, 0.5, 1.5)


def test_the_desk_is_stamped_as_its_box(world: Path) -> None:
    res = 0.1
    occupied = np.zeros((100, 100), dtype=bool)
    _stamp_statics(occupied, world_statics(world), res, 0.0, 0.0)

    def cell(x: float, y: float) -> bool:
        return bool(occupied[occupied.shape[0] - 1 - int(y / res), int(x / res)])

    assert cell(5.0, 1.5) and cell(6.3, 0.2) and not cell(3.5, 1.5) and not cell(4.0, 4.0)


def test_objects_sit_in_the_zone_that_covers_them(world: Path) -> None:
    (lobby,) = first_level(world)["zones"]
    assert [e["model"] for e in lobby["entities"]["static"]] == ["TestDesk", "TestPlant", "WorldKiosk"]
