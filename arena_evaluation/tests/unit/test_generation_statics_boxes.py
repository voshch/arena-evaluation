from __future__ import annotations

from pathlib import Path

import numpy as np

from arena_evaluation.generation.scenario.occupancy import asset_annotation, asset_box, find_assets, static_polygon, world_statics
from arena_evaluation.generation.scenario.zone_screen import _stamp_statics

ASSETS = Path(__file__).resolve().parents[4] / "_assets" / "default"
DESK = "Hospital/SM_ReceptionDesk_01a"


def test_annotations_resolve_through_the_object_directory_like_the_runtime() -> None:
    assert asset_annotation(DESK, ASSETS) is not None
    x_lo, x_hi, y_lo, y_hi = asset_box(DESK, ASSETS)
    assert (x_lo, round(x_hi, 2)) == (0.0, 2.42) and y_lo < -2.7 < y_hi, "the desk's box is offset from its origin"


def test_a_static_with_a_box_is_its_rotated_rectangle() -> None:
    desk = {"pose": [4.0, 2.8], "yaw": 0.0, "bbox": [0.0, 2.42, -2.76, 0.10], "radius": 0.5}
    poly = static_polygon(desk)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert (min(xs), round(max(xs), 2), round(min(ys), 2), round(max(ys), 2)) == (4.0, 6.42, 0.04, 2.9)
    turned = static_polygon({**desk, "yaw": 3.14159})
    assert round(min(p[0] for p in turned), 2) == 1.58 and round(max(p[0] for p in turned), 2) == 4.0


def test_the_desk_is_stamped_as_its_box() -> None:
    desk = {"pose": [4.0, 2.8], "yaw": 0.0, "bbox": [0.0, 2.42, -2.76, 0.10], "radius": 0.5}
    res, ox, oy = 0.1, 0.0, 0.0
    occupied = np.zeros((100, 100), dtype=bool)
    _stamp_statics(occupied, [desk], res, ox, oy)

    def cell(x, y):
        return occupied[occupied.shape[0] - 1 - int((y - oy) / res), int((x - ox) / res)]

    assert cell(5.0, 1.5) and cell(6.3, 0.2) and not cell(3.5, 1.5) and not cell(4.0, 4.0)


def test_the_worlds_statics_come_with_their_boxes() -> None:
    world = Path(__file__).resolve().parents[4] / "arena_simulation_setup" / "worlds" / "hospital_1"
    assert find_assets(world) == ASSETS
    statics = {s["name"]: s for s in world_statics(world)}
    desk = statics["reception_counter_1"]
    assert desk["bbox"] and round(desk["bbox"][1], 2) == 2.42 and desk["pose"] == [4.0, 2.8]
    with_box = sum(1 for s in statics.values() if "bbox" in s)
    assert with_box >= 0.9 * len(statics), f"{with_box} of {len(statics)} statics have a box"
