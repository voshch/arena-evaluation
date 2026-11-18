from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from arena_evaluation.processing.path import static_environment
from arena_evaluation.processing.path.static_environment import build_raster_grid, raster_grid_for_map


def _single_occupied_cell(radius_m: float):
    occupied = np.zeros((9, 9), dtype=bool)
    occupied[4, 4] = True
    return build_raster_grid(occupied, origin_xy=(0.0, 0.0), resolution_m=0.1, robot_radius_m=radius_m)


def test_raster_lethal_cells_are_the_centres_within_the_radius_of_the_occupied_cell():
    grid = _single_occupied_cell(0.25)
    assert grid.lethal.sum() == 21
    assert not grid.lethal[2, 2]
    assert grid.lethal[2, 3]


@pytest.mark.parametrize(("cell", "lethal"), [((4, 2), True), ((2, 2), False), ((4, 6), True), ((6, 6), False)])
def test_raster_lethal_mask_includes_the_centre_exactly_one_radius_away(cell: tuple[int, int], lethal: bool):
    assert bool(_single_occupied_cell(0.2).lethal[cell]) is lethal


def test_raster_without_occupied_cells_has_no_lethal_cell():
    grid = build_raster_grid(np.zeros((5, 5), dtype=bool), origin_xy=(0.0, 0.0), resolution_m=0.1, robot_radius_m=0.3)
    assert not grid.lethal.any()


def test_raster_refuses_a_non_positive_radius():
    with pytest.raises(ValueError, match="robot radius must be positive"):
        build_raster_grid(np.zeros((3, 3), dtype=bool), origin_xy=(0.0, 0.0), resolution_m=0.1, robot_radius_m=0.0)


def test_unnamed_map_has_no_raster_grid():
    assert raster_grid_for_map(None, None, 0.3) is None


def test_map_without_an_occupancy_image_is_an_error(monkeypatch):
    monkeypatch.setattr(static_environment.MapRegistry, "get_map", lambda map_name, run_dir=None: None)
    with pytest.raises(ValueError, match="has no occupancy image"):
        raster_grid_for_map("raster_probe_no_image", None, 0.3)


def test_raster_grid_flips_the_image_into_the_map_frame_and_is_cached(monkeypatch, tmp_path):
    image = np.full((6, 8), 255, dtype=np.uint8)
    image[0, 1] = 0
    png = tmp_path / "probe.png"
    Image.fromarray(image).save(png)
    monkeypatch.setattr(
        static_environment.MapRegistry,
        "get_map",
        lambda map_name, run_dir=None: {"png_path": str(png), "resolution": 0.1, "origin": [1.0, 2.0, 0.0]},
    )
    first = raster_grid_for_map("raster_probe_flip", None, 0.1)
    assert first is not None
    assert first.origin_xy == (1.0, 2.0)
    assert first.resolution_m == 0.1
    assert first.lethal[5, 1]
    assert not first.lethal[0, 1]
    assert raster_grid_for_map("raster_probe_flip", None, 0.1) is first
