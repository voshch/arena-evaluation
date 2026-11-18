"""Unit tests for the acoustic barrier grid derived from walls and doors (acoustics/barrier_map.py).

Geometry is built on the 5 cm lattice, so every expected cell is exact. The grid pads each side by five cells, so a
wall at world x = 0 sits in column 5 and a wall at y = 0 in row 5.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena_evaluation.processing.acoustics.barrier_map import (
    DoorOpening,
    WallSegment,
    derive_barrier_map,
)
from arena_evaluation.storage.exceptions import MetricInputError

RES = 0.05
PAD = 5
UNIT_EXTENT = [(0.0, 0.0), (1.0, 1.0)]


def _box(size: float = 1.0) -> list[WallSegment]:
    corners = [(0.0, 0.0), (size, 0.0), (size, size), (0.0, size)]
    return [
        WallSegment(start=corners[index], end=corners[(index + 1) % 4])
        for index in range(4)
    ]


def test_vertical_wall_marks_its_column_on_the_high_side():
    barrier_map = derive_barrier_map(
        walls=[WallSegment(start=(1.0, 0.0), end=(1.0, 1.0))],
        doors=[],
        extent=UNIT_EXTENT,
    )

    assert barrier_map.occupancy[PAD : PAD + 20, PAD + 20].all()
    assert int(barrier_map.occupancy.sum()) == 20


def test_horizontal_wall_marks_its_row_on_the_high_side():
    barrier_map = derive_barrier_map(
        walls=[WallSegment(start=(0.0, 1.0), end=(1.0, 1.0))],
        doors=[],
        extent=UNIT_EXTENT,
    )

    assert barrier_map.occupancy[PAD + 20, PAD : PAD + 20].all()
    assert int(barrier_map.occupancy.sum()) == 20


def test_enclosed_space_is_free_and_the_exterior_is_free_air():
    barrier_map = derive_barrier_map(walls=_box(), doors=[])

    assert (
        int((barrier_map.occupancy[PAD + 1 : PAD + 20, PAD + 1 : PAD + 20] == 0).sum())
        == 19 * 19
    )
    assert barrier_map.occupancy[0, 0] == 0


def test_positions_inside_a_walled_room_round_to_free_cells():
    barrier_map = derive_barrier_map(walls=_box(), doors=[])
    ox, oy = barrier_map.origin
    rng = np.random.default_rng(0)
    xs, ys = rng.uniform(0.1, 0.9, 500), rng.uniform(0.1, 0.9, 500)

    cols = np.floor((xs - ox) / RES + 0.5).astype(int)
    rows = np.floor((ys - oy) / RES + 0.5).astype(int)

    assert not barrier_map.occupancy[rows, cols].any()


def test_origin_is_the_centre_of_the_first_cell():
    barrier_map = derive_barrier_map(walls=_box(), doors=[])

    assert barrier_map.origin == pytest.approx(
        ((0 - PAD + 0.5) * RES, (0 - PAD + 0.5) * RES)
    )


def test_sloped_wall_marks_the_cells_it_crosses():
    barrier_map = derive_barrier_map(
        walls=[WallSegment(start=(0.0, 0.0), end=(1.0, 1.0))], doors=[]
    )

    assert barrier_map.occupancy[PAD + 10, PAD + 10] == 1


def test_corner_closure_marks_the_outer_cell_of_a_checkerboard():
    # Two walls meeting at (1, 1) only through a corner: the cell beyond the corner must not leak.
    walls = [
        WallSegment(start=(0.0, 1.0), end=(1.0, 1.0)),
        WallSegment(start=(1.0, 1.0), end=(1.0, 0.0)),
    ]

    barrier_map = derive_barrier_map(walls=walls, doors=[])

    assert barrier_map.occupancy[PAD + 20, PAD + 20] == 1
    assert barrier_map.occupancy[PAD + 19, PAD + 19] == 0


def test_door_cells_are_free_and_thick_across_the_wall():
    walls = [
        WallSegment(start=(1.0, 0.0), end=(1.0, 0.4)),
        WallSegment(start=(1.0, 0.6), end=(1.0, 1.0)),
    ]
    door = DoorOpening(name="between", start=(1.0, 0.4), end=(1.0, 0.6), width=0.1)

    barrier_map = derive_barrier_map(walls=walls, doors=[door], extent=UNIT_EXTENT)
    mask = barrier_map.door_masks["between"]

    assert mask.sum() == 4 * 2
    assert not barrier_map.occupancy[mask].any()
    assert barrier_map.door_cells_across["between"] == 2
    assert barrier_map.occupancy[PAD : PAD + 8, PAD + 20].all()
    assert barrier_map.occupancy[PAD + 12 : PAD + 20, PAD + 20].all()


def test_duplicate_door_names_are_rejected():
    door = DoorOpening(name="same", start=(0.5, 0.0), end=(0.6, 0.0), width=0.1)

    with pytest.raises(ValueError, match="unique"):
        derive_barrier_map(walls=_box(), doors=[door, door])


def test_geometry_without_walls_or_doors_is_rejected():
    with pytest.raises(MetricInputError, match="no walls, doors or extent"):
        derive_barrier_map(walls=[], doors=[])


def test_door_with_zero_span_is_rejected():
    door = DoorOpening(name="dot", start=(0.5, 0.0), end=(0.5, 0.0), width=0.1)

    with pytest.raises(ValueError, match="zero span"):
        derive_barrier_map(walls=_box(), doors=[door])
