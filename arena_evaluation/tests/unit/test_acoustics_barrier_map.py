"""Unit tests for the acoustic barrier grid derived from walls and doors (acoustics/barrier_map.py).

Geometry is built on the 5 cm lattice, so every expected cell is exact. The grid pads each side by five cells, so a
wall at world x = 0 sits in column 5 and a wall at y = 0 in row 5.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from arena_evaluation.processing.acoustics.barrier_map import (
    BarrierMap,
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


_ORACLE_SUBSAMPLE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
_ORACLE_EDGE_TOLERANCE_M = 1e-9


def _oracle_door_mask(door: DoorOpening, barrier_map: BarrierMap) -> np.ndarray:
    """The pre-optimisation whole-grid predicate, kept as the equality oracle."""
    height, width = barrier_map.occupancy.shape
    resolution = barrier_map.resolution
    col0 = round((barrier_map.origin[0] - resolution / 2) / resolution)
    row0 = round((barrier_map.origin[1] - resolution / 2) / resolution)
    grid_x, grid_y = np.meshgrid(
        (np.arange(col0, col0 + width) + 0.5) * resolution,
        (np.arange(row0, row0 + height) + 0.5) * resolution,
    )
    start = np.asarray(door.start, dtype=float)
    span = np.asarray(door.end, dtype=float) - start
    length = float(np.hypot(*span))
    direction = span / length
    half_width = door.width / 2
    mask = np.zeros(grid_x.shape, dtype=bool)
    left, bottom = grid_x - resolution / 2, grid_y - resolution / 2
    for fraction_x in _ORACLE_SUBSAMPLE_FRACTIONS:
        for fraction_y in _ORACLE_SUBSAMPLE_FRACTIONS:
            point_x = left + fraction_x * resolution
            point_y = bottom + fraction_y * resolution
            along = (point_x - start[0]) * direction[0] + (
                point_y - start[1]
            ) * direction[1]
            across = (point_y - start[1]) * direction[0] - (
                point_x - start[0]
            ) * direction[1]
            mask |= (
                (along >= -_ORACLE_EDGE_TOLERANCE_M)
                & (along <= length + _ORACLE_EDGE_TOLERANCE_M)
                & (np.abs(across) <= half_width + _ORACLE_EDGE_TOLERANCE_M)
            )
    return mask


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

    with pytest.raises(MetricInputError, match="unique"):
        derive_barrier_map(walls=_box(), doors=[door, door])


def test_geometry_without_walls_or_doors_is_rejected():
    with pytest.raises(MetricInputError, match="no walls, doors or extent"):
        derive_barrier_map(walls=[], doors=[])


def test_door_with_zero_span_is_rejected():
    door = DoorOpening(name="dot", start=(0.5, 0.0), end=(0.5, 0.0), width=0.1)

    with pytest.raises(MetricInputError, match="zero span"):
        derive_barrier_map(walls=_box(), doors=[door])


def test_resolution_not_a_whole_number_of_micrometres_is_rejected():
    with pytest.raises(MetricInputError, match="whole number of micrometres"):
        derive_barrier_map(walls=_box(), doors=[], resolution=1.0 / 3.0)


_DOOR_GEOMETRIES = (
    ((0.5, 0.5), (0.5, 1.5), 0.1),
    ((0.0, 1.0), (2.0, 1.0), 0.1),
    ((0.2, 0.2), (1.8, 1.4), 0.1),
    ((0.4, 0.8), (1.6, 1.0), 0.04),
    ((0.8, 0.6), (1.2, 1.4), 0.5),
    ((0.0, 0.0), (2.0, 2.0), 0.2),
)


@pytest.mark.parametrize(("start", "end", "width"), _DOOR_GEOMETRIES)
def test_bounded_door_mask_matches_the_full_grid_oracle(start, end, width):
    door = DoorOpening(name="d", start=start, end=end, width=width)
    barrier_map = derive_barrier_map(
        walls=_box(2.0), doors=[door], extent=[(0.0, 0.0), (2.0, 2.0)]
    )

    assert barrier_map.door_masks["d"].any()
    assert np.array_equal(
        barrier_map.door_masks["d"], _oracle_door_mask(door, barrier_map)
    )


def _large_world() -> tuple[list[WallSegment], list[DoorOpening]]:
    """A 300 x 300-cell world (15 m at 5 cm) with five interior walls and twenty doors."""
    size = 15.0
    walls = _box(size)
    doors: list[DoorOpening] = []
    for row in range(5):
        y = 2.0 + row * 2.5
        walls.append(WallSegment(start=(0.0, y), end=(size, y)))
        for col in range(4):
            x = 1.5 + col * 3.5
            doors.append(
                DoorOpening(
                    name=f"d{row}_{col}", start=(x, y), end=(x + 1.2, y), width=0.1
                )
            )
    return walls, doors


def _elapsed_derivation(walls: list[WallSegment], doors: list[DoorOpening]) -> float:
    started = time.perf_counter()
    derive_barrier_map(walls=walls, doors=doors)
    return time.perf_counter() - started


def test_door_mask_derivation_is_fast_on_a_large_world():
    walls, doors = _large_world()

    # Best of three wall-clock runs so a loaded CI box does not flake the coarse bound.
    elapsed_s = min(_elapsed_derivation(walls, doors) for _ in range(3))
    barrier_map = derive_barrier_map(walls=walls, doors=doors)

    # 300 x 300 cells at 5 cm, widened by the five-cell padding on each side.
    assert barrier_map.occupancy.shape == (310, 310)
    assert len(barrier_map.door_masks) == 20
    assert elapsed_s < 2.0, f"barrier derivation took {elapsed_s:.2f} s"
    for door in doors[:2]:
        assert np.array_equal(
            barrier_map.door_masks[door.name], _oracle_door_mask(door, barrier_map)
        )
