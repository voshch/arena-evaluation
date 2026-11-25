"""Unit tests for the acoustic barrier grid derived from walls and doors (acoustics/barrier_map.py)."""

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
    return [WallSegment(start=corners[index], end=corners[(index + 1) % 4]) for index in range(4)]


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
            along = (point_x - start[0]) * direction[0] + (point_y - start[1]) * direction[1]
            across = (point_y - start[1]) * direction[0] - (point_x - start[0]) * direction[1]
            mask |= (along >= -_ORACLE_EDGE_TOLERANCE_M) & (along <= length + _ORACLE_EDGE_TOLERANCE_M) & (np.abs(across) <= half_width + _ORACLE_EDGE_TOLERANCE_M)
    return mask


def test_sloped_wall_marks_the_cells_it_crosses():
    barrier_map = derive_barrier_map(walls=[WallSegment(start=(0.0, 0.0), end=(1.0, 1.0))], doors=[])

    assert barrier_map.occupancy[PAD + 10, PAD + 10] == 1


def test_duplicate_door_names_are_rejected():
    door = DoorOpening(name="same", start=(0.5, 0.0), end=(0.6, 0.0), width=0.1)

    with pytest.raises(MetricInputError, match="unique"):
        derive_barrier_map(walls=_box(), doors=[door, door])


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
            doors.append(DoorOpening(name=f"d{row}_{col}", start=(x, y), end=(x + 1.2, y), width=0.1))
    return walls, doors


def _elapsed_derivation(walls: list[WallSegment], doors: list[DoorOpening]) -> float:
    started = time.perf_counter()
    derive_barrier_map(walls=walls, doors=doors)
    return time.perf_counter() - started
