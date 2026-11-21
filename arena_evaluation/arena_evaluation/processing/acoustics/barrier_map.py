"""Acoustic barrier grid derived from a world's walls and doors."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

import numpy as np

from arena_evaluation.storage.exceptions import MetricInputError

_MICRO_PER_METRE = 1_000_000
_EDGE_TOLERANCE_M = 1e-9
_SLOPED_SAMPLES_PER_CELL = 8
_SUBSAMPLE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
_DOOR_MASK_MARGIN_CELLS = 2


@dataclasses.dataclass(frozen=True, slots=True)
class WallSegment:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclasses.dataclass(frozen=True, slots=True)
class DoorOpening:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float


@dataclasses.dataclass(frozen=True, slots=True)
class BarrierMap:
    """Barrier grid in solver orientation: row 0 is the lowest y, col 0 the lowest x."""

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float]
    door_masks: dict[str, np.ndarray]
    door_cells_across: dict[str, int]


def derive_barrier_map(
    *,
    walls: Iterable[WallSegment],
    doors: Sequence[DoorOpening],
    extent: Iterable[tuple[float, float]] = (),
    resolution: float = 0.05,
    padding_cells: int = 5,
) -> BarrierMap:
    """Derive the barrier grid. ``extent`` points (e.g. the corners of walkable zones) only widen the grid; they never block."""
    cell_um = _cell_micrometres(resolution)
    wall_list = list(walls)
    door_names = [door.name for door in doors]
    if len(set(door_names)) != len(door_names):
        raise MetricInputError(
            f"AcousticExposureCalculator: door names must be unique, got {sorted(door_names)}"
        )

    points_um = [
        *(_to_micrometres(point) for point in extent),
        *(
            _to_micrometres(point)
            for wall in wall_list
            for point in (wall.start, wall.end)
        ),
        *(_to_micrometres(corner) for door in doors for corner in _door_corners(door)),
    ]
    if not points_um:
        raise MetricInputError(
            "AcousticExposureCalculator: empty world (no walls, doors or extent to derive a barrier map from)"
        )
    col0 = min(x for x, _ in points_um) // cell_um - padding_cells
    row0 = min(y for _, y in points_um) // cell_um - padding_cells
    col1 = _ceil_div(max(x for x, _ in points_um), cell_um) + padding_cells
    row1 = _ceil_div(max(y for _, y in points_um), cell_um) + padding_cells

    barrier = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    for wall in wall_list:
        _mark_wall(barrier, wall, cell_um=cell_um, col0=col0, row0=row0)

    door_masks: dict[str, np.ndarray] = {}
    door_union = np.zeros(barrier.shape, dtype=bool)
    for door in doors:
        mask = _door_mask(barrier.shape, door, resolution, col0=col0, row0=row0)
        door_masks[door.name] = mask
        door_union |= mask
    barrier &= ~door_union

    _close_corner_leaks(barrier, protected=door_union)

    occupancy = barrier.astype(np.uint8)
    occupancy.flags.writeable = False
    for mask in door_masks.values():
        mask.flags.writeable = False

    return BarrierMap(
        occupancy=occupancy,
        resolution=resolution,
        origin=((col0 + 0.5) * resolution, (row0 + 0.5) * resolution),
        door_masks=door_masks,
        door_cells_across={
            door.name: _door_cells_across(
                door_masks[door.name], door, resolution, col0, row0
            )
            for door in doors
        },
    )


def _cell_micrometres(resolution: float) -> int:
    cell_um = round(resolution * _MICRO_PER_METRE)
    if cell_um <= 0 or abs(cell_um - resolution * _MICRO_PER_METRE) > 1e-6:
        raise MetricInputError(
            f"AcousticExposureCalculator: resolution {resolution} m must be a whole number of micrometres"
        )
    return cell_um


def _to_micrometres(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0] * _MICRO_PER_METRE), round(point[1] * _MICRO_PER_METRE)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _door_frame(door: DoorOpening) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Door start, unit direction, span length and half width; a zero span is invalid input geometry."""
    start = np.asarray(door.start, dtype=float)
    span = np.asarray(door.end, dtype=float) - start
    length = float(np.hypot(*span))
    if length == 0.0:
        raise MetricInputError(
            f"AcousticExposureCalculator: door {door.name!r} has zero span"
        )
    return start, span / length, length, door.width / 2


def _door_corners(door: DoorOpening) -> tuple[tuple[float, float], ...]:
    start, direction, length, half_width = _door_frame(door)
    across = np.array([-direction[1], direction[0]]) * half_width
    end = start + direction * length
    return tuple(
        tuple(float(value) for value in point)
        for point in (start + across, start - across, end - across, end + across)
    )


def _mark_wall(
    barrier: np.ndarray, wall: WallSegment, *, cell_um: int, col0: int, row0: int
) -> None:
    (x_a, y_a), (x_b, y_b) = _to_micrometres(wall.start), _to_micrometres(wall.end)
    if x_a == x_b:
        index = x_a // cell_um - col0
        rows = _half_open_range(min(y_a, y_b), max(y_a, y_b), cell_um, row0)
        barrier[rows, index] = True
    elif y_a == y_b:
        index = y_a // cell_um - row0
        cols = _half_open_range(min(x_a, x_b), max(x_a, x_b), cell_um, col0)
        barrier[index, cols] = True
    else:
        _mark_sloped_wall(
            barrier, (x_a, y_a), (x_b, y_b), cell_um=cell_um, col0=col0, row0=row0
        )


def _half_open_range(
    low_um: int, high_um: int, cell_um: int, origin_index: int
) -> slice:
    first = low_um // cell_um - origin_index
    stop = max(_ceil_div(high_um, cell_um), low_um // cell_um + 1) - origin_index
    return slice(first, stop)


def _mark_sloped_wall(
    barrier: np.ndarray,
    start_um: tuple[int, int],
    end_um: tuple[int, int],
    *,
    cell_um: int,
    col0: int,
    row0: int,
) -> None:
    steps = (
        max(abs(end_um[0] - start_um[0]), abs(end_um[1] - start_um[1]))
        // (cell_um // _SLOPED_SAMPLES_PER_CELL)
        + 1
    )
    for step in range(steps + 1):
        x_um = start_um[0] + (end_um[0] - start_um[0]) * step // steps
        y_um = start_um[1] + (end_um[1] - start_um[1]) * step // steps
        barrier[y_um // cell_um - row0, x_um // cell_um - col0] = True


def _door_mask(
    shape: tuple[int, int],
    door: DoorOpening,
    resolution: float,
    *,
    col0: int,
    row0: int,
) -> np.ndarray:
    """Full-grid door mask evaluated only on the door's padded bounding box.

    The box holds every cell whose sub-points can fall inside the door rectangle, so the mask is
    identical to the whole-grid predicate while a door touches a few hundred cells, not the grid.
    """
    mask = np.zeros(shape, dtype=bool)
    col_lo, col_hi, row_lo, row_hi = _door_cell_bounds(door, resolution)
    col_lo, col_hi = max(col_lo, col0), min(col_hi, col0 + shape[1])
    row_lo, row_hi = max(row_lo, row0), min(row_hi, row0 + shape[0])
    if col_lo >= col_hi or row_lo >= row_hi:
        return mask
    columns = np.arange(col_lo, col_hi)
    rows = np.arange(row_lo, row_hi)
    left = ((columns + 0.5) * resolution - resolution / 2)[np.newaxis, :]
    bottom = ((rows + 0.5) * resolution - resolution / 2)[:, np.newaxis]
    mask[row_lo - row0 : row_hi - row0, col_lo - col0 : col_hi - col0] = (
        _door_rectangle_mask(left, bottom, door, resolution)
    )
    return mask


def _door_cell_bounds(
    door: DoorOpening, resolution: float
) -> tuple[int, int, int, int]:
    """Half-open absolute column and row range holding every cell the door can mark."""
    corners = np.asarray(_door_corners(door), dtype=float)
    margin = _DOOR_MASK_MARGIN_CELLS * resolution + 2 * _EDGE_TOLERANCE_M
    return (
        math.floor((float(corners[:, 0].min()) - margin) / resolution),
        math.ceil((float(corners[:, 0].max()) + margin) / resolution) + 1,
        math.floor((float(corners[:, 1].min()) - margin) / resolution),
        math.ceil((float(corners[:, 1].max()) + margin) / resolution) + 1,
    )


def _door_rectangle_mask(
    left: np.ndarray, bottom: np.ndarray, door: DoorOpening, resolution: float
) -> np.ndarray:
    """A cell belongs to the door when any of its sub-points lies inside the door rectangle.

    Sub-points sit inside the cell, never on its edges, so a door edge on a cell edge is exact, and a door thinner than
    a cell still claims the cells that carry its wall line.
    """
    start, direction, length, half_width = _door_frame(door)
    mask = np.zeros((bottom.shape[0], left.shape[1]), dtype=bool)
    for fraction_x in _SUBSAMPLE_FRACTIONS:
        point_x = left + fraction_x * resolution
        for fraction_y in _SUBSAMPLE_FRACTIONS:
            point_y = bottom + fraction_y * resolution
            along = (point_x - start[0]) * direction[0] + (
                point_y - start[1]
            ) * direction[1]
            across = (point_y - start[1]) * direction[0] - (
                point_x - start[0]
            ) * direction[1]
            mask |= (
                (along >= -_EDGE_TOLERANCE_M)
                & (along <= length + _EDGE_TOLERANCE_M)
                & (np.abs(across) <= half_width + _EDGE_TOLERANCE_M)
            )
    return mask


def _door_cells_across(
    mask: np.ndarray, door: DoorOpening, resolution: float, col0: int, row0: int
) -> int:
    """How many door cells a path crosses when it passes straight through the door's midpoint."""
    start, direction, _length, _half_width = _door_frame(door)
    normal = np.array([-direction[1], direction[0]])
    middle = (start + np.asarray(door.end, dtype=float)) / 2
    reach = door.width / 2 + resolution
    offsets = np.arange(-reach, reach, resolution / _SLOPED_SAMPLES_PER_CELL)
    cols = np.floor((middle[0] + normal[0] * offsets) / resolution).astype(int) - col0
    rows = np.floor((middle[1] + normal[1] * offsets) / resolution).astype(int) - row0
    inside = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    crossed = {
        (row, col)
        for row, col in zip(rows[inside], cols[inside], strict=True)
        if mask[row, col]
    }
    return max(1, len(crossed))


def _close_corner_leaks(barrier: np.ndarray, *, protected: np.ndarray) -> None:
    """Close the free cell of every 2x2 checkerboard whose free pair touches only at the corner.

    Door cells are never closed; they are the openings the walls leave on purpose.
    """
    while True:
        south_west, south_east = barrier[:-1, :-1], barrier[:-1, 1:]
        north_west, north_east = barrier[1:, :-1], barrier[1:, 1:]
        close_north_east = (
            ~south_west & ~north_east & north_west & south_east & ~protected[1:, 1:]
        )
        close_south_east = (
            ~north_west & ~south_east & south_west & north_east & ~protected[:-1, 1:]
        )
        if not (close_north_east.any() or close_south_east.any()):
            return
        barrier[1:, 1:] |= close_north_east
        barrier[:-1, 1:] |= close_south_east
