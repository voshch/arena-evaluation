import numpy as np
import pytest
from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations


def test_acoustics_free_space():
    # 10x10 empty grid
    grid = np.zeros((10, 10), dtype=np.uint8)
    resolution = 1.0  # 1 meter per pixel for simplicity

    # Start at (0, 0)
    sx, sy = 0.0, 0.0

    # Target at (3, 4) -> true Euclidean distance = sqrt(3^2+4^2) = 5.0 m
    # Theta* recovers exact Euclidean geometry in open space (no staircase bias).
    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    expected_dist = np.sqrt(3.0**2 + 4.0**2)  # = 5.0 m (true Euclidean, not staircase)
    expected = 20.0 * np.log10(expected_dist)
    assert np.isclose(att[0], expected, atol=0.1)


def test_acoustics_one_wall():
    # 10x10 grid with a vertical wall at x=2
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[:, 2] = 255  # wall

    resolution = 1.0
    sx, sy = 0.0, 0.0

    tx = np.array([3.0], dtype=np.float32)
    ty = np.array([4.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    # The shortest path must cross the wall once.
    # With Theta* the path through the wall uses true Euclidean distance
    # (sqrt(3^2+4^2) = 5.0 m) rather than the old staircase distance.
    expected_dist = np.sqrt(3.0**2 + 4.0**2)  # = 5.0 m
    expected = 20.0 * np.log10(expected_dist) + 47.0
    assert np.isclose(att[0], expected, atol=0.5)  # slightly wider: wall forces grid step


def test_acoustics_pruning_dominance():
    # Grid where going around the wall is cheaper than going through it
    grid = np.zeros((10, 10), dtype=np.uint8)
    # Wall from y=0 to y=6 at x=2
    grid[0:7, 2] = 255

    resolution = 1.0
    sx, sy = 0.0, 3.0
    tx = np.array([4.0], dtype=np.float32)
    ty = np.array([3.0], dtype=np.float32)

    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=47.0, mic_distance=1.0)

    # Path 1: through the wall: distance = 4m, walls = 1. Cost = 20log10(5) + 47 = 60.97
    # Path 2: around the wall: dist is roughly 4 + 4 + 4 = 12m. Cost = 20log10(13) = 22.2
    # So going around is much cheaper! The solver should return the path around.

    # Manually calculate roughly the around distance: (0,3) -> (2,7) -> (4,3)
    # dist = sqrt(2^2 + 4^2) * 2 = 2 * sqrt(20) = 8.94

    # The actual solver might find a slightly different path on the 8-connected grid
    # But it must be < 40 dB
    assert att[0] < 40.0


def _field(grid, resolution, sx, sy, mic_distance, wall_tl=47.0):
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tx = np.ascontiguousarray(xx.ravel().astype(np.float32))
    ty = np.ascontiguousarray(yy.ravel().astype(np.float32))
    att = compute_attenuations(grid, resolution, sx, sy, tx, ty, wall_tl=wall_tl, mic_distance=mic_distance)
    return att.reshape(h, w)


def test_receiver_inside_mic_distance_costs_the_floor():
    grid = np.zeros((40, 40), dtype=np.uint8)
    tx = np.array([25.0, 23.0, 20.0], dtype=np.float32)
    ty = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    att = compute_attenuations(grid, 0.1, 20.0, 20.0, tx, ty, wall_tl=47.0, mic_distance=2.0)
    assert np.allclose(att, 20.0 * np.log10(2.0), atol=1e-4)


def test_wall_inside_mic_distance_adds_only_its_tl():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[:, 20] = 1
    tx = np.array([23.0], dtype=np.float32)
    ty = np.array([20.0], dtype=np.float32)
    att = compute_attenuations(grid, 0.1, 17.0, 20.0, tx, ty, wall_tl=47.0, mic_distance=1.0)
    assert np.isclose(att[0], 20.0 * np.log10(1.0) + 47.0, atol=1e-4)


def test_tied_costs_inside_floor_keep_shortest_paths_beyond_it():
    grid = np.zeros((40, 40), dtype=np.uint8)
    grid[18:23, 21] = 1
    floored = _field(grid, 0.1, 20.0, 20.0, mic_distance=1.0, wall_tl=1000.0)
    reference = _field(grid, 0.1, 20.0, 20.0, mic_distance=1e-3, wall_tl=1000.0)
    beyond = reference > 20.0 * np.log10(1.05)
    beyond &= reference < 100.0
    assert beyond.sum() > 1000
    assert np.abs(floored - reference)[beyond].max() < 0.25
