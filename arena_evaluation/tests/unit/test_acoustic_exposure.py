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
    expected = 20.0 * np.log10(expected_dist + 1.0)
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
    # Cost = 20*log10(5.0 + 1.0) + 47.0
    expected_dist = np.sqrt(3.0**2 + 4.0**2)  # = 5.0 m
    expected = 20.0 * np.log10(expected_dist + 1.0) + 47.0
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
    expected_around_cost = 20.0 * np.log10(8.94 + 1.0)

    # The actual solver might find a slightly different path on the 8-connected grid
    # But it must be < 40 dB
    assert att[0] < 40.0
