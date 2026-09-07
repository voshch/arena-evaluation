#!/usr/bin/env python3
"""Theta* solver verification tests.

These tests assert properties that *only* hold with Theta* any-angle path
planning and would fail (or be meaningless) against the old 8-connected
Dijkstra:

  1. Radial symmetry / circular wavefront:
       In an empty grid, attenuation must equal 20*log10(r) within tight
       tolerance for *all* directions, not just the 8 grid-aligned ones.

  2. True Euclidean distance recovery:
       The solver must return exactly 20*log10(d + mic) (closed-form) for
       unobstructed point pairs.  The old staircase distance must NOT appear.

  3. Single-wall crossing:
       After one air->wall transition the cost must be
       20*log10(euclidean_dist + mic) + wall_TL within tolerance.

  4. Multi-path wall routing:
       The solver prefers the shortest *total-cost* path (around vs through).

  5. No staircase regression:
       For pixels at non-45-degree angles the old Dijkstra bias is > 0.3 dB;
       Theta* must stay within 0.15 dB of the closed-form value.

Reference: Daniel et al. (2010), "Theta*: Any-angle path planning on grids",
Journal of Artificial Intelligence Research, 39, 533-579.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RES = 0.05  # 5 cm/px  (matches real maps)
MIC = 0.01  # very small so 20*log10(d + mic) ≈ 20*log10(d)
WALL_TL = 47.0


def _att(grid, sx_px, sy_px, tx_px, ty_px, *, res=RES, mic=MIC, wall_tl=WALL_TL, pixel_tl=None) -> float:
    """Single-target convenience wrapper (pixel-space coordinates)."""
    tx = np.array([float(tx_px)], dtype=np.float32)
    ty = np.array([float(ty_px)], dtype=np.float32)
    result = compute_attenuations(
        grid,
        res,
        float(sx_px),
        float(sy_px),
        tx,
        ty,
        wall_tl=wall_tl,
        mic_distance=mic,
        pixel_tl=pixel_tl,
    )
    return float(result[0])


def _expected_db(dx_px, dy_px, *, res=RES, mic=MIC) -> float:
    """Expected attenuation for a straight unobstructed path (dB)."""
    d = np.sqrt(dx_px**2 + dy_px**2) * res
    return 20.0 * np.log10(d + mic)


# ---------------------------------------------------------------------------
# 1. Radial symmetry (circular wavefront)
# ---------------------------------------------------------------------------


class TestRadialSymmetry:
    """In a 50x50 empty grid with source at centre, all pixels at the same
    Euclidean radius must have attenuation within 0.15 dB of 20*log10(r + mic).
    """

    @pytest.fixture(scope="class")
    def att_grid(self):
        grid = np.zeros((50, 50), dtype=np.uint8)
        cx, cy = 25.0, 25.0
        yy, xx = np.mgrid[0:50, 0:50]
        tx = xx.flatten().astype(np.float32)
        ty = yy.flatten().astype(np.float32)
        att = compute_attenuations(grid, RES, cx, cy, tx, ty, wall_tl=WALL_TL, mic_distance=MIC)
        return att.reshape(50, 50), cx, cy

    def test_max_error_below_threshold(self, att_grid):
        """Max error across all pixels must be < 0.15 dB."""
        att, cx, cy = att_grid
        yy, xx = np.mgrid[0:50, 0:50]
        dx = (xx - cx) * RES
        dy = (yy - cy) * RES
        dist = np.sqrt(dx**2 + dy**2)
        expected = 20.0 * np.log10(dist + MIC)
        # exclude source pixel itself (dist=0 -> -inf expected)
        mask = dist > 0.5 * RES
        err = np.abs(att[mask] - expected[mask])
        assert np.nanmax(err) < 0.15, f"Max radial error {np.nanmax(err):.3f} dB exceeds 0.15 dB — octagonal wavefront artifact still present"

    def test_diagonal_vs_cardinal_symmetry(self, att_grid):
        """Diagonal pixels must be within 0.1 dB of equal-radius cardinal pixels."""
        att, cx, cy = att_grid
        cx_i, cy_i = int(cx), int(cy)
        r = 10  # 10 px = 0.5 m
        # Cardinal: (cx+r, cy)
        a_card = att[cy_i, cx_i + r]
        # Diagonal: (cx + r/√2, cy + r/√2) — closest integer
        d = int(round(r / np.sqrt(2)))
        a_diag = att[cy_i + d, cx_i + d]
        # Both are r pixels from centre; expected values differ slightly
        exp_card = _expected_db(r, 0)
        exp_diag = _expected_db(d, d)
        assert abs(a_card - exp_card) < 0.10, f"Cardinal error {a_card - exp_card:.3f} dB"
        assert abs(a_diag - exp_diag) < 0.10, f"Diagonal error {a_diag - exp_diag:.3f} dB"

    def test_off_axis_accuracy(self, att_grid):
        """Pixels in non-45° off-axis directions must also be accurate (regression for
        the old Dijkstra bias which was worst at ~22.5°)."""
        att, cx, cy = att_grid
        cx_i, cy_i = int(cx), int(cy)
        # ~18.4° direction: (dx=15, dy=5)
        dx, dy = 15, 5
        a = att[cy_i + dy, cx_i + dx]
        exp = _expected_db(dx, dy)
        assert abs(a - exp) < 0.15, f"Off-axis (dx={dx}, dy={dy}) error {a - exp:.3f} dB — staircase bias still present"


# ---------------------------------------------------------------------------
# 2. True Euclidean distance recovery (no staircase)
# ---------------------------------------------------------------------------


class TestEuclideanRecovery:
    """Theta* must return 20*log10(euclidean + mic), not the staircase distance."""

    @pytest.mark.parametrize(
        "dx,dy",
        [
            (3, 4),  # 3-4-5 triangle, 5 m at res=1 m/px
            (5, 12),  # 5-12-13 triangle
            (8, 15),  # 8-15-17 triangle
            (7, 3),  # off-axis
            (10, 6),  # off-axis
        ],
    )
    def test_euclidean_distance_1m_per_px(self, dx, dy):
        """At 1 m/px resolution, solver must recover true Euclidean in open space."""
        res = 1.0
        mic = 0.01
        size = max(dx, dy) + 5
        grid = np.zeros((size, size), dtype=np.uint8)
        att = _att(grid, 0, 0, dx, dy, res=res, mic=mic)
        expected = 20.0 * np.log10(np.sqrt(dx**2 + dy**2) + mic)
        assert np.isclose(att, expected, atol=0.1), f"dx={dx} dy={dy}: got {att:.3f} dB, expected {expected:.3f} dB"

    def test_staircase_distance_not_returned(self):
        """Verify that the OLD staircase distance (3√2 + 1 = 5.243) is NOT used
        for the (3,4) pixel at 1 m/px — Theta* must give √(9+16) = 5.0 m."""
        res = 1.0
        mic = 0.01
        grid = np.zeros((10, 10), dtype=np.uint8)
        att = _att(grid, 0, 0, 3, 4, res=res, mic=mic)
        staircase_db = 20.0 * np.log10(3 * np.sqrt(2) + 1 + mic)  # old Dijkstra
        euclidean_db = 20.0 * np.log10(5.0 + mic)  # Theta*
        # Must be much closer to Euclidean than to staircase
        assert abs(att - euclidean_db) < abs(att - staircase_db), f"Solver returned {att:.3f} dB, staircase={staircase_db:.3f}, euclidean={euclidean_db:.3f} — Dijkstra bias still present"


# ---------------------------------------------------------------------------
# 3. Single wall crossing with Euclidean distance
# ---------------------------------------------------------------------------


class TestSingleWallCrossing:
    """Vertical wall at column 25; source left, target right.

    The straight path crosses one wall pixel.  Expected cost:
        20*log10(euclidean_dist + mic) + wall_TL
    where euclidean_dist is the pixel-space straight-line distance.
    """

    @pytest.fixture
    def scenario(self):
        W, H = 50, 50
        grid = np.zeros((H, W), dtype=np.uint8)
        grid[:, 25] = 1  # vertical wall
        return grid, W, H

    def test_wall_crossing_horizontal(self, scenario):
        """Horizontal crossing: source (5,25), target (45,25) — 40 px = 2 m."""
        grid, W, H = scenario
        att = _att(grid, 5, 25, 45, 25)
        expected = _expected_db(40, 0) + WALL_TL
        assert np.isclose(att, expected, atol=0.3), f"Horizontal wall crossing: {att:.2f} vs {expected:.2f} dB"

    def test_wall_crossing_diagonal(self, scenario):
        """Diagonal crossing: source (5,10), target (45,40).
        Must pay wall_TL once; Euclidean distance preserved."""
        grid, W, H = scenario
        dx, dy = 40, 30  # 40-30-50 triangle → 50 px = 2.5 m
        att = _att(grid, 5, 10, 45, 40)
        expected = _expected_db(dx, dy) + WALL_TL
        # Diagonal crossing forces grid steps near the wall, so allow wider tolerance
        assert np.isclose(att, expected, atol=1.0), f"Diagonal wall crossing: {att:.2f} vs {expected:.2f} dB"

    def test_wall_tl_paid_exactly_once(self, scenario):
        """Crossing the same wall multiple times on a loop should not happen — but
        verify TL is paid once: difference between wall and free-space == WALL_TL."""
        grid, W, H = scenario
        free_grid = np.zeros((H, W), dtype=np.uint8)
        att_wall = _att(grid, 5, 25, 45, 25)
        att_free = _att(free_grid, 5, 25, 45, 25)
        assert np.isclose(att_wall - att_free, WALL_TL, atol=0.3), f"Wall TL contribution: {att_wall - att_free:.2f} dB, expected {WALL_TL}"


# ---------------------------------------------------------------------------
# 4. Around-wall routing (multi-path pruning)
# ---------------------------------------------------------------------------


class TestMultiPathRouting:
    """Wall from y=0..6 at x=2; source (0,3), target (4,3).
    Going through the wall costs 20*log10(4+mic) + 47 ≈ 60 dB.
    Going around costs 20*log10(~9+mic) ≈ 20 dB.  Solver must pick the detour.
    """

    def test_prefers_detour_over_wall(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[0:7, 2] = 255  # partial wall

        att = _att(grid, 0, 3, 4, 3, res=1.0, mic=1.0)
        # Straight through: ~60 dB;  around: ~22 dB
        assert att < 40.0, f"Should prefer detour: {att:.1f} dB"

    def test_detour_cheaper_than_wall(self):
        grid_wall = np.zeros((10, 10), dtype=np.uint8)
        grid_wall[:, 5] = 1  # full wall (forced through)
        grid_free = np.zeros((10, 10), dtype=np.uint8)

        att_through = _att(grid_wall, 0, 5, 9, 5, res=1.0, mic=1.0)
        att_around = _att(grid_free, 0, 5, 9, 5, res=1.0, mic=1.0)
        assert att_through > att_around + 30.0, f"Wall must cost more: through={att_through:.1f}, free={att_around:.1f}"


# ---------------------------------------------------------------------------
# 5. No staircase regression across multiple off-axis directions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle_deg,r_px",
    [
        (10, 15),
        (22, 15),  # worst-case for old 8-connected bias
        (30, 15),
        (60, 15),
        (80, 15),
    ],
)
def test_no_staircase_bias_by_angle(angle_deg, r_px):
    """For each angle, the error vs closed-form must be < 0.15 dB.

    Old Dijkstra had errors up to ~0.8 dB at 22.5°; Theta* should fix all angles.
    """
    cx, cy = 25, 25
    dx = int(round(r_px * np.cos(np.radians(angle_deg))))
    dy = int(round(r_px * np.sin(np.radians(angle_deg))))
    size = 60
    grid = np.zeros((size, size), dtype=np.uint8)
    att = _att(grid, cx, cy, cx + dx, cy + dy)
    expected = _expected_db(dx, dy)
    assert abs(att - expected) < 0.15, f"angle={angle_deg}° r={r_px}px: error {att - expected:.3f} dB (staircase bias still present)"
