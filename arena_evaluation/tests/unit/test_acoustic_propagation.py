#!/usr/bin/env python3
"""Propagation physics tests for the acoustic Dijkstra solver.

Builds scenario grids (see acoustic_scenarios.py) and asserts the solver
respects the expected free-field / barrier physics:

  - inverse-square law: +6 dB per doubling of distance (20*log10, point source)
  - open door == line of sight through the doorway
  - closed door / solid wall == barrier TL (~47 dB) on the straight path
  - side-placed door == detour path, attenuation between LOS and blocked
  - narrowing corridor == more attenuation than a uniform corridor
"""

from __future__ import annotations

import pathlib
import numpy as np
import pytest

from arena_evaluation.processing.acoustics.impedance_grid import compute_attenuations

sys_path = pathlib.Path(__file__).resolve().parents[1] / "unit"  # allow local import in-tree
import sys  # noqa: E402

if str(sys_path) not in sys.path:
    sys.path.insert(0, str(sys_path))

from acoustic_scenarios import scenarios, build, door_pixels, RES, WALL_TL, MIC_DIST  # noqa: E402

DOOR_TL = 25.0  # closed-door transmission loss (lighter than the 47 dB wall)

WALL = 1


def _tl_map(grid, spec, open_doors=False):
    """Per-pixel TL map: walls 47 dB; door pixels DOOR_TL (or 0 when open)."""
    tl = np.where(grid == WALL, WALL_TL, 0.0).astype(np.float32)
    for y, x in door_pixels(spec):
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            tl[y, x] = 0.0 if open_doors else DOOR_TL
    return np.ascontiguousarray(tl)


def _solve(grid, start_m, target_m, wall_tl=WALL_TL, mic=MIC_DIST, pixel_tl=None) -> float:
    sx, sy = start_m
    tx, ty = target_m
    att = compute_attenuations(
        grid,
        RES,
        sx / RES,
        sy / RES,
        np.array([tx / RES], np.float32),
        np.array([ty / RES], np.float32),
        wall_tl=wall_tl,
        mic_distance=mic,
        pixel_tl=pixel_tl,
    )
    return float(att[0])


def _los_db(start_m, target_m, mic=MIC_DIST) -> float:
    d = np.hypot(target_m[0] - start_m[0], target_m[1] - start_m[1])
    return 20.0 * np.log10(d + mic)


# Inverse-square law (the 3 dB / 6 dB doubling rules)


def test_free_field_six_db_per_doubling():
    """Point-source spherical spreading: 2x distance = +6.02 dB (2x pressure, 4x energy)."""
    grid = np.zeros((400, 400), dtype=np.uint8)
    a1 = _solve(grid, (1.0, 20.0), (11.0, 20.0))  # d = 10 m
    a2 = _solve(grid, (1.0, 20.0), (21.0, 20.0))  # d = 20 m
    expect = 20.0 * np.log10((20.0 + 1.0) / (10.0 + 1.0))  # mic-corrected doubling
    assert np.isclose(a2 - a1, expect, atol=0.15), f"{a2 - a1:.2f} vs {expect:.2f} dB"


def test_free_field_three_db_double_energy():
    """+3 dB == double acoustic energy (10*log10(2)); distance ratio ~sqrt(2).
    Uses mic=0.01 so the +mic term is negligible and pixel-exact distances
    (5 m vs 7 m, ratio 1.4) approximate the sqrt(2) ratio."""
    grid = np.zeros((400, 400), dtype=np.uint8)
    a1 = _solve(grid, (1.0, 20.0), (6.0, 20.0), mic=0.01)  # d = 5 m
    a2 = _solve(grid, (1.0, 20.0), (8.0, 20.0), mic=0.01)  # d = 7 m
    expect = 20.0 * np.log10(7.0 / 5.0)
    assert np.isclose(a2 - a1, expect, atol=0.15), f"{a2 - a1:.2f} vs {expect:.2f} dB"


def test_free_field_matches_closed_form():
    grid = np.zeros((400, 400), dtype=np.uint8)
    for d_m in (3.0, 5.0, 8.0):
        att = _solve(grid, (1.0, 20.0), (1.0 + d_m, 20.0))
        assert np.isclose(att, _los_db((1.0, 20.0), (1.0 + d_m, 20.0)), atol=0.15)


# Two rooms, door middle


def test_open_door_middle_is_line_of_sight():
    spec = scenarios()["two_rooms_door_open_middle"]
    att = _solve(build({**spec, "_name": "two_rooms_door_open_middle"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    assert att < los + 2.0, f"open door should be ~LOS: {att:.1f} vs {los:.1f}"


def test_closed_door_middle_adds_wall_tl():
    spec = scenarios()["two_rooms_door_closed_middle"]
    att = _solve(build({**spec, "_name": "two_rooms_door_closed_middle"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    # straight path crosses the wall once -> ~+47 dB (solver may find a detour,
    # so assert the barrier dominates: >= LOS + 30 dB and > open-door result)
    assert att >= los + 30.0, f"closed door must block: {att:.1f} vs LOS {los:.1f}"
    open_spec = scenarios()["two_rooms_door_open_middle"]
    att_open = _solve(build({**open_spec, "_name": "two_rooms_door_open_middle"}), open_spec["start"], open_spec["target"])
    assert att > att_open + 25.0, f"closed vs open door: {att:.1f} vs {att_open:.1f}"


# Two rooms, door side


def test_side_door_is_detour():
    """Side-placed door forces a longer path: attenuation between LOS and blocked."""
    spec = scenarios()["two_rooms_door_open_side"]
    att = _solve(build({**spec, "_name": "two_rooms_door_open_side"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    # detour adds path length (small for 10 m rooms with a corner door)
    assert att > los + 0.15, f"side door should detour: {att:.1f} vs LOS {los:.1f}"
    # ... but well below a full wall crossing
    assert att < los + WALL_TL - 10.0, f"side door should not cost full TL: {att:.1f}"


def test_side_door_closed_blocks_like_wall():
    spec = scenarios()["two_rooms_door_closed_side"]
    att = _solve(build({**spec, "_name": "two_rooms_door_closed_side"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    assert att >= los + 30.0


# Two rooms, wall only


def test_wall_only_blocks():
    spec = scenarios()["two_rooms_wall_only"]
    att = _solve(build({**spec, "_name": "two_rooms_wall_only"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    assert att >= los + 30.0, f"solid wall must block: {att:.1f}"


def test_wall_only_harder_than_open_door():
    wall = scenarios()["two_rooms_wall_only"]
    door = scenarios()["two_rooms_door_open_middle"]
    a_wall = _solve(build({**wall, "_name": "two_rooms_wall_only"}), wall["start"], wall["target"])
    a_door = _solve(build({**door, "_name": "two_rooms_door_open_middle"}), door["start"], door["target"])
    assert a_wall > a_door + 25.0, f"{a_wall:.1f} vs {a_door:.1f}"


# Corridors


def test_corridor_line_of_sight():
    spec = scenarios()["long_corridor"]
    att = _solve(build({**spec, "_name": "long_corridor"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    assert np.isclose(att, los, atol=2.0), f"corridor should be ~LOS: {att:.1f} vs {los:.1f}"


def test_corridor_narrowing_increases_attenuation():
    narrow = scenarios()["corridor_narrowing"]
    plain = scenarios()["long_corridor"]
    a_narrow = _solve(build({**narrow, "_name": "corridor_narrowing"}), narrow["start"], narrow["target"])
    a_plain = _solve(build({**plain, "_name": "long_corridor"}), plain["start"], plain["target"])
    assert a_narrow > a_plain, f"narrowing must cost more: {a_narrow:.1f} vs {a_plain:.1f}"


def test_corridor_closed_door_blocks():
    spec = scenarios()["corridor_side_door"]
    att = _solve(build({**spec, "_name": "corridor_side_door"}), spec["start"], spec["target"])
    los = _los_db(spec["start"], spec["target"])
    assert att >= los + 30.0, f"closed corridor door must block: {att:.1f}"


# Door vs wall dB distinction (per-pixel TL)


def test_closed_door_cheaper_than_wall():
    """A closed door (25 dB) attenuates LESS than a solid wall (47 dB)."""
    door = scenarios()["two_rooms_door_closed_middle"]
    wall = scenarios()["two_rooms_wall_only"]
    g_door = build({**door, "_name": "two_rooms_door_closed_middle"})
    g_wall = build({**wall, "_name": "two_rooms_wall_only"})
    a_door = _solve(g_door, door["start"], door["target"], pixel_tl=_tl_map(g_door, door))
    a_wall = _solve(g_wall, wall["start"], wall["target"], pixel_tl=_tl_map(g_wall, wall))
    los = _los_db(door["start"], door["target"])
    assert np.isclose(a_door, los + DOOR_TL, atol=2.0), f"door: {a_door:.1f} vs {los + DOOR_TL:.1f}"
    assert np.isclose(a_wall, los + WALL_TL, atol=2.0), f"wall: {a_wall:.1f} vs {los + WALL_TL:.1f}"
    assert a_door < a_wall - 15.0, f"door {a_door:.1f} must be cheaper than wall {a_wall:.1f}"


def test_open_door_equals_free_space():
    """An OPEN door (TL 0) is acoustically free space."""
    spec = scenarios()["two_rooms_door_open_middle"]
    grid = build({**spec, "_name": "two_rooms_door_open_middle"})
    a_open = _solve(grid, spec["start"], spec["target"], pixel_tl=_tl_map(grid, spec, open_doors=True))
    a_plain = _solve(np.zeros_like(grid), spec["start"], spec["target"])
    assert np.isclose(a_open, a_plain, atol=0.5), f"{a_open:.1f} vs {a_plain:.1f}"


def test_door_state_flips_attenuation():
    """Same geometry, door open vs closed, must differ by ~DOOR_TL."""
    spec = scenarios()["two_rooms_door_open_middle"]
    grid = build({**spec, "_name": "two_rooms_door_open_middle"})
    a_open = _solve(grid, spec["start"], spec["target"], pixel_tl=_tl_map(grid, spec, open_doors=True))
    a_closed = _solve(grid, spec["start"], spec["target"], pixel_tl=_tl_map(grid, spec, open_doors=False))
    assert np.isclose(a_closed - a_open, DOOR_TL, atol=2.0), f"{a_closed - a_open:.1f}"


# Wall TL sensitivity


def test_wall_tl_scales_linearly():
    """A single wall crossing should scale 1:1 with the wall_tl parameter."""
    spec = scenarios()["two_rooms_door_closed_middle"]
    grid = build({**spec, "_name": "two_rooms_door_closed_middle"})
    start, target = spec["start"], spec["target"]
    a30 = _solve(grid, start, target, wall_tl=30.0)
    a47 = _solve(grid, start, target, wall_tl=47.0)
    a60 = _solve(grid, start, target, wall_tl=60.0)
    assert np.isclose(a47 - a30, 17.0, atol=1.5), f"{a47 - a30:.1f}"
    assert np.isclose(a60 - a47, 13.0, atol=1.5), f"{a60 - a47:.1f}"


# Semantic-door integration (see plan): carve door pixels per state


def test_door_carving_open_vs_closed():
    """Semantic door integration: door state flips the per-pixel TL in place."""
    spec = scenarios()["two_rooms_door_open_middle"]
    grid = build({**spec, "_name": "two_rooms_door_open_middle"})
    start, target = spec["start"], spec["target"]
    a_open = _solve(grid, start, target, pixel_tl=_tl_map(grid, spec, open_doors=True))
    a_closed = _solve(grid, start, target, pixel_tl=_tl_map(grid, spec, open_doors=False))
    los = _los_db(start, target)
    assert a_open < los + 2.0, f"open door should be ~LOS: {a_open:.1f}"
    assert a_closed >= los + 20.0, f"closed door should block: {a_closed:.1f}"
