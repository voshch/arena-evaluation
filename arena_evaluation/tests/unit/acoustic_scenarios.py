#!/usr/bin/env python3
"""Acoustic propagation test scenarios: programmatic occupancy grids + PNG exports.

Each scenario is a 2D uint8 grid (0 = free, 1 = wall) at RES=0.1 m/px, plus a
short description of the expected physics so the pytest suite can assert on it.

Scenarios:
  - two_rooms_door_open_middle    door gap in the middle of the shared wall (open)
  - two_rooms_door_closed_middle  same wall, door pixels blocked (closed)
  - two_rooms_door_open_side      door gap at the side of the shared wall
  - two_rooms_door_closed_side    same wall, side door blocked
  - two_rooms_wall_only           solid shared wall, no door
  - long_corridor                 straight corridor, source at one end
  - corridor_narrowing            corridor that narrows mid-way
  - corridor_side_door            corridor with a closed door splitting it
"""

from __future__ import annotations

import pathlib
import numpy as np
from PIL import Image

RES = 0.1  # meters per pixel
ROOM = 100  # 10 m x 10 m rooms (px)
WALL_TL = 47.0
MIC_DIST = 1.0

FREE, WALL = 0, 1


def _room_pair(door_gap: tuple[int, int] | None) -> np.ndarray:
    """Two ROOMxROOM rooms side by side, shared vertical wall at x=ROOM.
    door_gap = (y_start, y_end) pixel slice of free space, or None for solid wall."""
    grid = np.zeros((ROOM, 2 * ROOM + 1), dtype=np.uint8)
    grid[:, ROOM] = WALL  # shared wall
    if door_gap is not None:
        y0, y1 = door_gap
        grid[y0:y1, ROOM] = FREE  # carve the door
    return grid


def _corridor(width_px: int, length_px: int) -> np.ndarray:
    """Horizontal corridor of given width/height, open at both ends."""
    grid = np.zeros((width_px, length_px), dtype=np.uint8)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    return grid


def door_pixels(spec: dict) -> list[tuple[int, int]]:
    """Pixel coordinates of the door (shared-wall gap) for the two_rooms scenarios.

    For door-open variants these are the carved free pixels; for closed/wall-only
    variants they mark where the door would be. Used to build the per-pixel TL
    map so closed doors (25 dB) differ from plain walls (47 dB).
    """
    return spec.get("door_pixels", [])


def scenarios() -> dict[str, dict]:
    door_h = 20  # 2 m door height in px
    mid = ROOM // 2
    s = {}

    door_mid = [(y, ROOM) for y in range(mid - door_h // 2, mid + door_h // 2)]
    s["two_rooms_door_open_middle"] = {
        "grid": _room_pair((mid - door_h // 2, mid + door_h // 2)),
        "desc": "10x10m rooms, 2m door centered on the shared wall, OPEN",
        "start": (0.5, 5.0),
        "target": (19.5, 5.0),  # meters, straight through the door
        "expect": "line_of_sight",
        "door_pixels": door_mid,
    }
    s["two_rooms_door_closed_middle"] = {
        "grid": _room_pair(None),
        "desc": "10x10m rooms, door CLOSED (25 dB door TL, lighter than the 47 dB wall)",
        "start": (0.5, 5.0),
        "target": (19.5, 5.0),
        "expect": "blocked",
        "door_pixels": door_mid,
    }
    door_side = [(y, ROOM) for y in range(10, 10 + door_h)]
    s["two_rooms_door_open_side"] = {
        "grid": _room_pair((10, 10 + door_h)),
        "desc": "10x10m rooms, 2m door at the BOTTOM side of the shared wall, OPEN",
        "start": (0.5, 5.0),
        "target": (19.5, 5.0),
        "expect": "detour",
        "door_pixels": door_side,
    }
    s["two_rooms_door_closed_side"] = {
        "grid": _room_pair(None),
        "desc": "10x10m rooms, side door CLOSED (25 dB door TL)",
        "start": (0.5, 5.0),
        "target": (19.5, 5.0),
        "expect": "blocked",
        "door_pixels": door_side,
    }
    s["two_rooms_wall_only"] = {
        "grid": _room_pair(None),
        "desc": "10x10m rooms, solid wall, NO door at all",
        "start": (0.5, 5.0),
        "target": (19.5, 5.0),
        "expect": "blocked",
    }
    s["long_corridor"] = {
        "grid": _corridor(30, 300),  # 3 m wide, 30 m long
        "desc": "30m straight corridor, 3m wide",
        "start": (1.0, 1.5),
        "target": (29.0, 1.5),
        "expect": "line_of_sight",
    }
    s["corridor_narrowing"] = {
        "grid": _corridor(30, 300),
        "desc": "corridor narrowing from 3m to 1m mid-way",
        "start": (1.0, 1.5),
        "target": (29.0, 1.5),
        "expect": "narrowed",
    }
    s["corridor_side_door"] = {
        "grid": _corridor(30, 300),
        "desc": "corridor split by a closed door across its width",
        "start": (1.0, 1.5),
        "target": (29.0, 1.5),
        "expect": "blocked",
    }
    return s


def build(spec: dict) -> np.ndarray:
    """Materialize the scenario grid, applying the variant-specific edits."""
    name = spec["_name"] if "_name" in spec else ""
    grid = spec["grid"].copy()
    if name == "corridor_narrowing":
        # pinch the corridor to 1m for 10px in the middle
        mid = grid.shape[1] // 2
        h = grid.shape[0]
        pinch = h - 10
        grid[10:pinch, mid - 5 : mid + 5] = WALL
    elif name == "corridor_side_door":
        # full-width wall with a closed 2m door gap (blocked = keep wall)
        mid = grid.shape[1] // 2
        grid[:, mid] = WALL
    return grid


def export_pngs(out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Render each scenario as a PNG for visual inspection.

    Walls are white; DOOR pixels are drawn in gray (128) so doors stay visually
    distinct from plain walls even when closed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in scenarios().items():
        grid = build({**spec, "_name": name})
        img = (grid * 255).astype(np.uint8)
        for y, x in door_pixels(spec):
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                img[y, x] = 128  # door: gray
        p = out_dir / f"{name}.png"
        Image.fromarray(img).save(p)
        paths[name] = p
    return paths


if __name__ == "__main__":
    import sys

    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("test_images")
    paths = export_pngs(out)
    for name, p in paths.items():
        print(f"  {name:32s} {p}")
