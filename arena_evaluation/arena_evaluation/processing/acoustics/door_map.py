#!/usr/bin/env python3
"""Door geometry extraction: world.yaml door definitions -> pixel segments.

The acoustic solver works on the occupancy grid derived from the map PNG; doors
are not distinguishable there. This module overlays the *declarative* door
geometry (world.yaml, the same source the sim uses) onto the grid so each door
owns a pixel segment that the calculator can carve (open) or give a distinct
transmission loss (closed).

Door pixels are represented as a boolean mask per door, aligned with the
occupancy grid (row 0 = bottom of the map, matching MapRegistry output).
"""

from __future__ import annotations

import functools
import logging
import pathlib

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default transmission loss (dB) for a CLOSED door. Doors are lighter than the
# 47 dB brick wall: 25 dB is a typical sealed interior door.
DEFAULT_DOOR_TL_DB = 25.0


@functools.lru_cache(maxsize=32)
def _load_world_yaml(world_path: str) -> dict:
    with open(world_path) as f:
        return yaml.safe_load(f)


def _find_world_yaml(map_name: str, run_dir: pathlib.Path | str | None = None) -> pathlib.Path | None:
    """Locate the world.yaml for a map: run_dir/worlds/<map>... or via the
    arena_simulation_setup package worlds/ directory."""
    import os

    candidates: list[pathlib.Path] = []
    if run_dir is not None:
        candidates.append(pathlib.Path(run_dir) / "worlds" / map_name / "0" / "world.yaml")
        candidates.append(pathlib.Path(run_dir) / "worlds" / map_name / "world.yaml")

    # package worlds (same layout the sim uses)
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory("arena_simulation_setup"))
        candidates.append(share / "worlds" / map_name / "0" / "world.yaml")
        candidates.append(share / "worlds" / map_name / "world.yaml")
    except Exception:
        pass

    ws_roots = []
    if arena_dir := os.environ.get("ARENA_DIR"):
        ws_roots.append(pathlib.Path(arena_dir) / "arena_simulation_setup" / "worlds")
    for root in ws_roots:
        candidates.append(root / map_name / "0" / "world.yaml")
        candidates.append(root / map_name / "world.yaml")

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Pixel line from (x0,y0) to (x1,y1) (Bresenham, 8-connected)."""
    pts: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return pts


def door_segments(
    map_name: str,
    grid: np.ndarray,
    resolution: float,
    origin: tuple[float, float, float],
    run_dir: pathlib.Path | str | None = None,
) -> dict[str, tuple[np.ndarray, float]]:
    """Build per-door pixel masks + TL for a map's occupancy grid.

    Returns {door_name: (mask, closed_tl_db)} where mask is a bool array the
    same shape as *grid* (True on the door's pixels). Empty dict when no
    world.yaml is found or it declares no doors.
    """
    world_path = _find_world_yaml(map_name, run_dir=run_dir)
    if world_path is None:
        logger.debug("No world.yaml found for map %r, no door geometry", map_name)
        return {}

    try:
        world = _load_world_yaml(str(world_path))
    except Exception as e:
        logger.warning("Failed to load world.yaml %s: %s", world_path, e)
        return {}

    ox, oy, _ = origin
    h, w = grid.shape
    result: dict[str, tuple[np.ndarray, float]] = {}

    def collect_doors(level: dict, level_name: str) -> None:
        for zone in level.get("zones", []) or []:
            for door in zone.get("doors", []) or []:
                name = door.get("name") or f"door_{len(result)}"

                # door coords may be {x, y} dicts or [x, y, ...] lists
                def _pt(entry: dict | list | tuple) -> tuple[float, float]:
                    if isinstance(entry, dict):
                        return float(entry.get("x", 0.0)), float(entry.get("y", 0.0))
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        return float(entry[0]), float(entry[1])
                    return 0.0, 0.0

                sx, sy = _pt(door.get("start") or {})
                ex, ey = _pt(door.get("end") or {})
                width_m = float(door.get("width", 1.0))
                kind = str(door.get("kind", "sliding"))
                tl_db = float(door.get("tl_db", DEFAULT_DOOR_TL_DB))

                # door axis: line from start to end; thickness = width perpendicular
                x0, y0 = int(round((sx - ox) / resolution)), int(round((sy - oy) / resolution))
                x1, y1 = int(round((ex - ox) / resolution)), int(round((ey - oy) / resolution))

                line = _bresenham_line(x0, y0, x1, y1)
                if not line:
                    continue

                # thickness: offset the line perpendicular by +/- width/2
                dx, dy = x1 - x0, y1 - y0
                length = max(np.hypot(dx, dy), 1e-9)
                ux, uy = dx / length, dy / length
                half = max(1, int(round((width_m / 2.0) / resolution)))

                mask = np.zeros((h, w), dtype=bool)
                for px, py in line:
                    for off in range(-half, half + 1):
                        oxx = px - uy * off
                        oyy = py + ux * off
                        ix, iy = int(round(oxx)), int(round(oyy))
                        if 0 <= ix < w and 0 <= iy < h:
                            mask[iy, ix] = True

                n_on_wall = int(np.sum(mask & (grid == 1)))
                if n_on_wall == 0:
                    logger.warning(
                        "Door %r (%s) lands on no wall pixels - skipping (map/world mismatch?)",
                        name,
                        world_path,
                    )
                    continue

                result[f"{level_name}/{name}"] = (mask, tl_db)

    for fid, level in (world.get("levels") or {}).items():
        collect_doors(level, str(fid))
    if not result:
        # flat single-level layout
        collect_doors(world, "world")

    if result:
        logger.info("Door geometry: %d doors from %s", len(result), world_path)
    return result


def _entity_matches_door(door_key: str, entity: str) -> bool:
    """Check whether a semantic entity name refers to a world.yaml door.

    Semantic entities arrive with an ``env_N/`` prefix and a trailing
    ``/N`` Gazebo model-instance suffix (e.g. ``env_0/main_hallway/0``)
    while world.yaml doors are keyed as ``{level}/{name}`` (e.g.
    ``world/main_hallway``).  This helper strips both decorations so the
    door-state timeline can be wired to the geometry.
    """
    import re

    # Strip env_N/ prefix and trailing /N (Gazebo model instance id)
    normalized = re.sub(r"^env_\d+/", "", entity)
    normalized = re.sub(r"/\d+$", "", normalized)
    if normalized == door_key:
        return True
    # Fallback: match the leaf name (last meaningful path segment)
    door_leaf = door_key.rsplit("/", 1)[-1]
    entity_leaf = normalized.rsplit("/", 1)[-1]
    return door_leaf == entity_leaf


def build_pixel_tl(
    grid: np.ndarray,
    doors: dict[str, tuple[np.ndarray, float]],
    open_doors: set[str] | None = None,
    wall_tl_db: float = 47.0,
) -> np.ndarray:
    """Assemble the per-pixel TL map for the solver.

    - free space: 0 dB
    - walls: wall_tl_db
    - closed doors: their door TL (25 dB default)
    - open doors (in *open_doors*): 0 dB (carved, sound passes)

    Open doors take precedence: a door pixel that is open behaves as free.
    """
    tl = np.where(grid == 1, wall_tl_db, 0.0).astype(np.float32)
    open_doors = open_doors or set()
    for name, (mask, door_tl) in doors.items():
        if any(_entity_matches_door(name, e) for e in open_doors):
            tl[mask] = 0.0
        else:
            tl[mask] = float(door_tl)
    return np.ascontiguousarray(tl)
