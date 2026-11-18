#!/usr/bin/env python3
"""Acoustic barrier grid and door transmission loss for a world.

The solver works on a barrier grid derived from the world's walls and doors by
``barrier_map.derive_barrier_map``. The grid is not read from the rendered map PNG, so walls and doors sit where the
world puts them. This module loads a world, derives its grid, and gives each door its own cells with a closed
transmission loss (TL) for the solver.
"""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import typing

import numpy as np

from arena_evaluation.processing.acoustics.barrier_map import (
    BarrierMap,
    DoorOpening,
    WallSegment,
    derive_barrier_map,
)
from arena_evaluation.processing.world_lookup import installed_world_dir
from arena_evaluation.storage.exceptions import MetricInputError

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared import Level

logger = logging.getLogger(__name__)

# Default transmission loss (dB) for a CLOSED door. Doors are lighter than the
# 47 dB brick wall: 25 dB is a typical sealed interior door.
DEFAULT_DOOR_TL_DB = 25.0
DEFAULT_RESOLUTION_M = 0.05


@functools.lru_cache(maxsize=32)
def _load_level(world_dir: str) -> Level:
    """The world's levels flattened into one map frame, which is the frame agent positions are recorded in."""
    from arena_simulation_setup.tree.World import WorldView

    world = WorldView(pathlib.Path(world_dir)).load(validate=False)
    origins = (
        world.level_origins()
        if len(world.levels) > 1
        else dict.fromkeys(world.level_ids, (0.0, 0.0))
    )
    return world.compact_world(origins)


def _world_candidates(
    map_name: str, run_dir: pathlib.Path | str | None = None
) -> list[tuple[str, pathlib.Path | None]]:
    """Where a map's world may live, in lookup order, labelled for error messages.

    A path of None marks a location that cannot be used; its label says why.
    """
    candidates: list[tuple[str, pathlib.Path | None]] = []
    if run_dir is not None:
        candidates.append(
            ("run directory", pathlib.Path(run_dir) / "worlds" / map_name)
        )

    try:
        installed = installed_world_dir(map_name)
        candidates.append(("installed package", installed))
    except FileNotFoundError:
        candidates.append(
            (
                "installed package unavailable (AMENT_PREFIX_PATH unset or package missing)",
                None,
            )
        )

    if arena_dir := os.environ.get("ARENA_DIR"):
        candidates.append(
            (
                "ARENA_DIR source tree",
                pathlib.Path(arena_dir)
                / "arena_simulation_setup"
                / "worlds"
                / map_name,
            )
        )
    return candidates


def _world_dirs(
    map_name: str, run_dir: pathlib.Path | str | None = None
) -> list[pathlib.Path]:
    """The candidate locations that exist as directories, in lookup order."""
    return [
        path
        for _label, path in _world_candidates(map_name, run_dir=run_dir)
        if path is not None and path.is_dir()
    ]


def _find_world_level(
    map_name: str, run_dir: pathlib.Path | str | None = None
) -> tuple[pathlib.Path, Level] | None:
    """The first candidate directory holding a world, and that world's level."""
    for cand in _world_dirs(map_name, run_dir=run_dir):
        try:
            return cand, _load_level(str(cand))
        except FileNotFoundError:
            continue
    return None


@functools.lru_cache(maxsize=8)
def _barrier_map_for_world(world_dir: str, resolution: float) -> BarrierMap:
    level = _load_level(world_dir)
    walls = [
        WallSegment(start=(wall.start.x, wall.start.y), end=(wall.end.x, wall.end.y))
        for wall in level.all_walls
    ]
    doors = [
        DoorOpening(
            name=door.name,
            start=(door.start.x, door.start.y),
            end=(door.end.x, door.end.y),
            width=door.width,
        )
        for door in level.doors
    ]
    # Zones give the walkable extent, so that agents standing in open space are on the grid. They are not barriers.
    extent = [(point.x, point.y) for zone in level.zones for point in zone.polygon]
    return derive_barrier_map(
        walls=walls, doors=doors, extent=extent, resolution=resolution
    )


def load_barrier_map(
    map_name: str,
    run_dir: pathlib.Path | str | None = None,
    resolution: float = DEFAULT_RESOLUTION_M,
) -> BarrierMap:
    """The barrier grid of a world, derived from its zones, walls and doors."""
    found = _find_world_level(map_name, run_dir=run_dir)
    if found is None:
        searched = "; ".join(
            label if path is None else f"{label} at {path}"
            for label, path in _world_candidates(map_name, run_dir=run_dir)
        )
        raise MetricInputError(
            f"AcousticExposureCalculator: missing world map {map_name!r}; searched {searched}"
        )
    world_dir, _level = found
    return _barrier_map_for_world(str(world_dir), float(resolution))


def barrier_door_tl(barrier: BarrierMap) -> dict[str, tuple[np.ndarray, float]]:
    """Door masks keyed as the door-state timeline names them (``world/<name>``), with the closed-door TL per cell.

    The solver charges every barrier cell a path enters, so a door's closed TL is shared by the cells it crosses: each
    cell holds ``1 / door_cells_across`` of the door's TL, and a path through the whole door pays the full TL once.
    """
    return {
        f"world/{name}": (mask, DEFAULT_DOOR_TL_DB / barrier.door_cells_across[name])
        for name, mask in barrier.door_masks.items()
    }


def _entity_matches_door(door_key: str, entity: str) -> bool:
    """Check whether a semantic entity name refers to a world door.

    Semantic entities arrive with an ``env_N/`` prefix and a trailing
    ``/N`` Gazebo model-instance suffix (e.g. ``env_0/main_hallway/0``)
    while world doors are keyed as ``world/{name}`` (e.g.
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


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Pixel line from (x0, y0) to (x1, y1) (Bresenham, 8-connected)."""
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
    """Build per-door pixel masks + TL for a map's occupancy grid (kept for PR-A/plot compatibility)."""
    found = _find_world_level(map_name, run_dir=run_dir)
    if found is None:
        return {}
    world_dir, level = found

    ox, oy, _ = origin
    h, w = grid.shape
    result: dict[str, tuple[np.ndarray, float]] = {}

    for door in level.doors:
        x0, y0 = (
            int(round((door.start.x - ox) / resolution)),
            int(round((door.start.y - oy) / resolution)),
        )
        x1, y1 = (
            int(round((door.end.x - ox) / resolution)),
            int(round((door.end.y - oy) / resolution)),
        )
        line = _bresenham_line(x0, y0, x1, y1)
        if not line:
            continue
        dx, dy = x1 - x0, y1 - y0
        length = max(np.hypot(dx, dy), 1e-9)
        ux, uy = dx / length, dy / length
        half = max(1, int(round((door.width / 2.0) / resolution)))

        mask = np.zeros((h, w), dtype=bool)
        for px, py in line:
            for off in range(-half, half + 1):
                ix = int(round(px - uy * off))
                iy = int(round(py + ux * off))
                if 0 <= ix < w and 0 <= iy < h:
                    mask[iy, ix] = True

        near = mask.copy()
        near[1:, :] |= mask[:-1, :]
        near[:-1, :] |= mask[1:, :]
        near[:, 1:] |= mask[:, :-1]
        near[:, :-1] |= mask[:, 1:]
        if not (near & (grid == 1)).any():
            logger.warning(
                "Door %r (%s) lands on no wall pixels - skipping",
                door.name,
                world_dir,
            )
            continue

        result[f"world/{door.name}"] = (mask, DEFAULT_DOOR_TL_DB)

    return result
