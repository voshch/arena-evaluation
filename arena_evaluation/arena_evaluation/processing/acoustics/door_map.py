#!/usr/bin/env python3
"""Acoustic barrier grid and door transmission loss for a world.

The solver works on a barrier grid derived from the world's walls and doors by
``barrier_map.derive_barrier_map``. The grid is not read from the rendered map PNG, so walls and doors sit where the
world puts them. This module loads a world, derives its grid, and gives each door its own cells with a closed
transmission loss (TL) for the solver.

The closed-door and wall transmission losses are declared model parameters
(``constants.DOOR_TRANSMISSION_LOSS_DB`` and ``constants.WALL_TRANSMISSION_LOSS_DB``); ``barrier_door_tl`` and
``build_pixel_tl`` accept a keyword to vary them for a what-if run without changing code.
"""

from __future__ import annotations

import functools
import logging
import os
import pathlib
import re
import typing

import numpy as np

from arena_evaluation.processing.acoustics.barrier_map import (
    BarrierMap,
    DoorOpening,
    WallSegment,
    derive_barrier_map,
)
from arena_evaluation.processing.acoustics.constants import (
    DOOR_TRANSMISSION_LOSS_DB,
    WALL_TRANSMISSION_LOSS_DB,
)
from arena_evaluation.processing.world_lookup import installed_world_dir
from arena_evaluation.storage.exceptions import MetricInputError

if typing.TYPE_CHECKING:
    from arena_simulation_setup.shared import Level

logger = logging.getLogger(__name__)

# Default transmission loss (dB) of a CLOSED door, re-exported under its historical name.
DEFAULT_DOOR_TL_DB = DOOR_TRANSMISSION_LOSS_DB
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
    """Where a map's world may live, in declared lookup order, labelled for errors and provenance.

    Order: the episode run snapshot (``run_dir/worlds/<map>``), then the installed
    ``arena_simulation_setup`` share, then the ``ARENA_DIR`` source tree. A path of None marks a
    location that cannot be used; its label says why.
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


def resolve_world_dir(
    map_name: str, run_dir: pathlib.Path | str | None = None
) -> tuple[pathlib.Path, str] | None:
    """The world directory used for a map and the label of the copy it came from.

    Lookup order is the declared legacy path: run snapshot, installed package, then ARENA_DIR
    source tree. A directory is used only when its world loads; a format 1 world is accepted as
    migration input (the loader emits a DeprecationWarning pointing at `touch_world --migrate`).
    """
    for label, path in _world_candidates(map_name, run_dir=run_dir):
        if path is None or not path.is_dir():
            continue
        try:
            _load_level(str(path))
        except FileNotFoundError:
            continue
        return path, label
    return None


def _find_world_level(
    map_name: str, run_dir: pathlib.Path | str | None = None
) -> tuple[pathlib.Path, Level] | None:
    """The first candidate directory holding a world, and that world's level."""
    resolved = resolve_world_dir(map_name, run_dir=run_dir)
    if resolved is None:
        return None
    world_dir, _source = resolved
    return world_dir, _load_level(str(world_dir))


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
    """The barrier grid of a world, derived from its zones, walls and doors.

    The copy actually read is logged at INFO; `resolve_world_dir` reports it before loading.
    """
    resolved = resolve_world_dir(map_name, run_dir=run_dir)
    if resolved is None:
        searched = "; ".join(
            label if path is None else f"{label} at {path}"
            for label, path in _world_candidates(map_name, run_dir=run_dir)
        )
        raise MetricInputError(
            f"AcousticExposureCalculator: missing world map {map_name!r}; searched {searched}"
        )
    world_dir, source = resolved
    logger.info(
        "AcousticExposureCalculator: world %r read from %s at %s",
        map_name,
        source,
        world_dir,
    )
    return _barrier_map_for_world(str(world_dir), float(resolution))


def barrier_door_tl(
    barrier: BarrierMap, *, door_tl_db: float = DEFAULT_DOOR_TL_DB
) -> dict[str, tuple[np.ndarray, float]]:
    """Door masks keyed as the door-state timeline names them (``world/<name>``), with the closed-door TL per cell.

    ``door_tl_db`` is a declared model parameter, defaulting to ``DEFAULT_DOOR_TL_DB`` (from
    ``constants.DOOR_TRANSMISSION_LOSS_DB``) so a what-if run can vary it without changing code.
    The solver charges every barrier cell a path enters, so a door's TL is shared by the cells it
    crosses: each cell holds ``door_tl_db / door_cells_across``, a path through the whole door
    pays the full TL once, and a grazing path that clips fewer of those cells pays only the
    corresponding fraction, as an approximation of the closed-door transmission loss.
    """
    return {
        f"world/{name}": (mask, door_tl_db / barrier.door_cells_across[name])
        for name, mask in barrier.door_masks.items()
    }


_ENV_PREFIX = re.compile(r"^env_\d+/")
_TRAILING_INSTANCE = re.compile(r"/\d+$")
_FLATTENED_WORLD_NAMESPACE = "world/"


def _normalise_door_identity(name: str) -> str:
    """Full door-id path of a geometry key or semantic entity, with the encoding decorations removed.

    Accepted encodings, because the scene graph and the barrier map name a door differently:

    - geometry keys: ``world/<door id>`` (the flattened world namespace of `barrier_door_tl`),
      or ``<level>/<door id>``;
    - semantic entities: ``env_<n>/<door id>``, ``env_<n>/<door id>/<level id>`` and
      ``env_<n>/world/<door id>/<level id>``.

    A leading ``env_<n>/`` and a trailing all-digit segment (the level id or a Gazebo model
    instance) are metadata. The literal ``world/`` namespace is dropped so a flattened key
    compares with the door's authored scene-graph name. The full remaining path is compared,
    never the leaf.
    """
    normalized = "/".join(part for part in name.split("/") if part)
    normalized = _ENV_PREFIX.sub("", normalized, count=1)
    normalized = _TRAILING_INSTANCE.sub("", normalized, count=1)
    if normalized.startswith(_FLATTENED_WORLD_NAMESPACE):
        normalized = normalized[len(_FLATTENED_WORLD_NAMESPACE) :]
    return normalized


def _entity_matches_door(door_key: str, entity: str) -> bool:
    """Whether a semantic entity names the same door as a geometry key, by full normalised path."""
    return _normalise_door_identity(door_key) == _normalise_door_identity(entity)


def build_pixel_tl(
    grid: np.ndarray,
    doors: dict[str, tuple[np.ndarray, float]],
    open_doors: set[str] | None = None,
    wall_tl_db: float = WALL_TRANSMISSION_LOSS_DB,
) -> np.ndarray:
    """Assemble the per-pixel TL map for the solver.

    - free space: 0 dB
    - walls: ``wall_tl_db`` (default ``WALL_TRANSMISSION_LOSS_DB``)
    - closed doors: their per-cell door TL, already shared over ``door_cells_across``
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
