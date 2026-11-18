"""Process-wide caches of world assets and their zone geometry, keyed by world name."""

from __future__ import annotations

import dataclasses
import logging
import typing

from arena_evaluation.processing.metrics.common.zones import ZoneGeometry

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World import Level, World

logger = logging.getLogger(__name__)

world_cache: dict[str, list[ZoneGeometry] | None] = {}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LoadedWorld:
    """A world asset with its level origins, and its levels compacted into the flattened map frame."""

    world: World
    origins: dict[str, tuple[float, float]]
    flattened: Level


loaded_worlds: dict[str, LoadedWorld | None] = {}


def load_world(world_name: str) -> LoadedWorld | None:
    """The world asset of this name, cached; None when the asset is not available locally."""
    if world_name in loaded_worlds:
        return loaded_worlds[world_name]

    from arena_simulation_setup.tree.World import WorldIdentifier

    try:
        view = WorldIdentifier(world_name).resolve_sync()
        world = view.load()
    except FileNotFoundError as e:
        logger.warning("world '%s' not available locally: %s", world_name, e)
        loaded_worlds[world_name] = None
        return None

    origins = view.level_origins()
    if origins is None:
        origins = {level_id: (0.0, 0.0) for level_id in world.levels}
    loaded = LoadedWorld(
        world=world, origins=origins, flattened=world.compact_world(origins)
    )
    loaded_worlds[world_name] = loaded
    return loaded
