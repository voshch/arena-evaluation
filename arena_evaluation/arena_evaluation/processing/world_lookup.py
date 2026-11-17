"""The copy of each world that the installed arena_simulation_setup package ships."""

from __future__ import annotations

import pathlib


def installed_world_dir(map_name: str) -> pathlib.Path:
    """The installed copy of a world; raises when the arena_simulation_setup package cannot be found."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = pathlib.Path(get_package_share_directory("arena_simulation_setup"))
    except (ImportError, OSError, KeyError) as error:
        raise FileNotFoundError(
            f"arena_simulation_setup is not installed, so the world {map_name!r} cannot be located"
        ) from error
    return share / "worlds" / map_name
