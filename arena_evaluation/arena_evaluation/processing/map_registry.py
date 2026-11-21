"""Resolve a world into the raster map the metrics consume, cached beside the benchmark's data."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import pathlib
import subprocess
from collections.abc import Iterator
from typing import TYPE_CHECKING

import yaml
from PIL import Image

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World import Level

_log = logging.getLogger(__name__)

_MAP_YAML_CANDIDATES = ("map/map.yaml", "map.yaml")
_DEFAULT_RESOLUTION = 0.05
_ARENA_WORLDS_ROOT = "/opt/arena_ws/src/Arena/arena_simulation_setup/worlds"


class MapRegistry:
    @staticmethod
    def compacted_world(map_name: str, run_dir: pathlib.Path | None = None) -> tuple[Level | None, dict[str, tuple[float, float]] | None]:
        """The world's levels compacted onto one grid the way the runtime renders them, with its level origins. A world copied into the run's worlds folder wins over the workspace's. Raises when the world does not resolve."""
        with MapRegistry.compacted_world_scope(map_name, run_dir=run_dir) as compacted_and_origins:
            return compacted_and_origins

    @staticmethod
    @contextlib.contextmanager
    def compacted_world_scope(map_name: str, run_dir: pathlib.Path | None = None) -> Iterator[tuple[Level | None, dict[str, tuple[float, float]] | None]]:
        """Like compacted_world, with the world's assets resolvable inside the block. A run's copy of a world carries its layout but not its assets, so those come from the workspace world of the same name."""
        from arena_simulation_setup.tree.World import (
            WorldIdentifier,
            WorldView,
            world_assets,
        )

        bare, level_filter = WorldIdentifier.split_filter(map_name)
        run_world = pathlib.Path(run_dir) / "worlds" / bare if run_dir is not None else None
        view = WorldView(run_world) if run_world is not None and run_world.is_dir() else WorldIdentifier(bare).resolve_sync()
        if (view.path / "assets").is_dir():
            asset_view = view
        else:
            try:
                asset_view = WorldIdentifier(bare).resolve_sync()
            except (FileNotFoundError, ValueError, LookupError):
                asset_view = view
        with world_assets(asset_view):
            loaded = view.load(level_filter=level_filter)
            origins = view.level_origins()
            compacted = loaded.compact_world(origins=origins if origins is not None else {fid: (0.0, 0.0) for fid in loaded.level_ids})
            yield compacted, origins

    @staticmethod
    def _render_world_map(map_name: str, cache_dir: pathlib.Path) -> dict | None:
        """Resolve the world the way the runtime does and rasterize its compacted levels into the cache."""
        try:
            compacted, origins = MapRegistry.compacted_world(map_name)
        except Exception as e:
            _log.warning(f"world {map_name!r} did not resolve: {e!r}")
            return None
        if compacted is None:
            return None
        png_bytes, map_yaml_text = compacted.render_map_files(level_origins=origins)
        meta = yaml.safe_load(map_yaml_text)
        with Image.open(io.BytesIO(png_bytes)) as image:
            return _write_cache(cache_dir, map_name, image.convert("RGBA"), meta)

    @staticmethod
    def _world_preview(world_dir: pathlib.Path) -> pathlib.Path | None:
        """The preview map yaml of a single-level world directory, read through the world loader."""
        from arena_simulation_setup.tree.World import WorldView

        try:
            world = WorldView(world_dir).load(validate=False)
        except (FileNotFoundError, ValueError):
            return None
        if len(world.levels) != 1:
            return None
        path = WorldView(world_dir).map_yaml(next(iter(world.levels)))
        return path if path.exists() else None

    @staticmethod
    def _find_ros_map_dir(map_name: str, run_dir: pathlib.Path | None = None) -> pathlib.Path | None:
        """The first run-world, workspace-world or ROS-share world directory the name resolves to."""
        roots = [pathlib.Path(_ARENA_WORLDS_ROOT)]
        arena_dir = os.environ.get("ARENA_DIR")
        if arena_dir:
            roots.append(pathlib.Path(arena_dir) / "arena_simulation_setup/worlds")
        try:
            from ament_index_python.packages import get_package_share_directory

            roots.append(pathlib.Path(get_package_share_directory("arena_simulation_setup")) / "worlds")
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["rospack", "find", "arena_simulation_setup"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                roots.append(pathlib.Path(result.stdout.strip()) / "worlds")
        except Exception:
            pass

        candidates = [map_name]
        candidates += [map_name.split(sep)[0] for sep in ("_stage", "-", "_") if sep in map_name]
        for candidate in dict.fromkeys(candidates):
            if run_dir is not None:
                for sub in ("worlds", "maps"):
                    path = pathlib.Path(run_dir) / sub / candidate
                    if path.exists():
                        return path
            for root in roots:
                path = root / candidate
                if path.exists():
                    return path
        return None

    @staticmethod
    def get_map(
        map_name: str,
        cache_dir: pathlib.Path | None = None,
        run_dir: pathlib.Path | None = None,
    ) -> dict | None:
        """The cached raster metadata of a world, rendering or converting it on the first request."""
        if not map_name or map_name == "unknown":
            return None
        cache_dir = cache_dir or _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        png_path = cache_dir / f"{map_name}.png"
        meta_path = cache_dir / f"{map_name}.yaml"
        if png_path.exists() and meta_path.exists():
            return yaml.safe_load(meta_path.read_text())

        ros_map_dir = MapRegistry._find_ros_map_dir(map_name, run_dir=run_dir)
        if ros_map_dir is None:
            return MapRegistry._render_world_map(map_name, cache_dir)
        yaml_path = next(
            (ros_map_dir / candidate for candidate in _MAP_YAML_CANDIDATES if (ros_map_dir / candidate).exists()),
            None,
        ) or MapRegistry._world_preview(ros_map_dir)
        if yaml_path is None:
            return MapRegistry._render_world_map(map_name, cache_dir)

        map_meta = yaml.safe_load(yaml_path.read_text())
        image_path = yaml_path.parent / map_meta.get("image", "map.pgm")
        if not image_path.exists():
            return None
        try:
            with Image.open(image_path) as image:
                return _write_cache(cache_dir, map_name, image.convert("RGBA"), map_meta)
        except Exception as e:
            print(f"Failed to cache map {map_name}: {e}")
            return None


def _default_cache_dir() -> pathlib.Path:
    data_root = os.environ.get("ARENA_DATA_DIR") or os.path.join(os.environ.get("ARENA_WS_DIR", "/opt/arena_ws"), "data")
    return pathlib.Path(data_root) / "maps_cache"


def _write_cache(cache_dir: pathlib.Path, map_name: str, image: Image.Image, meta: dict) -> dict:
    """Write the PNG and its metadata yaml into the cache, returning the metadata."""
    png_path = cache_dir / f"{map_name}.png"
    image.save(png_path)
    cache_meta = {
        "png_path": str(png_path),
        "resolution": float(meta.get("resolution", _DEFAULT_RESOLUTION)),
        "origin": [float(x) for x in meta.get("origin", (0.0, 0.0, 0.0))],
        "width": image.width,
        "height": image.height,
    }
    (cache_dir / f"{map_name}.yaml").write_text(yaml.safe_dump(cache_meta))
    return cache_meta
