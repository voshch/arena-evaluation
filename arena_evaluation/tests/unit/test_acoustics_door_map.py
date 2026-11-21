"""Unit tests for the world-derived barrier map and door TL (acoustics/door_map.py)."""

from __future__ import annotations

import contextlib
import inspect
import os
import pathlib
from collections.abc import Iterator

import numpy as np
import pytest
import yaml

from arena_evaluation.processing.acoustics.door_map import (
    DEFAULT_DOOR_TL_DB,
    _find_world_level,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.storage.exceptions import MetricInputError

# Helpers


@contextlib.contextmanager
def _env_vars(**overrides: str | None) -> Iterator[None]:
    """Temporarily set or unset environment variables for the duration of a test."""
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _door(name="door", start=(0.0, 0.0), end=(4.0, 0.0), width=1.0, kind="sliding") -> dict:
    return {
        "name": name,
        "span": [list(start), list(end)],
        "width": width,
        "kind": kind,
    }


_ROOM = [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]]


def _level(level_id: str, doors: list[dict], zones: bool = True) -> dict:
    level: dict = {"id": level_id, "doors": doors}
    if zones:
        level["zones"] = [{"name": f"room_{level_id}", "polygon": _ROOM}]
    return level


def _write_world(
    base: pathlib.Path,
    *,
    levels: list[dict] | None = None,
    doors: list[dict] | None = None,
    corrupt: bool = False,
) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / "world.yaml"
    if corrupt:
        path.write_text("{{{{ not yaml ]]]")
    else:
        if levels is None:
            levels = [_level("0", doors or [])]
        path.write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}))
    return base


def _write_legacy_world(base: pathlib.Path, doors: list[dict]) -> pathlib.Path:
    level = base / "0"
    level.mkdir(parents=True, exist_ok=True)
    legacy_doors = [
        {
            "name": door["name"],
            "start": door["span"][0],
            "end": door["span"][1],
            "width": door["width"],
        }
        for door in doors
    ]
    zone = {"name": "room", "corners": _ROOM, "doors": legacy_doors}
    (level / "world.yaml").write_text(yaml.safe_dump({"zones": [zone]}))
    return base


def _wall_grid(height: int = 20, width: int = 30, wall_rows=(5, 12)) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.uint8)
    for r in wall_rows:
        grid[r, :] = 1
    return grid


# _find_world_level


def test_find_world_level_run_dir_without_world_skipped(tmp_path):
    run = tmp_path / "run"
    (run / "worlds" / "ae_nonesuch_map" / "scenarios").mkdir(parents=True)
    assert _find_world_level("ae_nonesuch_map", run_dir=run) is None


def test_find_world_level_ws_root_fallback(tmp_path):
    with _env_vars(ARENA_DIR=str(tmp_path / "src" / "Arena")):
        ws_world = _write_world(tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "m4")
        found = _find_world_level("m4", run_dir=None)
    assert found is not None
    assert found[0] == ws_world


# load_barrier_map / barrier_door_tl


def test_load_barrier_map_unusable_ament_is_named_in_the_error(tmp_path):
    with _env_vars(AMENT_PREFIX_PATH=None, ARENA_DIR=None):
        with pytest.raises(MetricInputError, match="installed package unavailable"):
            load_barrier_map("ae_nonesuch_map", run_dir=tmp_path / "run")


def test_load_barrier_map_world_without_walls_or_doors_raises(tmp_path):
    _write_world(tmp_path / "run" / "worlds" / "m1", levels=[_level("0", [], zones=False)])
    with pytest.raises(MetricInputError, match="no walls, doors or extent"):
        load_barrier_map("m1", run_dir=tmp_path / "run")


def test_model_defaults_are_the_shared_constants():
    from arena_evaluation.processing.acoustics import constants

    assert DEFAULT_DOOR_TL_DB == constants.DOOR_TRANSMISSION_LOSS_DB == 25.0
    door_default = inspect.signature(barrier_door_tl).parameters["door_tl_db"].default
    assert door_default == DEFAULT_DOOR_TL_DB
    wall_default = inspect.signature(build_pixel_tl).parameters["wall_tl_db"].default
    assert wall_default == constants.WALL_TRANSMISSION_LOSS_DB == 47.0


def test_barrier_doors_carve_the_tl_map(tmp_path):
    _write_world(
        tmp_path / "run" / "worlds" / "m1",
        doors=[_door("main_door", start=(10.0, 5.0), end=(16.0, 5.0))],
    )
    barrier = load_barrier_map("m1", run_dir=tmp_path / "run")
    doors = barrier_door_tl(barrier)
    mask = doors["world/main_door"][0]

    closed = build_pixel_tl(barrier.occupancy, doors)
    assert np.allclose(closed[mask], DEFAULT_DOOR_TL_DB * barrier.resolution / 1.0)

    opened = build_pixel_tl(barrier.occupancy, doors, open_doors={"env_0/world/main_door/2"})
    assert np.all(opened[mask] == 0.0)


# _entity_matches_door


# build_pixel_tl
