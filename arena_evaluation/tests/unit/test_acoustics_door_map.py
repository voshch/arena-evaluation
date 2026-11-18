"""Unit tests for the world-derived barrier map and door TL (acoustics/door_map.py).

Covers world discovery (run_dir / ament / ws-root), loading the barrier grid derived from a world, the door TL
mapping, entity-name matching, and the per-pixel TL assembly (walls 47 dB / doors 25 dB / open 0 dB).

All worlds are synthesized in tmp_path; ARENA_DIR is pointed at tmp_path when
the ws_root candidate must be exercised.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from collections.abc import Iterator

import numpy as np
import pytest
import yaml
from hypothesis import given, settings, strategies as st

from arena_evaluation.processing.acoustics.door_map import (
    DEFAULT_DOOR_TL_DB,
    _entity_matches_door,
    _find_world_level,
    _load_level,
    barrier_door_tl,
    build_pixel_tl,
    load_barrier_map,
)
from arena_evaluation.processing.world_lookup import installed_world_dir
from arena_evaluation.storage.exceptions import MetricInputError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _door(
    name="door", start=(0.0, 0.0), end=(4.0, 0.0), width=1.0, kind="sliding"
) -> dict:
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


# ---------------------------------------------------------------------------
# _find_world_level
# ---------------------------------------------------------------------------


def test_find_world_level_run_dir_legacy_layout(tmp_path):
    run = tmp_path / "run"
    world = _write_legacy_world(run / "worlds" / "m1", [_door("d1")])
    found = _find_world_level("m1", run_dir=run)
    assert found is not None
    assert found[0] == world
    assert [door.name for door in found[1].doors] == ["d1"]


def test_find_world_level_run_dir_format2(tmp_path):
    run = tmp_path / "run"
    world = _write_world(run / "worlds" / "m2", doors=[_door("d2")])
    found = _find_world_level("m2", run_dir=run)
    assert found is not None
    assert found[0] == world


def test_find_world_level_run_dir_without_world_skipped(tmp_path):
    run = tmp_path / "run"
    (run / "worlds" / "ae_nonesuch_map" / "scenarios").mkdir(parents=True)
    assert _find_world_level("ae_nonesuch_map", run_dir=run) is None


def test_installed_world_dir_raises_file_not_found_without_ament():
    with _env_vars(AMENT_PREFIX_PATH=None):
        with pytest.raises(FileNotFoundError, match="arena_simulation_setup"):
            installed_world_dir("ae_nonesuch_map")


def test_find_world_level_ws_root_fallback(tmp_path):
    with _env_vars(ARENA_DIR=str(tmp_path / "src" / "Arena")):
        ws_world = _write_world(
            tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "m4"
        )
        found = _find_world_level("m4", run_dir=None)
    assert found is not None
    assert found[0] == ws_world


def test_find_world_level_ws_root_legacy_layout(tmp_path):
    with _env_vars(ARENA_DIR=str(tmp_path / "src" / "Arena")):
        ws_world = _write_legacy_world(
            tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "m5",
            [_door("d5")],
        )
        found = _find_world_level("m5", run_dir=None)
    assert found is not None
    assert found[0] == ws_world


def test_find_world_level_not_found(tmp_path):
    # fake name cannot exist in run_dir, ament share, or the real ws roots
    assert _find_world_level("ae_nonesuch_map", run_dir=tmp_path / "run") is None


def test_load_level_cached(tmp_path):
    world = _write_world(tmp_path / "w", doors=[_door("first")])
    assert [door.name for door in _load_level(str(world)).doors] == ["first"]
    # lru_cache: mutation on disk is not picked up
    _write_world(tmp_path / "w", doors=[_door("second")])
    assert [door.name for door in _load_level(str(world)).doors] == ["first"]


# ---------------------------------------------------------------------------
# load_barrier_map / barrier_door_tl
# ---------------------------------------------------------------------------


def test_load_barrier_map_no_world_raises_and_names_the_search(tmp_path):
    with pytest.raises(MetricInputError, match="ae_nonesuch_map") as error:
        load_barrier_map("ae_nonesuch_map", run_dir=tmp_path / "run")
    assert "run directory" in str(error.value)


def test_load_barrier_map_unusable_ament_is_named_in_the_error(tmp_path):
    with _env_vars(AMENT_PREFIX_PATH=None, ARENA_DIR=None):
        with pytest.raises(MetricInputError, match="installed package unavailable"):
            load_barrier_map("ae_nonesuch_map", run_dir=tmp_path / "run")


def test_load_barrier_map_corrupt_world_yaml_raises(tmp_path):
    _write_world(tmp_path / "run" / "worlds" / "m1", corrupt=True)
    with pytest.raises(yaml.YAMLError):
        load_barrier_map("m1", run_dir=tmp_path / "run")


def test_load_barrier_map_world_without_walls_or_doors_raises(tmp_path):
    _write_world(
        tmp_path / "run" / "worlds" / "m1", levels=[_level("0", [], zones=False)]
    )
    with pytest.raises(MetricInputError, match="no walls, doors or extent"):
        load_barrier_map("m1", run_dir=tmp_path / "run")


def test_load_barrier_map_needs_walls_and_doors_not_zones(tmp_path):
    world_dir = tmp_path / "run" / "worlds" / "m1"
    world_dir.mkdir(parents=True)
    level = {
        "id": "0",
        "walls": [{"path": [[0.0, 0.0], [10.0, 0.0]]}],
        "doors": [_door("d", start=(4.0, 0.0), end=(6.0, 0.0), width=1.0)],
    }
    (world_dir / "world.yaml").write_text(
        yaml.safe_dump({"format": "arena-world/2", "levels": [level]})
    )

    barrier = load_barrier_map("m1", run_dir=tmp_path / "run", resolution=1.0)

    assert barrier is not None
    assert barrier.occupancy.sum() > 0
    assert not barrier.occupancy[barrier.door_masks["d"]].any()


def test_load_barrier_map_room_and_door_on_the_grid(tmp_path):
    _write_world(
        tmp_path / "run" / "worlds" / "m1",
        doors=[_door("main_door", start=(10.0, 5.0), end=(16.0, 5.0), width=1.0)],
    )

    barrier = load_barrier_map("m1", run_dir=tmp_path / "run", resolution=1.0)

    assert barrier is not None
    assert barrier.resolution == 1.0
    assert barrier.origin == (-4.5, -4.5)
    assert set(barrier.door_masks) == {"main_door"}
    door_mask = barrier.door_masks["main_door"]
    assert door_mask.sum() == 12
    assert not barrier.occupancy[door_mask].any()


def test_barrier_door_tl_keys_and_closed_tl(tmp_path):
    _write_world(
        tmp_path / "run" / "worlds" / "m1",
        doors=[_door("s", start=(10.0, 5.0), end=(16.0, 5.0))],
    )
    barrier = load_barrier_map("m1", run_dir=tmp_path / "run")

    tl_map = barrier_door_tl(barrier)

    assert set(tl_map) == {"world/s"}
    mask, tl = tl_map["world/s"]
    assert mask is barrier.door_masks["s"]
    assert tl == pytest.approx(DEFAULT_DOOR_TL_DB * barrier.resolution / 1.0)


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

    opened = build_pixel_tl(
        barrier.occupancy, doors, open_doors={"env_0/world/main_door/2"}
    )
    assert np.all(opened[mask] == 0.0)


# ---------------------------------------------------------------------------
# _entity_matches_door
# ---------------------------------------------------------------------------


def test_entity_matches_door_exact():
    assert _entity_matches_door("world/d1", "world/d1")


def test_entity_matches_door_strips_env_prefix():
    assert _entity_matches_door("world/d1", "env_0/world/d1")
    assert _entity_matches_door("world/d1", "env_42/world/d1/7")


def test_entity_matches_door_strips_model_suffix():
    assert _entity_matches_door("world/d1", "world/d1/3")


def test_entity_matches_door_leaf_fallback():
    assert _entity_matches_door(
        "level_a/main_hallway", "env_0/some_zone/main_hallway/0"
    )


def test_entity_matches_door_no_match():
    assert not _entity_matches_door("world/d1", "world/d2")
    assert not _entity_matches_door("world/d1", "env_0/world/d2/1")
    assert not _entity_matches_door("world/d1", "env_0/other")


# ---------------------------------------------------------------------------
# build_pixel_tl
# ---------------------------------------------------------------------------


def test_build_pixel_tl_baseline():
    grid = _wall_grid()
    tl = build_pixel_tl(grid, {})
    assert tl.dtype == np.float32
    assert tl.shape == grid.shape
    assert tl.flags["C_CONTIGUOUS"]
    assert np.all(tl[grid == 1] == 47.0)
    assert np.all(tl[grid == 0] == 0.0)


def test_build_pixel_tl_custom_wall_tl():
    grid = _wall_grid()
    tl = build_pixel_tl(grid, {}, wall_tl_db=40.0)
    assert np.all(tl[grid == 1] == 40.0)


def test_build_pixel_tl_closed_door_overrides_wall():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(grid, {"world/d1": (mask, DEFAULT_DOOR_TL_DB)})
    assert np.all(tl[mask] == DEFAULT_DOOR_TL_DB)


def test_build_pixel_tl_open_door_free():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(
        grid, {"world/d1": (mask, DEFAULT_DOOR_TL_DB)}, open_doors={"world/d1"}
    )
    assert np.all(tl[mask] == 0.0)
    # wall pixels outside the door still blocked
    assert np.all(tl[5, :10] == 47.0)


def test_build_pixel_tl_open_entity_env_prefixed():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(
        grid, {"world/d1": (mask, 25.0)}, open_doors={"env_0/world/d1/7"}
    )
    assert np.all(tl[mask] == 0.0)


def test_build_pixel_tl_open_entity_leaf_match():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(grid, {"level_a/d1": (mask, 25.0)}, open_doors={"env_3/d1/2"})
    assert np.all(tl[mask] == 0.0)


def test_build_pixel_tl_only_matching_door_opened():
    grid = _wall_grid()
    m1 = np.zeros_like(grid, dtype=bool)
    m1[5, 10:16] = True
    m2 = np.zeros_like(grid, dtype=bool)
    m2[12, 10:16] = True
    doors = {"world/d1": (m1, 25.0), "world/d2": (m2, 18.0)}
    tl = build_pixel_tl(grid, doors, open_doors={"world/d1"})
    assert np.all(tl[m1] == 0.0)
    assert np.all(tl[m2] == 18.0)


def test_build_pixel_tl_open_doors_none_means_all_closed():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(grid, {"world/d1": (mask, 25.0)}, open_doors=None)
    assert np.all(tl[mask] == 25.0)


def test_build_pixel_tl_door_in_free_space():
    # a door mask over free space still gets its TL (geometry wins)
    grid = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[3, 3:7] = True
    tl = build_pixel_tl(grid, {"world/d1": (mask, 25.0)})
    assert np.all(tl[mask] == 25.0)
    assert np.all(tl[~mask] == 0.0)


@given(
    width=st.integers(min_value=4, max_value=20),
    height=st.integers(min_value=4, max_value=20),
    wall_rows=st.lists(st.integers(min_value=0, max_value=19), max_size=4, unique=True),
    open_flags=st.lists(st.booleans(), max_size=3),
    tl_db=st.floats(min_value=1.0, max_value=60.0),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_build_pixel_tl_property(width, height, wall_rows, open_flags, tl_db):
    wall_rows = [r for r in wall_rows if r < height] or [height // 2]
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[wall_rows, :] = 1

    doors: dict[str, tuple[np.ndarray, float]] = {}
    door_cols = [1, 5, 9]
    for i, (x, _) in enumerate(zip(door_cols, open_flags)):
        if x >= width:
            break
        mask = np.zeros((height, width), dtype=bool)
        mask[wall_rows, x] = True
        mask[wall_rows, min(x + 1, width - 1)] = True
        doors[f"world/d{i}"] = (mask, tl_db)

    open_set = {
        f"world/d{i}" for i, fl in enumerate(open_flags) if fl and i * 4 + 1 < width
    }
    tl = build_pixel_tl(grid, doors, open_doors=open_set, wall_tl_db=47.0)

    assert tl.dtype == np.float32
    assert tl.shape == grid.shape
    # door pixels: 0 when open, door TL when closed
    for name, (mask, door_tl) in doors.items():
        expected = 0.0 if name in open_set else float(tl_db)
        assert np.all(tl[mask] == expected)
    # pixels outside any door: walls stay wall TL, free stays 0
    outside = np.ones((height, width), dtype=bool)
    for _, (mask, _) in doors.items():
        outside &= ~mask
    assert np.all(tl[outside & (grid == 1)] == 47.0)
    assert np.all(tl[outside & (grid == 0)] == 0.0)
