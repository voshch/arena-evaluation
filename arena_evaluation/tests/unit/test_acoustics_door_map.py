"""Unit tests for door geometry extraction (acoustics/door_map.py).

Covers Bresenham rasterization, world.yaml discovery (run_dir / ament / ws-root),
per-door pixel-mask building with thickness and wall-overlap checks, entity-name
matching, and the per-pixel TL assembly (walls 47 dB / doors 25 dB / open 0 dB).

All world.yaml files are synthesized in tmp_path; the module's `pathlib.Path`
binding is remapped when the absolute ws_root candidates must be exercised.
"""
from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest
import yaml
from hypothesis import given, settings, strategies as st

from arena_evaluation.processing.acoustics import door_map as dm
from arena_evaluation.processing.acoustics.door_map import (
    DEFAULT_DOOR_TL_DB,
    _bresenham_line,
    _entity_matches_door,
    _find_world_yaml,
    _load_world_yaml,
    build_pixel_tl,
    door_segments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remap_pathlib(monkeypatch, tmp_path: pathlib.Path) -> None:
    """Point ARENA_DIR at tmp_path / src / Arena for ws_root discovery."""
    monkeypatch.setenv("ARENA_DIR", str(tmp_path / "src" / "Arena"))


def _install_fake_ament(monkeypatch, tmp_path: pathlib.Path, *, raise_error: bool = False) -> None:
    """Stub ament_index_python.packages.get_package_share_directory."""
    pkg = types.ModuleType("ament_index_python.packages")

    def _gpsd(name):
        if raise_error:
            raise ImportError("ament_index_python unavailable")
        return str(tmp_path / "share")

    pkg.get_package_share_directory = _gpsd
    parent = types.ModuleType("ament_index_python")
    parent.packages = pkg
    monkeypatch.setitem(sys.modules, "ament_index_python", parent)
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", pkg)


def _door(name=None, start=(0.0, 0.0), end=(4.0, 0.0), width=1.0, kind="sliding",
          tl_db=None, coords_as_dict=True) -> dict:
    d: dict = {"width": width, "kind": kind}
    if name is not None:
        d["name"] = name
    if tl_db is not None:
        d["tl_db"] = tl_db
    if coords_as_dict:
        d["start"] = {"x": start[0], "y": start[1]}
        d["end"] = {"x": end[0], "y": end[1]}
    else:
        d["start"] = list(start)
        d["end"] = list(end)
    return d


def _write_world(base: pathlib.Path, *, levels: dict | None = None, flat: dict | None = None,
                 corrupt: bool = False) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / "world.yaml"
    if corrupt:
        path.write_text("{{{{ not yaml ]]]")
    else:
        doc: dict = {}
        if levels is not None:
            doc["levels"] = levels
        if flat is not None:
            doc.update(flat)
        path.write_text(yaml.safe_dump(doc))
    return path


def _wall_grid(height: int = 20, width: int = 30, wall_rows=(5, 12)) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.uint8)
    for r in wall_rows:
        grid[r, :] = 1
    return grid


def _levels_world(door_lists: list[list[dict]], level_names=None) -> dict:
    level_names = level_names or [f"level_{i}" for i in range(len(door_lists))]
    levels = {
        name: {"zones": [{"doors": doors}]}
        for name, doors in zip(level_names, door_lists)
    }
    return levels


# ---------------------------------------------------------------------------
# _bresenham_line
# ---------------------------------------------------------------------------

def test_bresenham_horizontal():
    assert _bresenham_line(0, 0, 4, 0) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_bresenham_vertical():
    assert _bresenham_line(2, 0, 2, 3) == [(2, 0), (2, 1), (2, 2), (2, 3)]


def test_bresenham_diagonal():
    assert _bresenham_line(0, 0, 3, 3) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_bresenham_reverse_direction():
    pts = _bresenham_line(4, 0, 0, 0)
    assert pts == [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]


def test_bresenham_single_point():
    assert _bresenham_line(5, 5, 5, 5) == [(5, 5)]


@given(
    x0=st.integers(min_value=-20, max_value=20),
    y0=st.integers(min_value=-20, max_value=20),
    x1=st.integers(min_value=-20, max_value=20),
    y1=st.integers(min_value=-20, max_value=20),
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_bresenham_properties(x0, y0, x1, y1):
    pts = _bresenham_line(x0, y0, x1, y1)
    assert pts[0] == (x0, y0)
    assert pts[-1] == (x1, y1)
    # 8-connected: consecutive pixels differ by at most 1 in each axis
    for (px, py), (qx, qy) in zip(pts, pts[1:]):
        assert abs(px - qx) <= 1 and abs(py - qy) <= 1
    # no more steps than the longer axis (+1 for the start pixel)
    assert len(pts) <= max(abs(x1 - x0), abs(y1 - y0)) + 1
    # all pixels stay inside the bounding box
    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    for px, py in pts:
        assert lo_x <= px <= hi_x and lo_y <= py <= hi_y


# ---------------------------------------------------------------------------
# _find_world_yaml
# ---------------------------------------------------------------------------

def test_find_world_yaml_run_dir_zero_subdir_preferred(tmp_path):
    run = tmp_path / "run"
    deep = _write_world(run / "worlds" / "m1" / "0", flat={"name": "deep"})
    _write_world(run / "worlds" / "m1", flat={"name": "shallow"})
    assert _find_world_yaml("m1", run_dir=run) == deep


def test_find_world_yaml_run_dir_flat(tmp_path):
    run = tmp_path / "run"
    flat = _write_world(run / "worlds" / "m2", flat={"name": "flat"})
    assert _find_world_yaml("m2", run_dir=run) == flat


def test_find_world_yaml_ament_share(tmp_path, monkeypatch):
    _install_fake_ament(monkeypatch, tmp_path)
    share_world = _write_world(tmp_path / "share" / "worlds" / "m3" / "0", flat={})
    assert _find_world_yaml("m3", run_dir=None) == share_world


def test_find_world_yaml_ament_failure_ignored(tmp_path, monkeypatch):
    _install_fake_ament(monkeypatch, tmp_path, raise_error=True)
    assert _find_world_yaml("ae_nonesuch_map", run_dir=None) is None


def test_find_world_yaml_ws_root_fallback(tmp_path, monkeypatch):
    _remap_pathlib(monkeypatch, tmp_path)
    ws_world = _write_world(tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "m4", flat={})
    assert _find_world_yaml("m4", run_dir=None) == ws_world


def test_find_world_yaml_ws_root_zero_subdir(tmp_path, monkeypatch):
    _remap_pathlib(monkeypatch, tmp_path)
    ws_world = _write_world(
        tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "m5" / "0", flat={}
    )
    assert _find_world_yaml("m5", run_dir=None) == ws_world


def test_find_world_yaml_not_found(tmp_path):
    # fake name cannot exist in run_dir, ament share, or the real ws roots
    assert _find_world_yaml("ae_nonesuch_map", run_dir=tmp_path / "run") is None


def test_load_world_yaml_cached(tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"a": 1}))
    assert _load_world_yaml(str(path)) == {"a": 1}
    # lru_cache: mutation on disk is not picked up
    path.write_text(yaml.safe_dump({"a": 2}))
    assert _load_world_yaml(str(path)) == {"a": 1}


# ---------------------------------------------------------------------------
# door_segments
# ---------------------------------------------------------------------------

def test_door_segments_no_world_yaml(tmp_path):
    assert door_segments("ae_nonesuch_map", _wall_grid(), 0.05, (0.0, 0.0, 0.0)) == {}


def test_door_segments_corrupt_world_yaml(tmp_path):
    world = _write_world(tmp_path / "run" / "worlds" / "m1" / "0", corrupt=True)
    assert door_segments("m1", _wall_grid(), 0.05, (0.0, 0.0, 0.0), run_dir=tmp_path / "run") == {}
    assert world.exists()


def test_door_segments_levels_layout(tmp_path):
    doors_a = [_door("d1", start=(10.0, 5.0), end=(16.0, 5.0))]
    doors_b = [_door("d2", start=(10.0, 12.0), end=(16.0, 12.0))]
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", levels=_levels_world([doors_a, doors_b]))

    grid = _wall_grid()
    result = door_segments("m1", grid, 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert set(result) == {"level_0/d1", "level_1/d2"}
    for name, (mask, tl) in result.items():
        assert mask.shape == grid.shape
        assert mask.dtype == bool
        assert tl == DEFAULT_DOOR_TL_DB
        assert mask.sum() > 0
    # door masks land on the wall rows
    assert np.all(result["level_0/d1"][0][4:7, 10:17].sum(axis=0) > 0)


def test_door_segments_flat_layout(tmp_path):
    flat = {"zones": [{"doors": [_door("main_door", start=(10.0, 5.0), end=(16.0, 5.0))]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert set(result) == {"world/main_door"}


def test_door_segments_empty_levels_falls_back_to_flat(tmp_path):
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", levels={})
    assert door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run") == {}


def test_door_segments_missing_name_generated(tmp_path):
    flat = {"zones": [{"doors": [_door(start=(10.0, 5.0), end=(16.0, 5.0))]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert set(result) == {"world/door_0"}


def test_door_segments_coords_as_lists(tmp_path):
    flat = {"zones": [{"doors": [_door("l1", start=(10.0, 5.0), end=(16.0, 5.0), coords_as_dict=False)]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert "world/l1" in result


def test_door_segments_custom_tl_db(tmp_path):
    flat = {"zones": [{"doors": [_door("s", start=(10.0, 5.0), end=(16.0, 5.0), tl_db=18.0)]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert result["world/s"][1] == 18.0


def test_door_segments_width_controls_thickness(tmp_path):
    # NOTE: separate run dirs so the lru-cached world.yaml never serves the
    # first (thin) document for the second call.
    thin = {"zones": [{"doors": [_door("thin", start=(10.0, 5.0), end=(16.0, 5.0), width=0.5)]}]}
    thick = {"zones": [{"doors": [_door("thick", start=(10.0, 5.0), end=(16.0, 5.0), width=4.0)]}]}
    _write_world(tmp_path / "run_a" / "worlds" / "m1" / "0", flat=thin)
    _write_world(tmp_path / "run_b" / "worlds" / "m1" / "0", flat=thick)
    thin_result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run_a")
    thick_result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run_b")
    assert thick_result["world/thick"][0].sum() > thin_result["world/thin"][0].sum()


def test_door_segments_off_wall_door_skipped(tmp_path):
    # door placed in free space, away from any wall pixel -> dropped
    flat = {"zones": [{"doors": [_door("float", start=(10.0, 3.0), end=(16.0, 3.0))]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert result == {}


def test_door_segments_out_of_bounds_door_skipped(tmp_path):
    flat = {"zones": [{"doors": [_door("far", start=(200.0, 200.0), end=(210.0, 200.0))]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert result == {}


def test_door_segments_degenerate_point_door_on_wall(tmp_path):
    flat = {"zones": [{"doors": [_door("dot", start=(10.0, 5.0), end=(10.0, 5.0))]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    result = door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert "world/dot" in result
    assert result["world/dot"][0].sum() >= 1


def test_door_segments_no_doors_declared(tmp_path):
    flat = {"zones": [{"doors": []}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    assert door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run") == {}


def test_door_segments_world_without_zones_key(tmp_path):
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat={"name": "plain"})
    assert door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run") == {}


def test_door_segments_missing_start_end_default_to_origin(tmp_path):
    # start/end absent -> (0,0) -> mask at grid origin; no wall there -> skipped
    flat = {"zones": [{"doors": [{"width": 1.0, "kind": "sliding"}]}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)
    assert door_segments("m1", _wall_grid(), 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run") == {}


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
    assert _entity_matches_door("level_a/main_hallway", "env_0/some_zone/main_hallway/0")


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
    tl = build_pixel_tl(grid, {"world/d1": (mask, DEFAULT_DOOR_TL_DB)}, open_doors={"world/d1"})
    assert np.all(tl[mask] == 0.0)
    # wall pixels outside the door still blocked
    assert np.all(tl[5, :10] == 47.0)


def test_build_pixel_tl_open_entity_env_prefixed():
    grid = _wall_grid()
    mask = np.zeros_like(grid, dtype=bool)
    mask[5, 10:16] = True
    tl = build_pixel_tl(grid, {"world/d1": (mask, 25.0)}, open_doors={"env_0/world/d1/7"})
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

    open_set = {f"world/d{i}" for i, fl in enumerate(open_flags) if fl and i * 4 + 1 < width}
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


def test_door_segments_and_build_pixel_tl_integration(tmp_path):
    """End-to-end: world.yaml door geometry carved into the TL map."""
    doors_yaml = [_door("main_door", start=(10.0, 5.0), end=(16.0, 5.0), tl_db=20.0)]
    flat = {"zones": [{"doors": doors_yaml}]}
    _write_world(tmp_path / "run" / "worlds" / "m1" / "0", flat=flat)

    grid = _wall_grid()
    segs = door_segments("m1", grid, 1.0, (0.0, 0.0, 0.0), run_dir=tmp_path / "run")
    assert set(segs) == {"world/main_door"}

    closed = build_pixel_tl(grid, segs)
    mask = segs["world/main_door"][0]
    assert np.all(closed[mask] == 20.0)
    assert closed[mask].max() == 20.0

    opened = build_pixel_tl(grid, segs, open_doors={"env_0/world/main_door/2"})
    assert np.all(opened[mask] == 0.0)
    # neighbours beyond the door remain wall-blocked
    assert np.all(opened[5, 0:10] == 47.0)

