"""Unit tests for map discovery / PGM->PNG caching (processing/map_registry.py).

All filesystem state is confined to tmp_path: worlds come in through a run
directory or $ARENA_DIR, the cache through $ARENA_DATA_DIR or an explicit path,
and `rospack` answers through `subprocess.run`, the process boundary.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from PIL import Image

from arena_evaluation.processing import map_registry as mr
from arena_evaluation.processing.map_registry import MapRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Proc:
    """Minimal subprocess.CompletedProcess stand-in."""

    def __init__(self, stdout: str):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _write_pgm(path: pathlib.Path, width: int, height: int, gray: int = 200) -> None:
    """Synthesize a tiny binary (P5) PGM in-test."""
    header = f"P5\n{width} {height}\n255\n".encode()
    path.write_bytes(header + bytes([gray]) * (width * height))


def _write_world_yaml(base: pathlib.Path, subdir: str | None, image: str = "map.pgm", resolution: float = 0.05, origin: list[float] | None = None) -> pathlib.Path:
    target = base / subdir if subdir else base
    target.mkdir(parents=True, exist_ok=True)
    meta: dict = {"image": image, "resolution": resolution}
    if origin is not None:
        meta["origin"] = origin
    yaml_path = target / "map.yaml"
    yaml_path.write_text(yaml.safe_dump(meta))
    return target


def _run_world(tmp_path: pathlib.Path, name: str = "ae_map_r1") -> pathlib.Path:
    return tmp_path / "run" / "worlds" / name


def _get_map(tmp_path: pathlib.Path, name: str = "ae_map_r1") -> dict | None:
    return MapRegistry.get_map(name, cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")


# ---------------------------------------------------------------------------
# _find_ros_map_dir
# ---------------------------------------------------------------------------


def test_find_ros_map_dir_rospack_hit(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    map_dir = pkg / "worlds" / "ae_map_r1"
    map_dir.mkdir(parents=True)
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    assert MapRegistry._find_ros_map_dir("ae_map_r1") == map_dir


def test_find_ros_map_dir_rospack_hit_but_dir_missing_falls_to_arena_dir(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    map_dir = tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "ae_map_r2"
    map_dir.mkdir(parents=True)
    monkeypatch.setenv("ARENA_DIR", str(tmp_path / "src" / "Arena"))
    assert MapRegistry._find_ros_map_dir("ae_map_r2") == map_dir


def test_find_ros_map_dir_rospack_failure_falls_to_arena_dir(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("rospack not installed")

    monkeypatch.setattr(mr.subprocess, "run", _boom)
    map_dir = tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "ae_map_r3"
    map_dir.mkdir(parents=True)
    monkeypatch.setenv("ARENA_DIR", str(tmp_path / "src" / "Arena"))
    assert MapRegistry._find_ros_map_dir("ae_map_r3") == map_dir


def test_find_ros_map_dir_not_found(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    assert MapRegistry._find_ros_map_dir("ae_map_nonesuch_xyz") is None


# ---------------------------------------------------------------------------
# _world_preview
# ---------------------------------------------------------------------------


def _write_format2_world(base: pathlib.Path, level_ids: list[str]) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    levels = [{"id": level_id, "zones": [{"name": f"room_{level_id}", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]}]} for level_id in level_ids]
    (base / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": levels}))
    return base


def test_world_preview_single_level_map(tmp_path):
    world = _write_format2_world(tmp_path / "world", ["0"])
    preview = _write_world_yaml(world, "map")
    (preview / "map.yaml").rename(preview / "0.yaml")
    assert MapRegistry._world_preview(world) == world / "map" / "0.yaml"


def test_world_preview_without_preview_file(tmp_path):
    world = _write_format2_world(tmp_path / "world", ["0"])
    assert MapRegistry._world_preview(world) is None


def test_world_preview_multi_level_world(tmp_path):
    world = _write_format2_world(tmp_path / "world", ["0", "1"])
    for level_id in ("0", "1"):
        preview = _write_world_yaml(world, "map")
        (preview / "map.yaml").rename(preview / f"{level_id}.yaml")
    assert MapRegistry._world_preview(world) is None


def test_world_preview_directory_without_world(tmp_path):
    assert MapRegistry._world_preview(tmp_path) is None


def test_get_map_reads_single_level_world_preview(tmp_path):
    world = _write_format2_world(tmp_path / "run" / "worlds" / "ae_map_r1", ["0"])
    preview = _write_world_yaml(world, "map", image="0.pgm", resolution=0.1)
    (preview / "map.yaml").rename(preview / "0.yaml")
    _write_pgm(preview / "0.pgm", 4, 3)
    meta = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path / "cache", run_dir=tmp_path / "run")
    assert meta is not None
    assert meta["resolution"] == 0.1
    assert (meta["width"], meta["height"]) == (4, 3)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [None, "", "unknown"])
def test_get_map_invalid_names(name, tmp_path):
    assert MapRegistry.get_map(name, cache_dir=tmp_path) is None


def test_get_map_cache_hit_returns_cached_meta(tmp_path):
    meta = {
        "png_path": str(tmp_path / "ae_map_r1.png"),
        "resolution": 0.05,
        "origin": [0.0, 0.0, 0.0],
        "width": 4,
        "height": 3,
    }
    (tmp_path / "ae_map_r1.png").write_bytes(b"fake png bytes")
    (tmp_path / "ae_map_r1.yaml").write_text(yaml.safe_dump(meta))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) == meta


def test_get_map_cache_partial_hit_ignored(tmp_path):
    # PNG exists but no YAML -> treated as cache miss
    (tmp_path / "ae_map_nonesuch_xyz.png").write_bytes(b"fake")
    assert MapRegistry.get_map("ae_map_nonesuch_xyz", cache_dir=tmp_path) is None


def test_get_map_no_world_dir(tmp_path):
    assert MapRegistry.get_map("ae_map_nonesuch_xyz", cache_dir=tmp_path) is None


def test_get_map_world_dir_without_map_yaml(tmp_path):
    d = _run_world(tmp_path, "ae_map_nonesuch_xyz")
    d.mkdir(parents=True)
    (d / "map.pgm").write_bytes(b"x")
    assert _get_map(tmp_path, "ae_map_nonesuch_xyz") is None


def test_get_map_image_file_missing(tmp_path):
    _write_world_yaml(_run_world(tmp_path), "map", image="map.pgm")
    assert _get_map(tmp_path) is None


def test_get_map_pgm_to_png_conversion(tmp_path):
    world = _write_world_yaml(_run_world(tmp_path), "map", image="map.pgm", resolution=0.1, origin=[1.0, 2.0, 0.0])
    _write_pgm(world / "map.pgm", 8, 6)

    meta = _get_map(tmp_path)
    assert meta is not None
    png_path = tmp_path / "cache" / "ae_map_r1.png"
    assert png_path.exists()
    img = Image.open(png_path)
    assert img.mode == "RGBA"
    assert (img.width, img.height) == (8, 6)

    assert meta["png_path"] == str(png_path)
    assert meta["resolution"] == 0.1
    assert meta["origin"] == [1.0, 2.0, 0.0]
    assert meta["width"] == 8
    assert meta["height"] == 6

    # Second call now served from cache (returns raw cached yaml content).
    cached = _get_map(tmp_path)
    assert cached == meta


def test_get_map_default_resolution_and_origin(tmp_path):
    world = _write_world_yaml(_run_world(tmp_path), "map", image="map.pgm")
    _write_pgm(world / "map.pgm", 5, 5)
    meta = _get_map(tmp_path)
    assert meta["resolution"] == 0.05
    assert meta["origin"] == [0.0, 0.0, 0.0]


def test_get_map_default_image_name(tmp_path):
    # yaml without "image" key -> default "map.pgm"
    world = _write_world_yaml(_run_world(tmp_path), "map")
    _write_pgm(world / "map.pgm", 3, 3)
    assert _get_map(tmp_path) is not None


def test_get_map_candidate_order_map_subdir_preferred(tmp_path):
    world_dir = _run_world(tmp_path)
    sub = _write_world_yaml(world_dir, "map", resolution=0.5)
    _write_pgm(sub / "map.pgm", 3, 3)
    _write_world_yaml(world_dir, None, resolution=0.01)  # flat fallback candidate
    meta = _get_map(tmp_path)
    assert meta["resolution"] == 0.5


def test_get_map_origin_coerced_to_floats(tmp_path):
    world = _write_world_yaml(_run_world(tmp_path), "map", origin=[1, -2, 0])  # ints in yaml
    _write_pgm(world / "map.pgm", 3, 3)
    meta = _get_map(tmp_path)
    assert meta["origin"] == [1.0, -2.0, 0.0]


def test_get_map_conversion_failure(tmp_path, capsys):
    world = _write_world_yaml(_run_world(tmp_path), "map")
    (world / "map.pgm").write_bytes(b"this is definitely not an image")
    assert _get_map(tmp_path) is None
    assert not (tmp_path / "cache" / "ae_map_r1.png").exists()
    assert not (tmp_path / "cache" / "ae_map_r1.yaml").exists()
    assert "Failed to cache map" in capsys.readouterr().out


def test_get_map_default_cache_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))
    assert MapRegistry.get_map("ae_map_nonesuch_xyz") is None
    assert (tmp_path / "data" / "maps_cache").is_dir()
