"""Unit tests for map discovery / PGM->PNG caching (processing/map_registry.py).

All filesystem state is confined to tmp_path. The module's `pathlib.Path`
attribute is swapped for a remapping proxy where the absolute /opt/arena_ws
locations need to be exercised, so no real workspace directories are touched.
"""

from __future__ import annotations

import pathlib
import types

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


def _remap_pathlib(monkeypatch, module, tmp_path: pathlib.Path) -> None:
    """Swap a module's `pathlib.Path` so /opt/arena_ws (and /home/nelson/arena_ws)
    prefixes resolve under tmp_path. Only the target module's binding is touched."""
    real_path = pathlib.Path

    def _patched(s):
        s = str(s)
        for prefix in ("/opt/arena_ws", "/home/nelson/arena_ws"):
            if s.startswith(prefix):
                return real_path(str(tmp_path) + s[len(prefix) :])
        return real_path(s)

    monkeypatch.setattr(module, "pathlib", types.SimpleNamespace(Path=_patched))


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


# ---------------------------------------------------------------------------
# _find_ros_map_dir
# ---------------------------------------------------------------------------


def test_find_ros_map_dir_rospack_hit(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    map_dir = pkg / "worlds" / "ae_map_r1"
    map_dir.mkdir(parents=True)
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    assert MapRegistry._find_ros_map_dir("ae_map_r1") == map_dir


def test_find_ros_map_dir_rospack_hit_but_dir_missing_falls_to_ws_root(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    map_dir = tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "ae_map_r2"
    map_dir.mkdir(parents=True)
    _remap_pathlib(monkeypatch, mr, tmp_path)
    assert MapRegistry._find_ros_map_dir("ae_map_r2") == map_dir


def test_find_ros_map_dir_rospack_failure_ws_root(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("rospack not installed")

    monkeypatch.setattr(mr.subprocess, "run", _boom)
    map_dir = tmp_path / "src" / "Arena" / "arena_simulation_setup" / "worlds" / "ae_map_r3"
    map_dir.mkdir(parents=True)
    _remap_pathlib(monkeypatch, mr, tmp_path)
    assert MapRegistry._find_ros_map_dir("ae_map_r3") == map_dir


def test_find_ros_map_dir_not_found(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Proc(str(pkg)))
    assert MapRegistry._find_ros_map_dir("ae_map_nonesuch_xyz") is None


# ---------------------------------------------------------------------------
# get_map — guards and cache
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


def test_get_map_cache_partial_hit_ignored(tmp_path, monkeypatch):
    # PNG exists but no YAML -> treated as cache miss
    (tmp_path / "ae_map_r1.png").write_bytes(b"fake")
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: None))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is None


def test_get_map_no_world_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: None))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is None


def test_get_map_world_dir_without_map_yaml(tmp_path, monkeypatch):
    d = tmp_path / "world"
    d.mkdir()
    (d / "map.pgm").write_bytes(b"x")
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: d))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is None


def test_get_map_image_file_missing(tmp_path, monkeypatch):
    world = _write_world_yaml(tmp_path, "map", image="map.pgm")
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is None


def test_get_map_pgm_to_png_conversion(tmp_path, monkeypatch):
    world = _write_world_yaml(tmp_path, "map", image="map.pgm", resolution=0.1, origin=[1.0, 2.0, 0.0])
    _write_pgm(world / "map.pgm", 8, 6)
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))

    meta = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path)
    assert meta is not None
    png_path = tmp_path / "ae_map_r1.png"
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
    cached = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path)
    assert cached == meta


def test_get_map_default_resolution_and_origin(tmp_path, monkeypatch):
    world = _write_world_yaml(tmp_path, "map", image="map.pgm")
    _write_pgm(world / "map.pgm", 5, 5)
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))
    meta = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path)
    assert meta["resolution"] == 0.05
    assert meta["origin"] == [0.0, 0.0, 0.0]


def test_get_map_default_image_name(tmp_path, monkeypatch):
    # yaml without "image" key -> default "map.pgm"
    world = _write_world_yaml(tmp_path, "map")
    _write_pgm(world / "map.pgm", 3, 3)
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is not None


def test_get_map_candidate_order_map_subdir_preferred(tmp_path, monkeypatch):
    world_dir = tmp_path / "world"
    sub = _write_world_yaml(world_dir, "map", resolution=0.5)
    _write_pgm(sub / "map.pgm", 3, 3)
    _write_world_yaml(world_dir, None, resolution=0.01)  # flat fallback candidate
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world_dir))
    meta = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path)
    assert meta["resolution"] == 0.5


def test_get_map_origin_coerced_to_floats(tmp_path, monkeypatch):
    world = _write_world_yaml(tmp_path, "map", origin=[1, -2, 0])  # ints in yaml
    _write_pgm(world / "map.pgm", 3, 3)
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))
    meta = MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path)
    assert meta["origin"] == [1.0, -2.0, 0.0]


def test_get_map_conversion_failure(tmp_path, monkeypatch, capsys):
    world = _write_world_yaml(tmp_path, "map")
    (world / "map.pgm").write_bytes(b"this is definitely not an image")
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: world.parent))
    assert MapRegistry.get_map("ae_map_r1", cache_dir=tmp_path) is None
    assert not (tmp_path / "ae_map_r1.png").exists()
    assert not (tmp_path / "ae_map_r1.yaml").exists()
    assert "Failed to cache map" in capsys.readouterr().out


def test_get_map_default_cache_dir_created(tmp_path, monkeypatch):
    # cache_dir=None -> /opt/arena_ws/data/maps_cache, remapped into tmp_path
    monkeypatch.setattr(MapRegistry, "_find_ros_map_dir", staticmethod(lambda n: None))
    _remap_pathlib(monkeypatch, mr, tmp_path)
    assert MapRegistry.get_map("ae_map_r1") is None
    assert (tmp_path / "data" / "maps_cache").is_dir()
