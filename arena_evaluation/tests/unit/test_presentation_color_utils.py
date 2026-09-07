"""Unit tests for the accessibility colour palette (YAML-driven).

Covers ``get_color_palette`` (cache, YAML loading, white/black exclusion,
fallbacks, failure paths) and ``set_global_color_palette`` (Plotly template
colourway + Seaborn theme application, import-error tolerance).
"""

import io
import pathlib
import sys
import types

import pytest

from arena_evaluation.presentation import color_utils
from arena_evaluation.presentation.color_utils import (
    get_color_palette,
    set_global_color_palette,
)

_DEFAULT_PALETTE = [
    "#41b6e6",
    "#d3273e",
    "#00bfb2",
    "#ffc845",
    "#be84a3",
    "#dc582a",
    "#1d4289",
    "#94a596",
]


# ── Helpers ────────────────────────────────────────────────────────────────


def _reset_cache(monkeypatch) -> None:
    monkeypatch.setattr(color_utils, "_PALETTE_CACHE", None)


def _fake_path_module(exists: bool = True):
    """Substitute ``color_utils.pathlib`` so the config path exists or not.

    NOTE: the source calls the *builtin* ``open(config_path)`` (not
    ``Path.open``), so file content is faked via ``color_utils.open`` instead.
    """

    class _FakePath(pathlib.PosixPath):
        def exists(self):  # noqa: A003 - override of pathlib API
            return exists

    fake = types.SimpleNamespace(Path=_FakePath)
    return fake


def _fake_open(content: str = "", raise_on_open: bool = False):
    def _fake_open_fn(path, mode="r", *args, **kwargs):
        if raise_on_open:
            raise OSError("simulated I/O failure")
        return io.StringIO(content)

    return _fake_open_fn


def _patch_config_file(monkeypatch, content: str, exists: bool = True):
    monkeypatch.setattr(color_utils, "pathlib", _fake_path_module(exists=exists))
    # The source calls the builtin open(); shadowing it as a module global
    # (raising=False) makes the fake win over the builtin.
    monkeypatch.setattr(color_utils, "open", _fake_open(content), raising=False)


def _restore_real_global_state() -> None:
    """Re-apply the real palette to Plotly/Seaborn after tests that mutated
    the module cache (restores process-global plotly/seaborn settings)."""
    color_utils._PALETTE_CACHE = None
    set_global_color_palette()


# ── get_color_palette ──────────────────────────────────────────────────────


def test_default_palette_when_config_missing(monkeypatch):
    _reset_cache(monkeypatch)
    _patch_config_file(monkeypatch, content="", exists=False)
    palette = get_color_palette()
    assert palette == _DEFAULT_PALETTE
    assert all(p.startswith("#") for p in palette)


def test_palette_is_cached_between_calls(monkeypatch):
    _reset_cache(monkeypatch)
    first = get_color_palette()
    second = get_color_palette()
    assert first is second


def test_palette_loaded_from_yaml_excludes_white_and_black(monkeypatch):
    _reset_cache(monkeypatch)
    yaml_content = "palette:\n  white: '#ffffff'\n  black: '#000000'\n  accent_blue: '#123456'\n  alert_red: '#654321'\n"
    _patch_config_file(monkeypatch, content=yaml_content)
    assert get_color_palette() == ["#123456", "#654321"]


def test_palette_uses_yaml_key_case_insensitively(monkeypatch):
    _reset_cache(monkeypatch)
    yaml_content = "palette:\n  WHITE: '#ffffff'\n  Black: '#000000'\n  c: '#abcabc'\n"
    _patch_config_file(monkeypatch, content=yaml_content)
    assert get_color_palette() == ["#abcabc"]


def test_palette_empty_yaml_dict_falls_back_to_default(monkeypatch):
    _reset_cache(monkeypatch)
    # Only white/black declared -> colors list ends up empty -> default.
    _patch_config_file(
        monkeypatch,
        content="palette:\n  white: '#ffffff'\n  black: '#000000'\n",
    )
    assert get_color_palette() == _DEFAULT_PALETTE


def test_palette_yaml_without_palette_key_falls_back(monkeypatch):
    _reset_cache(monkeypatch)
    _patch_config_file(monkeypatch, content="other: {a: 1}\n")
    assert get_color_palette() == _DEFAULT_PALETTE


def test_palette_open_failure_warns_and_falls_back(monkeypatch, capsys):
    _reset_cache(monkeypatch)
    _patch_config_file(monkeypatch, content="palette: {}\n")
    monkeypatch.setattr(color_utils, "open", _fake_open(raise_on_open=True), raising=False)
    assert get_color_palette() == _DEFAULT_PALETTE
    out = capsys.readouterr().out
    assert "Warning: Failed to load color palette" in out


def test_palette_yaml_parse_failure_warns_and_falls_back(monkeypatch, capsys):
    _reset_cache(monkeypatch)

    def _boom(*args, **kwargs):
        raise ValueError("bad yaml")

    monkeypatch.setattr(color_utils.yaml, "safe_load", _boom)
    _patch_config_file(monkeypatch, content="palette: {}\n")
    assert get_color_palette() == _DEFAULT_PALETTE
    out = capsys.readouterr().out
    assert "Warning: Failed to load color palette" in out


# ── set_global_color_palette (real Plotly/Seaborn) ─────────────────────────


def test_set_global_color_palette_applies_plotly_colorway(monkeypatch):
    import plotly.io as pio

    _reset_cache(monkeypatch)
    monkeypatch.setattr(color_utils, "_PALETTE_CACHE", ["#123456", "#abcdef"])
    set_global_color_palette()
    assert pio.templates.default == "plotly_white"
    assert pio.templates["plotly_white"].layout.colorway == ("#123456", "#abcdef")
    _restore_real_global_state()


def test_set_global_color_palette_does_not_raise_with_real_libs(monkeypatch):
    _reset_cache(monkeypatch)
    set_global_color_palette()  # must not raise


# ── set_global_color_palette (fake modules → branch coverage) ──────────────


class _TemplateStore:
    """dict-like stand-in for pio.templates with a ``default`` attribute."""

    def __init__(self, initial=None):
        self._d = dict(initial or {})
        self.default = "plotly_dark"

    def __contains__(self, key):
        return key in self._d

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value


def test_set_global_color_palette_creates_template_when_plotly_white_missing(monkeypatch):
    import plotly.io as pio

    _reset_cache(monkeypatch)
    monkeypatch.setattr(color_utils, "_PALETTE_CACHE", ["#111111", "#222222"])
    store = _TemplateStore()
    monkeypatch.setattr(pio, "templates", store)
    set_global_color_palette()
    assert store["plotly_white"].layout["colorway"] == ("#111111", "#222222")
    assert store.default == "plotly_white"


def test_set_global_color_palette_sets_colorway_when_template_exists(monkeypatch):
    import plotly.graph_objects as go
    import plotly.io as pio

    _reset_cache(monkeypatch)
    monkeypatch.setattr(color_utils, "_PALETTE_CACHE", ["#333333"])
    existing = go.layout.Template(layout=dict(colorway=("#999999",)))
    store = _TemplateStore(initial={"plotly_white": existing})
    monkeypatch.setattr(pio, "templates", store)
    set_global_color_palette()
    assert store["plotly_white"].layout["colorway"] == ("#333333",)
    assert store.default == "plotly_white"


def test_set_global_color_palette_skips_plotly_on_import_error(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setitem(sys.modules, "plotly.io", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    set_global_color_palette()  # must not raise; seaborn block still runs


def test_set_global_color_palette_applies_seaborn_theme(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(color_utils, "_PALETTE_CACHE", ["#445566"])
    captured = {}

    def _fake_set_theme(**kwargs):
        captured.update(kwargs)

    fake_sns = types.SimpleNamespace(set_theme=_fake_set_theme)
    monkeypatch.setitem(sys.modules, "seaborn", fake_sns)
    set_global_color_palette()
    assert captured["style"] == "whitegrid"
    assert captured["palette"] == ["#445566"]


def test_set_global_color_palette_skips_seaborn_on_import_error(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setitem(sys.modules, "plotly.io", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "seaborn", None)
    set_global_color_palette()  # must not raise
