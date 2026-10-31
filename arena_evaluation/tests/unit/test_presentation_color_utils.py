"""Unit tests for the accessibility colour palette (YAML-driven).

Covers ``load_color_palette`` (real YAML files, white/black exclusion,
fallbacks, failure paths) and ``set_global_color_palette`` (the real Plotly
template colourway and Seaborn theme, import-error tolerance).
"""

import sys

import matplotlib as mpl
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from matplotlib.colors import to_hex

from arena_evaluation.presentation.color_utils import (
    get_color_palette,
    load_color_palette,
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


@pytest.fixture
def palette_file(tmp_path):
    def write(content: str):
        path = tmp_path / "color_palette.yaml"
        path.write_text(content)
        return path

    return write


@pytest.fixture
def theme():
    """Restore the process-global Plotly template and Matplotlib cycle a test changes."""
    template = pio.templates["plotly_white"]
    colorway = template.layout.colorway
    default = pio.templates.default
    rc = mpl.rcParams.copy()
    yield
    template.layout.colorway = colorway
    pio.templates["plotly_white"] = template
    pio.templates.default = default
    mpl.rcParams.update(rc)


def _cycle() -> list[str]:
    return [to_hex(color) for color in mpl.rcParams["axes.prop_cycle"].by_key()["color"]]


def test_default_palette_when_config_missing(tmp_path):
    palette = load_color_palette(tmp_path / "missing.yaml")
    assert palette == _DEFAULT_PALETTE
    assert all(p.startswith("#") for p in palette)


def test_palette_is_cached_between_calls():
    assert get_color_palette() is get_color_palette()


def test_palette_loaded_from_yaml_excludes_white_and_black(palette_file):
    path = palette_file("palette:\n  white: '#ffffff'\n  black: '#000000'\n  accent_blue: '#123456'\n  alert_red: '#654321'\n")
    assert load_color_palette(path) == ["#123456", "#654321"]


def test_palette_uses_yaml_key_case_insensitively(palette_file):
    path = palette_file("palette:\n  WHITE: '#ffffff'\n  Black: '#000000'\n  c: '#abcabc'\n")
    assert load_color_palette(path) == ["#abcabc"]


def test_palette_empty_yaml_dict_falls_back_to_default(palette_file):
    # Only white/black declared -> colors list ends up empty -> default.
    assert load_color_palette(palette_file("palette:\n  white: '#ffffff'\n  black: '#000000'\n")) == _DEFAULT_PALETTE


def test_palette_yaml_without_palette_key_falls_back(palette_file):
    assert load_color_palette(palette_file("other: {a: 1}\n")) == _DEFAULT_PALETTE


def test_palette_open_failure_warns_and_falls_back(tmp_path, capsys):
    assert load_color_palette(tmp_path) == _DEFAULT_PALETTE
    assert "Warning: Failed to load color palette" in capsys.readouterr().out


def test_palette_yaml_parse_failure_warns_and_falls_back(palette_file, capsys):
    assert load_color_palette(palette_file("palette: [unclosed\n")) == _DEFAULT_PALETTE
    assert "Warning: Failed to load color palette" in capsys.readouterr().out


def test_set_global_color_palette_applies_plotly_colorway(theme):
    set_global_color_palette(["#123456", "#abcdef"])
    assert pio.templates.default == "plotly_white"
    assert pio.templates["plotly_white"].layout.colorway == ("#123456", "#abcdef")


def test_set_global_color_palette_defaults_to_the_configured_palette(theme):
    set_global_color_palette()
    assert pio.templates["plotly_white"].layout.colorway == tuple(get_color_palette())


def test_set_global_color_palette_creates_template_when_plotly_white_missing(theme):
    del pio.templates["plotly_white"]
    pio.templates.default = None
    set_global_color_palette(["#111111", "#222222"])
    assert pio.templates["plotly_white"].layout.colorway == ("#111111", "#222222")
    assert pio.templates.default == "plotly_white"


def test_set_global_color_palette_sets_colorway_when_template_exists(theme):
    pio.templates["plotly_white"] = go.layout.Template(layout=dict(colorway=("#999999",)))
    set_global_color_palette(["#333333"])
    assert pio.templates["plotly_white"].layout.colorway == ("#333333",)
    assert pio.templates.default == "plotly_white"


def test_set_global_color_palette_skips_plotly_on_import_error(theme, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly.io", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    set_global_color_palette(["#445566"])
    assert _cycle() == ["#445566"]


def test_set_global_color_palette_applies_seaborn_theme(theme):
    set_global_color_palette(["#445566"])
    assert _cycle() == ["#445566"]
    assert mpl.rcParams["axes.grid"] is True


def test_set_global_color_palette_skips_seaborn_on_import_error(theme, monkeypatch):
    before = _cycle()
    monkeypatch.setitem(sys.modules, "plotly.io", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "seaborn", None)
    set_global_color_palette(["#445566"])
    assert _cycle() == before
