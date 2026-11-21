"""Unit tests for the accessibility colour palette (YAML-driven)."""

import matplotlib as mpl
import plotly.io as pio
import pytest
from matplotlib.colors import to_hex

from arena_evaluation.presentation.color_utils import (
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


def test_palette_yaml_parse_failure_warns_and_falls_back(palette_file, capsys):
    assert load_color_palette(palette_file("palette: [unclosed\n")) == _DEFAULT_PALETTE
    assert "Warning: Failed to load color palette" in capsys.readouterr().out


def test_set_global_color_palette_creates_template_when_plotly_white_missing(theme):
    del pio.templates["plotly_white"]
    pio.templates.default = None
    set_global_color_palette(["#111111", "#222222"])
    assert pio.templates["plotly_white"].layout.colorway == ("#111111", "#222222")
    assert pio.templates.default == "plotly_white"
