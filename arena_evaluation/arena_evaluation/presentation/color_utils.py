import pathlib

import yaml

_PALETTE_CACHE = None
_DEFAULT_PALETTE = ["#41b6e6", "#d3273e", "#00bfb2", "#ffc845", "#be84a3", "#dc582a", "#1d4289", "#94a596"]


def load_color_palette(config_path: pathlib.Path) -> list[str]:
    """Hex colors of a palette yaml, in file order, minus pure white and black, the default palette when it has none."""
    if not config_path.exists():
        return list(_DEFAULT_PALETTE)

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        palette_dict = data.get("palette", {})
        colors = []
        for name, hex_code in palette_dict.items():
            if name.lower() not in ["white", "black"]:
                colors.append(hex_code)

        return colors if colors else list(_DEFAULT_PALETTE)
    except Exception as e:
        print(f"Warning: Failed to load color palette from {config_path}: {e}")
        return list(_DEFAULT_PALETTE)


def get_color_palette() -> list[str]:
    """Colors of config/color_palette.yaml, cached."""
    global _PALETTE_CACHE
    if _PALETTE_CACHE is None:
        _PALETTE_CACHE = load_color_palette(pathlib.Path(__file__).resolve().parents[2] / "config" / "color_palette.yaml")
    return _PALETTE_CACHE


def set_global_color_palette(palette: list[str] | None = None) -> None:
    """Apply a palette, the configured one by default, to Plotly and Seaborn globally."""
    if palette is None:
        palette = get_color_palette()

    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        if "plotly_white" in pio.templates:
            layout = pio.templates["plotly_white"].layout
            assert isinstance(layout, go.Layout)
            layout.colorway = palette
        else:
            pio.templates["plotly_white"] = go.layout.Template(layout=dict(colorway=palette))
        pio.templates.default = "plotly_white"
    except ImportError:
        pass

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette=palette)
    except ImportError:
        pass
