import pathlib

import yaml

_PALETTE_CACHE = None


def get_color_palette() -> list[str]:
    """Hex colors from config/color_palette.yaml, in file order, minus pure white and black."""
    global _PALETTE_CACHE
    if _PALETTE_CACHE is not None:
        return _PALETTE_CACHE

    config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / "color_palette.yaml"

    default_palette = ["#41b6e6", "#d3273e", "#00bfb2", "#ffc845", "#be84a3", "#dc582a", "#1d4289", "#94a596"]

    if not config_path.exists():
        _PALETTE_CACHE = default_palette
        return _PALETTE_CACHE

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        palette_dict = data.get("palette", {})
        colors = []
        for name, hex_code in palette_dict.items():
            if name.lower() not in ["white", "black"]:
                colors.append(hex_code)

        _PALETTE_CACHE = colors if colors else default_palette
    except Exception as e:
        print(f"Warning: Failed to load color palette from {config_path}: {e}")
        _PALETTE_CACHE = default_palette

    return _PALETTE_CACHE


def set_global_color_palette() -> None:
    """Apply the accessibility color palette to Plotly and Seaborn globally."""
    palette = get_color_palette()

    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        if "plotly_white" in pio.templates:
            pio.templates["plotly_white"].layout.colorway = palette
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
