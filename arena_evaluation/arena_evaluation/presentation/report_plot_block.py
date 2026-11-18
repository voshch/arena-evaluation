"""Shared plot block and inline Plotly bundle for the two report writers."""

from __future__ import annotations

import base64
import html
import pathlib
from typing import Final

import plotly.offline

_PNG_DATA_URI_PREFIX: Final[str] = "data:image/png;base64,"

_PLOT_BLOCK_STYLES: Final[str] = (
    "<style>"
    ".plot-block{margin:1rem 0;padding:1rem;border:1px solid #ddd;"
    "border-radius:6px}"
    ".plot-static{display:none}"
    "@media print{.plot-interactive{display:none!important}"
    ".plot-static{display:block!important}}"
    "</style>"
    "<noscript><style>"
    ".plot-interactive{display:none!important}"
    ".plot-static{display:block!important}"
    "</style></noscript>"
)


def plotly_bundle_script() -> str:
    """Return the Plotly bundle verbatim as one inline script element."""
    return f"<script>{plotly.offline.get_plotlyjs()}</script>"


def plot_block_styles() -> str:
    """Return the CSS that shows the static PNG in print and without scripts."""
    return _PLOT_BLOCK_STYLES


def png_data_uri(png_path: pathlib.Path) -> str:
    """Return the inline data URI of one static PNG file."""
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"{_PNG_DATA_URI_PREFIX}{encoded}"


def plot_block(
    *,
    plot_id: str,
    title: str,
    interactive_html: str,
    png_path: pathlib.Path,
) -> str:
    """Build one plot block: interactive Plotly figure plus its static PNG."""
    label = html.escape(title or plot_id)
    return (
        f"<figure class='plot-block' data-plot-id='{html.escape(plot_id)}'>"
        f"<div class='plot-interactive'>{interactive_html}</div>"
        f"<div class='plot-static'>"
        f"<img src='{png_data_uri(png_path)}' alt='{label} static render'/>"
        "</div></figure>"
    )
