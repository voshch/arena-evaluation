"""Structural checks of a rendered report page, read with an HTML parser."""

from __future__ import annotations

from html.parser import HTMLParser

PNG_DATA_URI_PREFIX = "data:image/png;base64,"
_EXTERNAL_PREFIXES = ("data:", "http://", "https://", "#")


class PageParts(HTMLParser):
    """Count the figures, PNG images and bundles of a page; list relative references.

    Script bodies are kept as text and never matched as markup, so a bundle that
    contains HTML-looking strings does not change the counts.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self.plotly_figures = 0
        self.png_images = 0
        self.references: list[str] = []
        self._script_text: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "script":
            self._script_text = []
        if tag == "img" and values.get("src", "").startswith(PNG_DATA_URI_PREFIX):
            self.png_images += 1
        if tag == "div" and "plotly-graph-div" in values.get("class", "").split():
            self.plotly_figures += 1
        for name in ("src", "href"):
            target = values.get(name, "")
            if target and not target.startswith(_EXTERNAL_PREFIXES):
                self.references.append(target)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script_text is not None:
            self.scripts.append("".join(self._script_text))
            self._script_text = None

    def handle_data(self, data: str) -> None:
        if self._script_text is not None:
            self._script_text.append(data)

    @property
    def bundle_count(self) -> int:
        """Number of script elements that carry the Plotly bundle banner."""
        return sum("plotly.js v" in script for script in self.scripts)


def parse_page(page: str) -> PageParts:
    """Parse a rendered page and return its structural parts."""
    parts = PageParts()
    parts.feed(page)
    parts.close()
    return parts
