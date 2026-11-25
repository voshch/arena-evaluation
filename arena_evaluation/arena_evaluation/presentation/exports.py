"""LaTeX export of the report a build actually approved: its tables and its figures."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import TYPE_CHECKING

from arena_evaluation.presentation.plot_types.table import TableRenderer

if TYPE_CHECKING:
    from .report_builder import ReportBuilder

__all__ = ["LatexExport", "export_report_latex", "latex_escape", "table_to_booktabs"]

_TEXT_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LatexExport:
    """The written LaTeX file and the report blocks that made it."""

    path: pathlib.Path
    tables: tuple[str, ...]
    figures: tuple[str, ...]


def latex_escape(value: str) -> str:
    """Escape the characters LaTeX treats specially, so a report value compiles."""
    flattened = str(value).replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    return "".join(_TEXT_ESCAPES.get(character, character) for character in flattened)


def table_to_booktabs(headers: list[str], rows: list[list[str]], *, caption: str | None = None) -> str:
    """A booktabs tabular with every header and cell escaped."""
    escaped_headers = [latex_escape(header) for header in headers]
    body = "\n".join(" & ".join(latex_escape(cell) for cell in row) + r" \\" for row in rows)
    caption_line = f"\\caption{{{latex_escape(caption)}}}\n" if caption else ""
    return f"\\begin{{table}}[htbp]\n\\centering\n{caption_line}\\begin{{tabular}}{{{'l' * len(escaped_headers)}}}\n\\toprule\n{' & '.join(escaped_headers)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}"


def _figure_block(caption: str, relative_path: str) -> str:
    return f"\\begin{{figure}}[htbp]\n\\centering\n\\includegraphics[width=\\linewidth]{{\\detokenize{{{relative_path}}}}}\n\\caption{{{latex_escape(caption)}}}\n\\end{{figure}}"


def export_report_latex(builder: ReportBuilder, output_path: pathlib.Path) -> LatexExport:
    """Write one LaTeX document with the report's approved tables and rendered figures."""
    build = builder.build_result
    if build is None:
        raise RuntimeError("The report must be built before it can be exported; call build() first.")
    output = pathlib.Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = builder.manifest
    specs = {spec.id: spec for spec in manifest.plots}
    table_ids: list[str] = []
    figure_ids: list[str] = []
    blocks: list[str] = [r"\section*{" + latex_escape(manifest.title or builder.report_name) + "}"]

    for record in build.plots:
        spec = specs.get(record.plot_id)
        if not record.passed or spec is None:
            continue
        resolved = builder.resolved_spec(spec)
        if resolved.type == "table":
            frame, diagnostics = builder.resolve_plot(spec)
            if diagnostics.status == "FAILED":
                continue
            headers, rows = TableRenderer(resolved).tableau(frame)
            blocks.append(table_to_booktabs(headers, rows, caption=resolved.title or resolved.id))
            table_ids.append(record.plot_id)
        else:
            png = builder.plots_dir / f"{record.plot_id}.png"
            if not png.is_file():
                continue
            relative = pathlib.Path(os.path.relpath(png, output.parent)).as_posix()
            blocks.append(_figure_block(resolved.title or resolved.id, relative))
            figure_ids.append(record.plot_id)

    if not figure_ids and not table_ids:
        raise RuntimeError("Nothing approved was rendered, so there is no LaTeX report to export.")
    document = "\\documentclass[10pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\usepackage[T1]{fontenc}\n\\begin{document}\n" + "\n\n".join(blocks) + "\n\\end{document}\n"
    output.write_text(document, encoding="utf-8")
    return LatexExport(path=output, tables=tuple(table_ids), figures=tuple(figure_ids))
