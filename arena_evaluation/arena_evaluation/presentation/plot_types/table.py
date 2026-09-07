"""HTML table plot renderer for aggregated metric columns and notes."""

from __future__ import annotations

import html as html_escape
import pathlib
import typing

import polars as pl

from arena_evaluation.presentation.plot_types.base import BasePlotRenderer

if typing.TYPE_CHECKING:
    from arena_evaluation.presentation.report_builder import ReportBuilder  # noqa: F401


def _load_notes(notes, benchmark_dir: pathlib.Path | None) -> list[dict[str, str]]:
    """Normalize notes input into a list of label/value mappings."""
    rows: list[dict[str, str]] = []
    if notes is None:
        return rows
    if isinstance(notes, list):
        for item in notes:
            if isinstance(item, dict):
                rows.append(
                    {
                        "label": str(item.get("label", item.get("key", ""))),
                        "value": str(item.get("value", "")),
                    }
                )
        return rows
    if isinstance(notes, str):
        notes = notes.strip()
        if not notes:
            return rows

        path: pathlib.Path | None = None
        if "\n" not in notes and len(notes) < 256:
            try:
                candidate = pathlib.Path(notes)
                if candidate.is_file():
                    path = candidate
                elif benchmark_dir is not None and (benchmark_dir / candidate).is_file():
                    path = benchmark_dir / candidate
            except (OSError, ValueError):
                path = None

        if path is not None:
            text = path.read_text()
        else:
            text = notes  # inline YAML or text

        try:
            import yaml

            data = yaml.safe_load(text)
            if isinstance(data, list):
                return _load_notes(data, benchmark_dir)
            if isinstance(data, dict):
                return [{"label": str(k), "value": str(v)} for k, v in data.items()]
        except Exception:
            pass

        # Fallback: "key: value" text lines (agent-friendly free text).
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                rows.append({"label": key.strip(), "value": value.strip()})
            else:
                rows.append({"label": "", "value": line})
    return rows


def _render_notes_callout(notes_rows: list[dict[str, str]]) -> str:
    """Render agent notes as a standalone callout section (never table rows)."""
    if not notes_rows:
        return ""
    parts = [
        "<div class='notes-callout'>",
        "<div class='notes-callout-title'>Analysis Notes</div>",
    ]
    for note in notes_rows:
        label = html_escape.escape(note.get("label", ""), quote=True)
        value = html_escape.escape(note.get("value", ""), quote=True).replace("\n", "<br>")
        if label:
            parts.append(
                f"<div class='notes-callout-row'>"
                f"<span class='notes-callout-label'>{label}</span>"
                f"<span class='notes-callout-value'>{value}</span>"
                f"</div>"
            )
        else:
            parts.append(
                f"<div class='notes-callout-row notes-callout-row-plain'>"
                f"<span class='notes-callout-value'>{value}</span></div>"
            )
    parts.append("</div>")
    return "".join(parts)


class TableRenderer(BasePlotRenderer):
    PLOT_TYPE = "table"
    run_dir: pathlib.Path | None = None  # set by the renderer dispatcher (notes resolution)

    def _columns(self) -> list[dict[str, str]]:
        cols = (self.spec.options or {}).get("columns") or []
        return [
            {"metric": str(c.get("metric", "")), "label": str(c.get("label", c.get("metric", ""))),
             "format": str(c.get("format", "{:.2f}"))}
            for c in cols
            if isinstance(c, dict) and c.get("metric")
        ]

    def _agent_rows(self) -> list[dict[str, str]]:
        """Agent-authored rows: exactly as given, no predefined layout."""
        rows = (self.spec.options or {}).get("rows") or []
        out: list[dict[str, str]] = []
        for item in rows:
            if isinstance(item, dict):
                out.append({
                    "label": str(item.get("label", item.get("key", ""))),
                    "value": str(item.get("value", "")),
                })
        return out

    def _data_rows(self, df: pl.DataFrame) -> list[list[str]]:
        """Aggregate the metrics frame per group_by into label/value rows."""
        group_by = self.spec.options.get("group_by") or []
        if isinstance(group_by, str):
            group_by = [group_by]
        group_cols = [g for g in group_by if g in df.columns]
        cols = self._columns()
        if not group_cols or not cols:
            return []

        list_cols = [c for c in [*group_cols, *(c["metric"] for c in cols)] if c in df.columns and df.schema[c] == pl.List]
        if list_cols:
            need = [c for c in [*group_cols, *(c["metric"] for c in cols)] if c in df.columns]
            df = df.select(need).explode(list_cols)
        df = self._apply_row_filters(df)

        agg = [
            pl.col(c["metric"]).mean().alias(f"__m{i}__")
            for i, c in enumerate(cols)
            if c["metric"] in df.columns
        ]
        if not agg:
            return []
        grouped = df.group_by(group_cols).agg(agg).sort(group_cols)

        rows: list[list[str]] = []
        for row in grouped.iter_rows(named=True):
            key = " / ".join(str(row.get(g, "")) for g in group_cols)
            values = []
            for i, c in enumerate(cols):
                val = row.get(f"__m{i}__")
                try:
                    values.append(c["format"].format(float(val)) if val is not None else "N/A")
                except (TypeError, ValueError):
                    values.append("N/A")
            rows.append([key, *values])
        return rows

    def _render_data_table(self, df: pl.DataFrame) -> str | None:
        opts = self.spec.options or {}
        cols = self._columns()
        group_cols = opts.get("group_by") or []
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        group_cols = [g for g in group_cols if g in df.columns]

        header = [*[g.replace("_", " ").title() for g in group_cols], *[c["label"] for c in cols]]
        if not header:
            return None
        rows = self._data_rows(df)
        if not rows:
            return None

        html = ["<table class='dataframe' style='border-collapse: collapse; margin: 8px 0;'>"]
        html.append("<thead><tr>" + "".join(f"<th style='border:1px solid #ccc; padding:4px 10px; background:#f5f5f5;'>{h}</th>" for h in header) + "</tr></thead>")
        html.append("<tbody>")
        for row in rows:
            cells = row + [""] * (len(header) - len(row))
            html.append("<tr>" + "".join(f"<td style='border:1px solid #ccc; padding:4px 10px;'>{c}</td>" for c in cells) + "</tr>")
        html.append("</tbody></table>")
        return "".join(html)

    def _render_agent_table(self) -> str | None:
        """Agent-authored rows as a clean two-column table."""
        rows = self._agent_rows()
        if not rows:
            return None
        parts = ["<table class='dataframe' style='border-collapse: collapse; margin: 8px 0;'>",
                 "<thead><tr><th style='border:1px solid #ccc; padding:4px 10px; background:#f5f5f5;'>Label</th>"
                 "<th style='border:1px solid #ccc; padding:4px 10px; background:#f5f5f5;'>Value</th></tr></thead>",
                 "<tbody>"]
        for row in rows:
            label = html_escape.escape(row.get("label", ""), quote=True)
            value = html_escape.escape(row.get("value", ""), quote=True).replace("\n", "<br>")
            parts.append(f"<tr><td style='border:1px solid #ccc; padding:4px 10px;'>{label}</td>"
                         f"<td style='border:1px solid #ccc; padding:4px 10px;'>{value}</td></tr>")
        parts.append("</tbody></table>")
        return "".join(parts)

    def _render_html(self, df: pl.DataFrame, benchmark_dir: pathlib.Path | None) -> str | None:
        opts = self.spec.options or {}
        data_table = self._render_data_table(df)
        agent_table = self._render_agent_table()
        note_rows = _load_notes(opts.get("notes"), benchmark_dir)
        notes_callout = _render_notes_callout(note_rows)

        pieces = [p for p in (data_table, agent_table, notes_callout) if p]
        if not pieces:
            return None
        return "".join(pieces)

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        df_filtered = self._apply_filters(df)
        return self._render_html(df_filtered, self.run_dir)

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pass  # Tables render as HTML only (like timeseries).
