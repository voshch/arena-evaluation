#!/usr/bin/env python3
"""Diff a legacy report manifest against its hand-migrated replacement by plot intent."""

from __future__ import annotations

import argparse
import pathlib
import re

import yaml

from arena_evaluation.presentation.legacy_manifest import (
    LegacyPlot,
    load_legacy_manifest,
    load_metric_map,
)
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_TOKENS = frozenset({"the", "by", "and", "of", "vs", "with", "local", "planner", "per"})
_TYPE_FAMILY = {
    "box": "distribution",
    "violin": "distribution",
    "histogram": "distribution",
    "bar": "aggregate",
    "table": "table",
    "heatmap": "matrix",
    "radar": "profile",
    "scatter": "scatter",
    "line": "series",
    "timeseries": "series",
    "trajectory": "trajectory",
}
_MATCH_THRESHOLD = 0.34


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN.findall(text.lower()) if token not in _STOP_TOKENS}


def _legacy_metrics(plot: LegacyPlot) -> set[str]:
    names: set[str] = set()
    if plot.data_key and plot.data_key != "*":
        names.add(plot.data_key)
    for key in ("metrics", "y"):
        value = plot.options.get(key)
        values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple)) else ()
        names.update(str(v) for v in values if isinstance(v, str))
    raw_columns = plot.options.get("columns")
    for column in raw_columns if isinstance(raw_columns, (list, tuple)) else ():
        if isinstance(column, dict) and column.get("metric"):
            names.add(str(column["metric"]))
    return names


def _new_metrics(spec: PlotSpec) -> set[str]:
    names: set[str] = set()
    params = (spec.data.params or {}) if spec.data is not None else {}
    raw_metrics = params.get("metrics")
    for metric in raw_metrics if isinstance(raw_metrics, (list, tuple)) else ():
        names.add(str(metric))
    if isinstance(params.get("metric"), str):
        names.add(str(params["metric"]))
    channels = spec.channels
    for value in (channels.y, channels.x, channels.color):
        values = [value] if isinstance(value, str) else (value or ())
        names.update(str(v) for v in values if isinstance(v, str))
    return names


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _row(
    legacy_id: str | None,
    status: str,
    new_id: str | None,
    score: float | None,
    reason: str,
) -> dict[str, object]:
    return {
        "legacy_id": legacy_id,
        "status": status,
        "new_id": new_id,
        "score": score,
        "reason": reason,
    }


def reconcile(legacy_path: pathlib.Path, new_path: pathlib.Path) -> list[dict[str, object]]:
    """Return one row per old plot (plus new-only rows): ported / missing, with the score."""
    legacy = load_legacy_manifest(legacy_path)
    new_manifest = VizManifest.load(new_path)
    new_by_id = {spec.id: spec for spec in new_manifest.plots}
    rows: list[dict[str, object]] = []
    remaining: list[LegacyPlot] = []
    consumed: set[str] = set()
    for plot in legacy.plots:
        if plot.id in new_by_id:
            consumed.add(plot.id)
            rows.append(_row(plot.id, "ported", plot.id, 1.0, "same id"))
        else:
            remaining.append(plot)
    unused = {key: value for key, value in new_by_id.items() if key not in consumed}
    for plot in remaining:
        family = _TYPE_FAMILY.get(plot.type, plot.type)
        legacy_metrics = _legacy_metrics(plot)
        legacy_tokens = _tokens(plot.title)
        best_id: str | None = None
        best_score = 0.0
        for candidate_id, candidate in unused.items():
            if _TYPE_FAMILY.get(candidate.type, candidate.type) != family:
                continue
            score = 0.75 * _jaccard(legacy_metrics, _new_metrics(candidate)) + 0.25 * (len(legacy_tokens & _tokens(candidate.title)) / max(1, len(legacy_tokens | _tokens(candidate.title))))
            if score > best_score:
                best_id, best_score = candidate_id, score
        if best_id is not None and best_score >= _MATCH_THRESHOLD:
            del unused[best_id]
            rows.append(
                _row(
                    plot.id,
                    "ported",
                    best_id,
                    round(best_score, 3),
                    "matched by metric set and title",
                )
            )
        else:
            rows.append(
                _row(
                    plot.id,
                    "missing",
                    None,
                    round(best_score, 3),
                    "no new plot of the same family draws this metric set; port it or record the intentional drop",
                )
            )
    for extra_id in sorted(unused):
        rows.append(
            _row(
                None,
                "new_only",
                extra_id,
                None,
                "no legacy plot; a deliberate addition",
            )
        )
    return rows


def summarise(rows: list[dict[str, object]]) -> dict[str, int]:
    """Counts by status."""
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=pathlib.Path)
    parser.add_argument("new", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_metric_map()  # fail early when the checked-in map is missing
    rows = reconcile(args.legacy, args.new)
    counts = summarise(rows)
    print(f"{args.legacy.name} vs {args.new}: {counts}")
    for row in rows:
        if row["status"] != "ported":
            print(f"  {row['status']:8s} {row['legacy_id'] or row['new_id']}: {row['reason']}")
    if args.out is not None:
        args.out.write_text(
            yaml.safe_dump(
                {
                    "legacy": str(args.legacy),
                    "new": str(args.new),
                    "counts": counts,
                    "rows": rows,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return 0 if counts.get("missing", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
