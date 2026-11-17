#!/usr/bin/env python3
"""Convert pre-overhaul report manifests into the single report grammar.

The converter is deterministic: it maps the old ``data_key``/``differentiate``/``group_by``/
``filter`` declarations onto the new grammar, translates metric names through
``configs/legacy/legacy_metric_map.yaml`` and the metric registry, and accounts for every
old plot in ``migration_report.yaml`` as converted, dropped-with-reason or ``TODO``.
Bindings it cannot resolve deterministically stay TODO; nothing is guessed.

Usage::

    python scripts/migrate_legacy_manifests.py \\
        configs/benchmark/manifests/social.yaml --dry-run

    python scripts/migrate_legacy_manifests.py \\
        configs/benchmark/manifests/*.yaml --output-dir /tmp/migrated

    python scripts/migrate_legacy_manifests.py --legacy-plots plot_declarations/all_metrics.yaml
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from arena_evaluation.presentation.legacy_manifest import (
    classify_legacy_plot_declaration,
    convert_legacy_file,
    load_metric_map,
    write_legacy_bundle,
)


def _known_metrics() -> frozenset[str]:
    from arena_evaluation.processing.metrics.dictionary import metric_dictionary

    return frozenset(metric_dictionary()["metric"].to_list())


def _metric_tables() -> dict[str, str]:
    from arena_evaluation.processing.metrics.dictionary import metric_dictionary

    tables: dict[str, str] = {}
    for row in metric_dictionary().iter_rows(named=True):
        tables.setdefault(str(row["metric"]), str(row["fact_table"] or ""))
    return tables


def _default_output_dir() -> pathlib.Path:
    from arena_evaluation.storage.config_layout import package_config_roots

    roots = [root for root in package_config_roots() if (root / "reports").is_dir()]
    if not roots:
        raise SystemExit("no writable configs/reports root found; pass --output-dir")
    return roots[0] / "reports"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="*",
        type=pathlib.Path,
        help="old configs/benchmark/manifests/<name>.yaml files",
    )
    parser.add_argument(
        "--metric-map",
        type=pathlib.Path,
        default=None,
        help="legacy metric map (default: configs/legacy/legacy_metric_map.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="directory to write <name>/report.yaml + <name>/migration_report.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing report.yaml in the output bundle",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the plan without writing files"
    )
    parser.add_argument(
        "--legacy-plots",
        type=pathlib.Path,
        action="append",
        default=[],
        metavar="FILE",
        help="classify one legacy plot_declarations/ file (convert/best-effort/archive)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if not args.manifests and not args.legacy_plots:
        print("nothing to do: pass a manifest and/or --legacy-plots", file=sys.stderr)
        return 2
    metric_map = load_metric_map(args.metric_map)
    known = _known_metrics()
    metric_tables = _metric_tables()
    output_dir = args.output_dir or _default_output_dir()
    status = 0
    for path in args.manifests:
        migration = convert_legacy_file(
            path,
            metric_map=metric_map,
            known_metrics=known,
            metric_tables=metric_tables,
        )
        print(
            f"{path}: {len(migration.entries)} plots -> "
            f"{migration.converted} converted, {migration.dropped} dropped, "
            f"{migration.todo} todo; {len(migration.manifest.plots)} plots in the skeleton"
        )
        for entry in migration.entries:
            if entry.status != "converted":
                print(f"  {entry.status:8s} {entry.plot_id}: {entry.reason}")
        if args.dry_run:
            continue
        try:
            report_path, migration_path = write_legacy_bundle(
                migration, output_dir, overwrite=args.overwrite
            )
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"  wrote {report_path} and {migration_path}")
    for declaration in args.legacy_plots:
        classification = classify_legacy_plot_declaration(declaration)
        print(
            f"{classification['file']}: {classification['disposition']} - "
            f"{classification['reason']}"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
