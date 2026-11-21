"""Golden tests for the legacy report-manifest converter (Round-2, R2-9)."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import COMPUTATIONS
from arena_evaluation.presentation.data_sources import declared_data_sources
from arena_evaluation.presentation.legacy_manifest import (
    classify_legacy_plot_declaration,
    convert_legacy_file,
    load_legacy_manifest,
    load_metric_map,
    render_migration_report,
)
from arena_evaluation.presentation.plot_types import RENDERER_REGISTRY
from arena_evaluation.presentation.resolver import DataResolver
from arena_evaluation.processing.metrics.dictionary import metric_dictionary

_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "legacy"
_SOCIAL = _FIXTURES / "social.yaml"


def _known_metrics() -> frozenset[str]:
    return frozenset(metric_dictionary()["metric"].to_list())


def _migrate(path: pathlib.Path):
    return convert_legacy_file(path, metric_map=load_metric_map(), known_metrics=_known_metrics())


def test_every_old_plot_is_accounted_for_and_converted_plots_are_valid() -> None:
    migration = _migrate(_SOCIAL)
    account = {entry.plot_id: entry for entry in migration.entries}

    assert len(account) == 31
    assert {entry.status for entry in migration.entries} <= {
        "converted",
        "dropped",
        "todo",
    }
    for entry in migration.entries:
        assert entry.reason, f"{entry.plot_id} has no recorded disposition reason"
    for plot in migration.manifest.plots:
        assert plot.id in account
        assert plot.type in RENDERER_REGISTRY, plot.id
        if plot.data is not None and plot.data.computation is not None:
            assert plot.data.computation in COMPUTATIONS, plot.id
        if plot.data is not None and plot.data.source is not None:
            assert plot.data.source in declared_data_sources(), plot.id
    assert len(migration.manifest.plots) == migration.converted


def test_no_silent_drop_reports_every_dropped_plot_with_a_reason() -> None:
    migration = _migrate(_SOCIAL)
    report = render_migration_report(migration)

    reported = {row["id"] for row in report["plots"]}
    assert reported == {plot.id for plot in load_legacy_manifest(_SOCIAL).plots}
    dropped = [row for row in report["plots"] if row["status"] == "dropped"]
    assert dropped
    assert all(row["reason"] for row in dropped)
    assert report["counts"] == {
        "total": 31,
        "converted": migration.converted,
        "dropped": migration.dropped,
        "todo": migration.todo,
    }


def test_legacy_manifest_rejects_unknown_fields(tmp_path: pathlib.Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        yaml.safe_dump(
            {
                "name": "broken",
                "plots": [{"id": "p", "type": "bar", "unknown_field": 1}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown_field"):
        load_legacy_manifest(broken)


def test_converted_social_plots_render_a_non_blank_png(tmp_path: pathlib.Path) -> None:
    from star_benchmark import write_benchmark

    from arena_evaluation.presentation.checks import image_is_blank
    from arena_evaluation.presentation.static_renderer import StaticRenderer

    bench = tmp_path / "bench"
    write_benchmark(
        bench,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "success": True,
                "time_to_goal": 10.0,
                "personal_space_intrusion_integral": 1.5,
                "total_time_looking_at_pedestrians": 0.4,
            },
            {
                "episode_id": 1,
                "planner": "teb",
                "success": False,
                "time_to_goal": 12.0,
                "personal_space_intrusion_integral": 2.5,
                "total_time_looking_at_pedestrians": 0.6,
            },
        ],
    )
    migration = _migrate(_SOCIAL)
    by_id = {plot.id: plot for plot in migration.manifest.plots}
    resolver = DataResolver(catalog=BenchmarkCatalog.open(bench), benchmark_dir=bench)
    checked = 0
    for plot_id in ("violin_psii_local", "box_looking_at_peds_local"):
        spec = by_id.get(plot_id)
        if spec is None:
            continue
        frame, diagnostics = resolver.resolve(spec)
        assert diagnostics.status == "PASSED", diagnostics.issues
        png = tmp_path / f"{plot_id}.png"
        StaticRenderer().render(spec, frame, png)
        assert png.exists()
        assert image_is_blank(png) is False
        checked += 1
    assert checked == 2


def test_legacy_plot_declaration_files_are_classified(tmp_path: pathlib.Path) -> None:
    code_schema = tmp_path / "my_plots.yaml"
    code_schema.write_text(yaml.safe_dump({"show_plots": ["a"], "datasets": []}), encoding="utf-8")
    documentation = tmp_path / "sample_schema.yaml"
    documentation.write_text("plot_key: [unclosed\n", encoding="utf-8")

    assert classify_legacy_plot_declaration(code_schema)["disposition"] == "best-effort"
    assert classify_legacy_plot_declaration(documentation)["disposition"] == "archive"
