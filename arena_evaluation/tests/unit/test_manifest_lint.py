"""Lint the shipped declarations: schema-only validity and per-manifest bindability."""

from __future__ import annotations

import pathlib

from star_benchmark import write_benchmark

from arena_evaluation.presentation.lint import (
    lint_declarations,
    lint_manifest_bindings,
)
from arena_evaluation.presentation.viz_manifest import (
    PlotChannels,
    PlotSpec,
    VizManifest,
)


def test_schema_only_lint_is_valid_without_a_benchmark() -> None:
    report = lint_declarations()

    assert report.valid is True
    assert report.schema_valid is True
    assert report.checked_manifests == 8
    assert report.checked_plots == 40
    assert report.bound_plots == 0
    assert report.unbound == ()


def test_bindability_separates_a_bound_plot_from_an_all_null_one(
    tmp_path: pathlib.Path,
) -> None:
    bench_dir = tmp_path / "bench_lint"
    write_benchmark(
        bench_dir,
        [
            {"episode_id": 0, "planner": "dwb", "success": True},
            {"episode_id": 1, "planner": "teb", "success": False},
        ],
    )
    manifest = VizManifest(
        name="lint_binding",
        title="Lint Binding",
        plots=[
            PlotSpec(
                id="bound_table",
                type="table",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="episode_id"),
            ),
            PlotSpec(
                id="all_null_box",
                type="box",
                data={"source": "v_episode_summary"},
                channels=PlotChannels(x="planner", y="time_to_goal"),
            ),
        ],
    )

    findings = lint_manifest_bindings(manifest, bench_dir)

    assert len(findings) == 1
    assert "all_null_box" in findings[0]
    assert "100% NULL" in findings[0]
