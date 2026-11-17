"""Unit tests for the unified B11 configuration and run directory layout."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from arena_evaluation.analytics.queries import QueryCatalog, QueryDeclaration
from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.tree import (
    ConfigPathResolver,
    ContestIdentifier,
    SuiteIdentifier,
)
from arena_evaluation.presentation.manifest_identifier import (
    ManifestIdentifier,
)
from arena_evaluation.presentation.manifest_registry import (
    ManifestNotFoundError,
    resolve_manifest,
)
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.resolver import DataResolver, find_recipe_path
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest
from arena_evaluation.storage.config_layout import (
    CONFIG_ASSETS,
    CONFIG_KINDS,
    all_config_roots,
    bundle_path,
    configs_readme_path,
    find_rendered_plots,
    find_rendered_reports,
    run_analysis_root,
    run_report_output_dir,
)
from star_benchmark import write_benchmark

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _PKG_ROOT / "configs"


def test_package_configs_holds_only_five_kinds_the_flat_assets_and_readme() -> None:
    entries = {p.name for p in _CONFIGS_DIR.iterdir()}
    expected = (
        {spec.folder for spec in CONFIG_KINDS.values()}
        | {asset.filename for asset in CONFIG_ASSETS.values()}
        | {"README.md"}
    )
    assert entries == expected
    readme = configs_readme_path()
    assert readme is not None and readme.is_file()


def test_every_kind_folder_holds_only_bundles_and_parses() -> None:
    for kind_name, spec in CONFIG_KINDS.items():
        kind_dir = _CONFIGS_DIR / spec.folder
        children = sorted(kind_dir.iterdir())
        assert children, f"{kind_dir} is empty"
        for child in children:
            assert child.is_dir(), f"Flat file {child} is not allowed in {kind_dir}"
            entry = child / spec.filename
            assert entry.is_file(), f"Missing bundle entry {entry}"
            if kind_name == "suite":
                Suite.parse(
                    child.name, yaml.safe_load(entry.read_text(encoding="utf-8"))
                )
            elif kind_name == "contest":
                Contest.parse(
                    child.name, yaml.safe_load(entry.read_text(encoding="utf-8"))
                )
            elif kind_name == "report":
                VizManifest.load(entry)
            elif kind_name == "plot":
                PlotSpec.model_validate(
                    yaml.safe_load(entry.read_text(encoding="utf-8"))
                )
            elif kind_name == "query":
                payload = yaml.safe_load(entry.read_text(encoding="utf-8"))
                QueryDeclaration(
                    name=str(payload.get("name") or child.name),
                    description=str(payload.get("description") or ""),
                    sql=str(payload["sql"]),
                    parameters=dict(payload.get("parameters") or {}),
                    title=str(payload.get("title") or child.name),
                )


def test_flat_config_files_are_not_found(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "configs"
    for spec in CONFIG_KINDS.values():
        d = cfg / spec.folder
        d.mkdir(parents=True)
        (d / f"flat_{spec.name}.yaml").write_text("name: flat\n", encoding="utf-8")

    saved_suite = SuiteIdentifier._resolvers
    saved_contest = ContestIdentifier._resolvers
    saved_manifest = ManifestIdentifier._resolvers
    SuiteIdentifier._resolvers = [ConfigPathResolver(SuiteIdentifier, cfg)]
    ContestIdentifier._resolvers = [ConfigPathResolver(ContestIdentifier, cfg)]
    ManifestIdentifier._resolvers = [ConfigPathResolver(ManifestIdentifier, cfg)]
    try:
        with pytest.raises(FileNotFoundError):
            SuiteIdentifier(name="flat_suite").resolve_source_sync()
        with pytest.raises(FileNotFoundError):
            ContestIdentifier(name="flat_contest").resolve_source_sync()
        with pytest.raises(ManifestNotFoundError):
            resolve_manifest("flat_report")
        assert list(SuiteIdentifier.listall()) == []
        assert list(ContestIdentifier.listall()) == []
        assert list(ManifestIdentifier.listall()) == []
    finally:
        SuiteIdentifier._resolvers = saved_suite
        ContestIdentifier._resolvers = saved_contest
        ManifestIdentifier._resolvers = saved_manifest

    assert find_recipe_path("flat_plot", extra_dirs=[cfg]) is None
    qcat = QueryCatalog([cfg])
    with pytest.raises(KeyError):
        qcat.get_query("flat_query")


def test_run_local_bundles_override_package_bundles(tmp_path: pathlib.Path) -> None:
    analysis = run_analysis_root(tmp_path)
    report_file = bundle_path(analysis, "report", "standard")
    report_file.parent.mkdir(parents=True)
    report_file.write_text(
        "manifest_version: '1.0'\nname: standard\ntitle: Run Standard\nplots: []\n",
        encoding="utf-8",
    )
    assert resolve_manifest("standard", benchmark_dir=tmp_path).title == "Run Standard"

    plot_file = bundle_path(analysis, "plot", "scorecard_table")
    plot_file.parent.mkdir(parents=True)
    plot_file.write_text(
        "id: scorecard_table\ntype: table\ntitle: Run Scorecard\ndata:\n  source: v_planner_scorecard\n",
        encoding="utf-8",
    )
    resolved_plot = DataResolver(benchmark_dir=tmp_path).resolve_spec(
        PlotSpec(id="p", uses="scorecard_table")
    )
    assert resolved_plot.title == "Run Scorecard"

    query_file = bundle_path(analysis, "query", "robot_speed_distribution")
    query_file.parent.mkdir(parents=True)
    query_file.write_text(
        "name: robot_speed_distribution\ndescription: override\nsql: SELECT 7 AS v\n",
        encoding="utf-8",
    )
    assert (
        "SELECT 7 AS v"
        in QueryCatalog(run_dir=tmp_path).get_query("robot_speed_distribution").sql
    )


def test_report_build_writes_into_reports_subdir_only(tmp_path: pathlib.Path) -> None:
    bench = tmp_path / "run"
    write_benchmark(
        bench,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "success": 1.0,
                "time_to_goal": 10.0,
                "path_length_m": 8.0,
            },
            {
                "episode_id": 1,
                "planner": "teb",
                "success": 0.0,
                "time_to_goal": 20.0,
                "path_length_m": 12.0,
            },
        ],
    )
    manifest = VizManifest.model_validate(
        {
            "manifest_version": "1.0",
            "name": "custom_rep",
            "title": "Custom Report",
            "data_source": "v_episode_summary",
            "summary": [{"metric": "time_to_goal", "label": "Time (s)"}],
            "summary_group_by": ["planner"],
            "plots": [
                {
                    "id": "time_bar",
                    "type": "bar",
                    "title": "Time",
                    "data": {
                        "computation": "group_summary",
                        "params": {
                            "metrics": ["time_to_goal"],
                            "group_cols": ["planner"],
                        },
                    },
                    "channels": {"x": "planner", "y": "mean"},
                }
            ],
        }
    )
    builder = ReportBuilder(bench, manifest=manifest)
    out_html = builder.build()
    expected_dir = run_report_output_dir(bench, "custom_rep")
    assert out_html == expected_dir / "report.html"
    assert out_html.is_file()
    report_text = out_html.read_text(encoding="utf-8")
    assert "plotly.js v" in report_text
    assert not (expected_dir / "plotly.min.js").exists()
    assert (expected_dir / "plots" / "time_bar.png").is_file()
    assert not (bench / "report.html").exists()
    assert not (bench / "report_manifest.yaml").exists()
    assert not (bench / "viz_manifest.yaml").exists()
    assert not (bench / "plots").exists()
    assert find_rendered_reports(bench) == [expected_dir / "report.html"]
    assert [p.name for p in find_rendered_plots(bench)] == ["time_bar.png"]
    assert not (_PKG_ROOT / "arena_evaluation" / "plot_declarations").exists()

    with pytest.raises(ValueError, match="name"):
        ReportBuilder(bench, manifest="{plots: []}")


def test_all_config_roots_orders_imported_before_share_and_source(
    tmp_path: pathlib.Path,
) -> None:
    from arena_evaluation.benchmark.dashboard import run_status
    from arena_evaluation.benchmark.state import Manifest
    from arena_evaluation.storage.config_layout import share_dir

    imp = tmp_path / "imported_root"
    imp.mkdir()
    roots = list(all_config_roots(imp))
    assert roots[0] == imp
    assert _CONFIGS_DIR in roots
    sd = share_dir()
    if sd is not None and (sd / "configs").resolve() != _CONFIGS_DIR.resolve():
        assert roots.index((sd / "configs").resolve()) < roots.index(
            _CONFIGS_DIR.resolve()
        )

    # Dashboard has_report uses find_rendered_reports under reports/<name>/report.html
    run_dir = tmp_path / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.yaml").write_text(
        Manifest(
            run_id="run_1",
            suite_name="basic",
            contest_name="basic",
            simulator="dummy",
            env_n=1,
            headless=True,
            created_at="2026-01-01T00:00:00Z",
            arena_git_sha="abc",
            arena_git_dirty=False,
            cli_args=[],
            config_hash="h",
            scale_episodes=1,
            suite={},
            contest={},
            steps=[],
        ).to_yaml(),
        encoding="utf-8",
    )
    assert run_status(run_dir, light=True)["has_report"] is False
    rep_dir = run_report_output_dir(run_dir, "standard")
    rep_dir.mkdir(parents=True)
    (rep_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    assert run_status(run_dir, light=True)["has_report"] is True
