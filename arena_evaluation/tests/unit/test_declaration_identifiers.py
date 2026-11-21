"""Plots, queries and reports resolve local first and then from the benchmark buckets."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from arena_simulation_setup import tree as sim_tree

from arena_evaluation.analytics.query_identifier import QueryIdentifier
from arena_evaluation.analytics.queries import QueryCatalog, read_query_declaration
from arena_evaluation.presentation import resolver
from arena_evaluation.presentation.plot_identifier import PlotIdentifier
from arena_evaluation.storage import config_identifiers
from arena_evaluation.storage.config_identifiers import (
    ConfigPathResolver,
    bundle_filename,
)

_PKG_CONFIGS: Path = Path(config_identifiers.__file__).parents[2] / "configs"
_QUERY_YAML = "title: {title}\ndescription: fixture\nsql: SELECT 1 AS x\n"


def _write_query(root: Path, name: str, title: str) -> None:
    bundle = root / "queries" / name
    bundle.mkdir(parents=True)
    (bundle / "query.yaml").write_text(_QUERY_YAML.format(title=title))


@pytest.mark.parametrize(
    ("folder", "filename"),
    [
        ("suites", "suite.yaml"),
        ("contests", "contest.yaml"),
        ("reports", "report.yaml"),
        ("plots", "plot.yaml"),
        ("queries", "query.yaml"),
    ],
)
def test_bundle_filename_comes_from_the_folder_of_its_config_kind(
    folder: str, filename: str
) -> None:
    assert bundle_filename(folder) == filename


def test_config_path_resolver_finds_query_bundles_in_the_queries_folder(
    tmp_path: Path,
) -> None:
    _write_query(tmp_path, "telemetry", "Telemetry")
    local = ConfigPathResolver(QueryIdentifier, tmp_path)

    found = asyncio.run(local.resolve(QueryIdentifier(name="telemetry")))

    assert found == tmp_path / "queries" / "telemetry" / "query.yaml"
    assert [identifier.name for identifier in local.listall()] == ["telemetry"]


def test_bucket_cache_bundle_resolves_while_its_ttl_marker_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sim_tree, "ARENA_ASSETS_DIR", tmp_path)
    bundle = tmp_path / "arena-internal" / "plots" / "fleet_speed"
    bundle.mkdir(parents=True)
    (bundle / "plot.yaml").write_text("id: fleet_speed\ntype: table\n")
    (bundle / ".ttl").write_text(str(int(time.time())))
    bucket = sim_tree.NetResolver(
        PlotIdentifier,
        provider="arena-internal",
        ttl=3600,
        formats=(),
        annotated=False,
        list_prefix="plots",
    )

    located = asyncio.run(bucket.resolve(PlotIdentifier(name="fleet_speed")))

    assert located == bundle
    assert PlotIdentifier(name="fleet_speed").load(located) == {
        "id": "fleet_speed",
        "type": "table",
    }


def test_local_query_shadows_the_bucket_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_query(tmp_path / "cache" / "arena-internal", "dup", "Bucket copy")
    local = tmp_path / "local"
    _write_query(local, "dup", "Local copy")
    monkeypatch.setattr(config_identifiers, "ARENA_ASSETS_DIR", tmp_path / "cache")
    monkeypatch.setattr(config_identifiers, "BENCHMARK_PROVIDERS", ["arena-internal"])

    catalog = QueryCatalog(search_dirs=[local])

    assert catalog.get_query("dup").title == "Local copy"


def test_unknown_query_is_fetched_from_the_bucket_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"title": "Remote", "description": "fixture", "sql": "SELECT 1 AS x"}
    monkeypatch.setattr(QueryIdentifier, "resolve_sync", lambda self, **kwargs: payload)
    catalog = QueryCatalog(search_dirs=[])

    assert catalog.has_query("remote_slice")
    assert catalog.get_query("remote_slice").title == "Remote"


def test_query_name_defaults_to_its_bundle_folder(tmp_path: Path) -> None:
    bundle = tmp_path / "queries" / "named_by_folder"
    bundle.mkdir(parents=True)
    (bundle / "query.yaml").write_text("sql: SELECT 1 AS x\n")

    assert read_query_declaration(bundle / "query.yaml").name == "named_by_folder"


def test_plot_lookup_falls_back_to_the_bucket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched = tmp_path / "plots" / "fleet_speed" / "plot.yaml"
    fetched.parent.mkdir(parents=True)
    fetched.write_text("id: fleet_speed\n")
    verdict = type("Verdict", (), {"path": fetched})()
    monkeypatch.setattr(PlotIdentifier, "resolve_source_sync", lambda self: verdict)

    assert resolver.find_recipe_path("fleet_speed") == fetched


def test_plot_lookup_returns_none_when_no_root_or_bucket_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(self: PlotIdentifier) -> None:
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(PlotIdentifier, "resolve_source_sync", missing)

    assert resolver.find_recipe_path("no_such_plot") is None


def test_repository_ships_the_restored_declarations_per_kind() -> None:
    assert sorted(path.name for path in (_PKG_CONFIGS / "plots").iterdir()) == [
        "acoustic_field",
        "ade_box",
        "angle_over_length_violin",
        "contact_closing_share",
        "encounter_hierarchy_sunburst",
        "encounter_hierarchy_treemap",
        "energy_total_bar",
        "energy_total_interval",
        "energy_vs_time_tradeoff",
        "frechet_distance_box",
        "jerk_mean_violin",
        "mean_velocity_violin",
        "mhd_box",
        "multi_planner_radar",
        "paired_episode_battery_soc_grid",
        "paired_episode_power_grid",
        "path_efficiency_violin",
        "pedestrian_contacts_bar",
        "pedestrian_trajectories_by_world",
        "psii_bar",
        "psii_violin",
        "reference_relative_throughput_bar",
        "reference_relative_throughput_interval",
        "reference_relative_throughput_violin",
        "robot_trajectories_by_world",
        "roughness_mean_violin",
        "scorecard_table",
        "scot_bar",
        "scot_interval",
        "spl_geodesic_bar",
        "spl_geodesic_interval",
        "success_rate_bar",
        "success_rate_interval",
        "success_without_contact_interval",
        "success_without_contact_panel",
        "time_to_goal_bar",
        "time_to_goal_interval",
        "tti_min_bar",
        "tti_min_interval",
        "velocity_timeseries",
    ]
    assert sorted(path.name for path in (_PKG_CONFIGS / "queries").iterdir()) == [
        "blind_vs_aware_standoff_delta",
        "contact_attribution_summary",
        "encounter_approach_breakdown",
        "encounter_hierarchy_summary",
        "energy_timeseries_slice",
        "local_planner_world_scorecard",
        "near_miss_encounter_spikes",
        "paired_episode_energy_traces",
        "pedestrian_arena_peds_non_ref",
        "pedestrian_deflection_by_inter_planner",
        "pedestrian_deflection_by_local_planner",
        "pedestrian_exposure_by_planner",
        "planner_scorecard_overview",
        "reference_relative_throughput_summary",
        "robot_speed_distribution",
        "robot_tracks_non_ref",
        "safety_social_summary",
        "telemetry_slice",
    ]
    assert sorted(path.name for path in (_PKG_CONFIGS / "reports").iterdir()) == [
        "characterization",
        "ecological",
        "everything",
        "failure_modes",
        "reference",
        "safety",
        "social",
        "standard",
    ]
