"""Contract tests for all bundled report manifests and plot declaration recipes."""

from __future__ import annotations

import datetime as dt
import pathlib
import random

import polars as pl
import pytest

from arena_evaluation.analytics.catalog import BenchmarkCatalog
from arena_evaluation.analytics.computations import COMPUTATIONS
from arena_evaluation.analytics.queries import QueryCatalog
from arena_evaluation.presentation.data_sources import declared_data_sources
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.presentation.resolver import DataResolver, load_recipe
from arena_evaluation.presentation.viz_manifest import PlotSpec, VizManifest
from arena_evaluation.processing.metrics.dictionary import metric_dictionary
from arena_evaluation.storage.registry import FACT_TABLES, dimensions
from arena_evaluation.storage.registry.dimensions import DimWorldGeometry
from arena_evaluation.storage.registry.series import SERIES_TABLES
from arena_evaluation.storage.registry.telemetry import (
    PedestrianArenaPeds,
    RobotTracks,
)

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFESTS_DIR = _PKG_ROOT / "configs" / "reports"
_TEST_WORLD_YAML = """format: arena-world/2
levels:
  - id: L1
    zones:
      - name: floor
        polygon:
          - [0.0, 0.0]
          - [4.0, 0.0]
          - [4.0, 4.0]
          - [0.0, 4.0]
    walls:
      - path: [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
      - path: [[2.0, 0.0], [2.0, 4.0]]
    doors:
      - name: d1
        kind: sliding
        span: [[2.0, 1.5], [2.0, 2.5]]
        width: 0.2
"""
_DECLARATIONS_DIR = _PKG_ROOT / "configs" / "plots"


def _manifest_paths() -> list[pathlib.Path]:
    paths = sorted(_MANIFESTS_DIR.glob("*/report.yaml"))
    assert paths, f"No manifests found in {_MANIFESTS_DIR}"
    return paths


def _declaration_paths() -> list[pathlib.Path]:
    paths = sorted(_DECLARATIONS_DIR.glob("*/plot.yaml"))
    assert paths, f"No plot declarations found in {_DECLARATIONS_DIR}"
    return paths


def _assert_valid_plot_binding(spec: PlotSpec) -> None:
    if spec.uses:
        spec = DataResolver().resolve_spec(spec)
    assert spec.type not in ("acoustic_field_animation", "dendrogram")
    if spec.data.source is not None:
        assert spec.data.source in declared_data_sources(), (
            f"{spec.id}: unknown source {spec.data.source!r}"
        )
    if spec.data.computation is not None:
        assert spec.data.computation in COMPUTATIONS, (
            f"{spec.id}: unknown computation {spec.data.computation!r}"
        )
    if spec.data.stored_query is not None:
        QueryCatalog().get_query(spec.data.stored_query)


@pytest.mark.parametrize("manifest_path", _manifest_paths(), ids=lambda p: p.name)
def test_bundled_manifest_conforms_to_single_grammar_and_star_schema(
    manifest_path: pathlib.Path,
) -> None:
    manifest = VizManifest.load(manifest_path)
    assert manifest.plots, f"{manifest_path.name} has no plots"
    for spec in manifest.plots:
        _assert_valid_plot_binding(spec)


@pytest.mark.parametrize("recipe_path", _declaration_paths(), ids=lambda p: p.name)
def test_bundled_plot_declaration_conforms_to_single_grammar(
    recipe_path: pathlib.Path,
) -> None:
    spec = PlotSpec.model_validate(load_recipe(recipe_path.parent.name))
    _assert_valid_plot_binding(spec)


def _dummy_val(name: str, dtype: pl.DataType, rng: random.Random, idx: int) -> object:
    if dtype == pl.Boolean:
        return idx % 4 != 0
    if dtype.is_integer():
        return idx % 5 + 1
    if dtype.is_float():
        return round(0.5 + (idx % 7) * 0.4 + rng.uniform(0.01, 0.2), 3)
    if dtype in (pl.Utf8, pl.String):
        return f"{name}_v"
    if isinstance(dtype, pl.Datetime):
        return dt.datetime(2026, 1, 1)
    if isinstance(dtype, pl.Duration):
        return dt.timedelta(seconds=1)
    return None


@pytest.fixture(scope="module")
def full_registry_bench(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    from star_benchmark import write_benchmark

    bench = tmp_path_factory.mktemp("full_registry_bench")
    rng = random.Random(17)
    schemas = [
        dimensions.DimEpisodes.polars_schema(),
        dimensions.DimRobots.polars_schema(),
    ]
    schemas += [table.polars_schema() for table in FACT_TABLES.values()]
    planners = ("dwb", "teb", "mppi")
    robots = ("jackal", "burger")
    rows: list[dict[str, object]] = []
    robot_tracks: list[dict[str, object]] = []

    for episode_id in range(12):
        row: dict[str, object] = {}
        for schema in schemas:
            for name, dtype in schema.items():
                row.setdefault(name, _dummy_val(name, dtype, rng, episode_id))
        row.update(
            {
                "run_id": "run",
                "episode_id": episode_id,
                "robot_id": "robot_0",
                "env_id": "env_0",
                "planner": planners[episode_id % 3],
                "robot_model": robots[episode_id % 2],
                "world": "map_a",
                "is_reference": False,
                # An evaluated episode: the required invalid_reason is the empty string, not a dummy
                # token, so the synthetic benchmark's episodes are valid for ranking (L3-002/L3-030).
                "invalid_reason": "",
                "success": episode_id % 4 != 0,
                "invalid_reason": "",
            }
        )
        rows.append(row)
        for step in range(4):
            robot_tracks.append(
                {
                    "episode_id": episode_id,
                    "t_ns": step * 1_000_000_000,
                    "x": float(step),
                    "y": float(episode_id % 3) * 0.5,
                    "yaw": 0.0,
                    "vx": 0.8,
                    "vy": 0.0,
                    "wz": 0.0,
                }
            )

    write_benchmark(
        bench,
        rows,
        robot_tracks=robot_tracks,
        contacts=[
            {
                "episode_id": 0,
                "contact_id": "c0",
                # Only a pedestrian contact carries a closing share: a wall cannot close in (L3-013).
                "collider_kind": "pedestrian",
                "v_close_robot_mps": 0.6,
                "v_close_target_mps": 0.4,
                "relative_impact_speed_mps": 1.0,
                "freeze_clearance_m": 0.5,
            }
        ],
    )

    # Write world geometry, series tables, and pedestrian telemetry from registry schemas
    geom_df = pl.DataFrame(
        [
            {
                "world": "map_a",
                "element_kind": "wall",
                "element_name": "w0",
                "part_idx": 0,
                "vertex_idx": 0,
                "x": -1.0,
                "y": -1.0,
                "null_reason": None,
            },
            {
                "world": "map_a",
                "element_kind": "wall",
                "element_name": "w0",
                "part_idx": 0,
                "vertex_idx": 1,
                "x": 5.0,
                "y": -1.0,
                "null_reason": None,
            },
        ],
        schema=DimWorldGeometry.polars_schema(),
    )
    geom_df.write_parquet(bench / f"{DimWorldGeometry.table_name}.parquet")
    metric_dictionary().write_parquet(bench / "dim_metrics.parquet")

    # A-028: the shipped acoustic-field plot needs the world's walls/doors in the run.
    world_dir = bench / "worlds" / "map_a"
    world_dir.mkdir(parents=True, exist_ok=True)
    (world_dir / "world.yaml").write_text(_TEST_WORLD_YAML, encoding="utf-8")

    # One pedestrian per episode makes crowd_condition real (blind for the first half,
    # visible for the second), so the E8/E9 crowd-condition declarations bind and render.
    ped_schema = dimensions.DimPedestrians.polars_schema()
    ped_rows: list[dict[str, object]] = []
    for episode_id in range(12):
        ped_row = {
            name: _dummy_val(name, dtype, rng, episode_id)
            for name, dtype in ped_schema.items()
        }
        ped_row.update(
            {
                "run_id": "run",
                "episode_id": episode_id,
                "env_id": "env_0",
                "ped_id": f"ped_{episode_id}",
                "robot_awareness": 0.0 if episode_id < 6 else 1.0,
            }
        )
        ped_rows.append(ped_row)
    pl.DataFrame(ped_rows, schema=ped_schema).write_parquet(
        bench / f"{dimensions.DimPedestrians.table_name}.parquet"
    )

    for episode_id in range(12):
        ep_dir = bench / "episodes" / f"episode_{episode_id:03d}"
        topics_dir = ep_dir / "topics"
        topics_dir.mkdir(parents=True, exist_ok=True)
        for series_cls in SERIES_TABLES.values():
            s_schema = series_cls.polars_schema()
            s_rows = []
            for step in range(4):
                sr = {
                    name: _dummy_val(name, dtype, rng, episode_id + step)
                    for name, dtype in s_schema.items()
                }
                sr["run_id"] = "run"
                sr["episode_id"] = episode_id
                sr["robot_id"] = "robot_0"
                if "is_reference" in s_schema:
                    sr["is_reference"] = False
                if "phase_kind" in s_schema:
                    sr["phase_kind"] = (
                        "idle",
                        "linear_cruise",
                        "pivot",
                        "arc_cruise",
                        "holonomic_cruise",
                        "ramp_up",
                        "ramp_down",
                        "ramp_apex",
                    )[(episode_id * 4 + step) % 8]
                if "t_s" in s_schema:
                    sr["t_s"] = float(step)
                s_rows.append(sr)
            pl.DataFrame(s_rows, schema=s_schema).write_parquet(
                ep_dir
                / "metrics"
                / f"series_{series_cls.table_name.removeprefix('series_')}.parquet"
            )

        ped_schema = PedestrianArenaPeds.polars_schema()
        ped_rows = [
            {
                **{
                    name: _dummy_val(name, dtype, rng, step)
                    for name, dtype in ped_schema.items()
                },
                "t_ns": step * 1_000_000_000,
                "env_id": "env_0",
                "x": float(step) + 0.5,
                "y": -0.5,
            }
            for step in range(4)
        ]
        pl.DataFrame(ped_rows, schema=ped_schema).write_parquet(
            topics_dir / f"{PedestrianArenaPeds.table_name}.parquet"
        )
        rt_schema = RobotTracks.polars_schema()
        rt_rows = [
            {
                **{
                    name: _dummy_val(name, dtype, rng, step)
                    for name, dtype in rt_schema.items()
                },
                "t_ns": step * 1_000_000_000,
                # A telemetry row carries its recorded keys: the trajectory query joins the episode on
                # (episode_id, robot_id), so a dummy robot id would fan the sample out onto no episode.
                "env_id": "env_0",
                "robot_id": "robot_0",
                "x": float(step),
                "y": float(episode_id % 3) * 0.5,
            }
            for step in range(4)
        ]
        pl.DataFrame(rt_rows, schema=rt_schema).write_parquet(
            topics_dir / f"{RobotTracks.table_name}.parquet"
        )

    return bench


@pytest.mark.parametrize(
    "manifest_path", _manifest_paths(), ids=lambda p: p.parent.name
)
def test_bundled_manifest_builds_with_zero_failed_checks(
    manifest_path: pathlib.Path,
    full_registry_bench: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    manifest = VizManifest.load(manifest_path)
    if manifest.name is None:
        manifest = manifest.model_copy(update={"name": manifest_path.parent.name})
    out_dir = tmp_path / manifest_path.parent.name
    builder = ReportBuilder(full_registry_bench, output_dir=out_dir, manifest=manifest)
    report_path = builder.build()
    assert report_path is not None and report_path.exists()
    assert builder.build_checks.failed_count == 0, (
        f"{manifest_path.parent.name} had failed checks: {builder.build_checks.failures}"
    )


def test_shipped_reports_preserve_metrics_inter_planners_filters_and_trajectories() -> (
    None
):
    import yaml

    configs_dir = pathlib.Path(__file__).resolve().parents[2] / "configs"
    reports = {
        p.parent.name: yaml.safe_load(p.read_text(encoding="utf-8"))
        for p in _manifest_paths()
    }
    # No legacy options.columns anywhere in manifests or plot recipes
    for rname, rdata in reports.items():
        for p in rdata["plots"]:
            opts = p.get("options") or {}
            assert "columns" not in opts, (
                f"{rname}:{p['id']} still uses options.columns"
            )
    for rpath in _declaration_paths():
        rdata = yaml.safe_load(rpath.read_text(encoding="utf-8"))
        assert "columns" not in (rdata.get("options") or {}), (
            f"{rpath} uses options.columns"
        )

    # Pivot heatmaps bind their real metrics, not success
    ev_plots = {p["id"]: p for p in reports["everything"]["plots"]}
    assert ev_plots["heatmap_spl_pivot"]["data"]["params"]["metrics"] == [
        "spl_geodesic"
    ]
    assert ev_plots["heatmap_ecot_pivot"]["data"]["params"]["metrics"] == ["e_cot"]
    assert ev_plots["heatmap_psii_pivot"]["data"]["params"]["metrics"] == [
        "personal_space_intrusion_integral"
    ]
    sf_plots = {p["id"]: p for p in reports["safety"]["plots"]}
    assert sf_plots["heatmap_spl_pivot"]["data"]["params"]["metrics"] == [
        "spl_geodesic"
    ]

    # All 6 inter plots group by inter_planner
    for pid in ("bar_success_inter", "bar_spl_inter", "bar_passing_compliance_inter"):
        assert ev_plots[pid]["channels"]["x"] == "inter_planner"
        assert ev_plots[pid]["data"]["params"]["group_cols"] == ["inter_planner"]
    for pid in ("bar_success_inter", "bar_spl_inter"):
        assert sf_plots[pid]["channels"]["x"] == "inter_planner"
        assert sf_plots[pid]["data"]["params"]["group_cols"] == ["inter_planner"]
    soc_plots = {p["id"]: p for p in reports["social"]["plots"]}
    assert soc_plots["violin_deflection_inter"]["channels"]["x"] == "inter_planner"

    # Characterization preserves 21 phase_kind filters and 18 smooth_window params
    ch_plots = reports["characterization"]["plots"]
    filtered = [
        p
        for p in ch_plots
        if "phase_kind" in str((p["data"].get("params") or {}).get("filter", ""))
        or "phase_kind" in str(p["data"].get("where", ""))
    ]
    smoothed = [
        p for p in ch_plots if (p["data"].get("params") or {}).get("smooth_window")
    ]
    assert len(filtered) == 21
    assert len(smoothed) == 18

    # contact_closing_share recipe is a bar on contact_attribution_summary
    ccs = yaml.safe_load(
        (configs_dir / "plots" / "contact_closing_share" / "plot.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert ccs["type"] == "bar"
    assert ccs["data"]["stored_query"] == "contact_attribution_summary"
    assert ccs["channels"]["facet_col"] == "crowd_condition"

    # standard, social, everything, safety bind both non-reference trajectory queries
    for rname in ("standard", "social", "everything", "safety"):
        by_id = {p["id"]: p for p in reports[rname]["plots"]}
        assert (
            by_id["trajectory_plot"]["data"]["stored_query"] == "robot_tracks_non_ref"
        )
        assert (
            by_id["pedestrian_trajectory_plot"]["data"]["stored_query"]
            == "pedestrian_arena_peds_non_ref"
        )

    # All group_summary bars declare err_plus/err_minus; all distribution_bins histograms declare bin_low/bin_high
    for rname, rdata in reports.items():
        for p in rdata["plots"]:
            comp = (p.get("data") or {}).get("computation")
            opts = p.get("options") or {}
            if p.get("type") == "bar" and comp == "group_summary":
                assert opts.get("err_plus") == "err_plus", (
                    f"{rname}:{p['id']} missing options.err_plus"
                )
                assert opts.get("err_minus") == "err_minus", (
                    f"{rname}:{p['id']} missing options.err_minus"
                )
            if p.get("type") == "histogram" and comp == "distribution_bins":
                assert opts.get("bin_low") == "bin_low", (
                    f"{rname}:{p['id']} missing options.bin_low"
                )
                assert opts.get("bin_high") == "bin_high", (
                    f"{rname}:{p['id']} missing options.bin_high"
                )

    # Converted tables preserve full source metric lists, labels, and formats
    expected_table_metrics = {
        ("failure_modes", "table_executive_failure_triage"): 11,
        ("failure_modes", "table_theta_star_verification"): 9,
        ("failure_modes", "table_mar_verification"): 3,
        ("failure_modes", "table_proxemics_verification"): 7,
        ("failure_modes", "table_etiquette_verification"): 5,
        ("failure_modes", "table_kinematics_verification"): 9,
        ("failure_modes", "table_energy_verification"): 10,
        ("everything", "master_summary_table"): 10,
        ("safety", "safety_summary_table"): 8,
        ("ecological", "eco_summary_table"): 5,
        ("standard", "overview_table"): 7,
    }
    for (rname, pid), count in expected_table_metrics.items():
        by_id = {p["id"]: p for p in reports[rname]["plots"]}
        tbl = by_id[pid]
        assert len(tbl["data"]["params"]["metrics"]) == count, (
            f"{rname}:{pid} metric count"
        )
        assert len(tbl["options"]["labels"]) == count, f"{rname}:{pid} labels count"
        assert len(tbl["options"]["format"]) == count, f"{rname}:{pid} format count"


@pytest.mark.parametrize(
    "recipe_path", _declaration_paths(), ids=lambda p: p.parent.name
)
def test_bundled_recipe_executes_and_contains_declared_columns(
    recipe_path: pathlib.Path,
    full_registry_bench: pathlib.Path,
) -> None:
    catalog = BenchmarkCatalog.open(full_registry_bench)
    resolver = DataResolver(catalog=catalog, benchmark_dir=full_registry_bench)
    spec = resolver.resolve_spec(
        PlotSpec.model_validate(load_recipe(recipe_path.parent.name))
    )
    df, _ = resolver.resolve(spec)
    assert not df.is_empty(), f"{recipe_path.name} resolved to empty DataFrame"
    for ch_val in (
        spec.channels.x,
        spec.channels.y,
        spec.channels.color,
        spec.channels.facet_col,
        spec.channels.facet_row,
        spec.channels.hierarchy,
    ):
        if ch_val is None:
            continue
        cols = [ch_val] if isinstance(ch_val, str) else list(ch_val)
        for col in cols:
            assert col in df.columns, (
                f"{recipe_path.name} missing column {col!r} in {df.columns}"
            )
