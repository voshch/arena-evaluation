"""Contract tests for PR5/H2: self-describing lake, snapshot_id, merge runs, and declaration lint."""

from __future__ import annotations

import os

import json
import pathlib

import polars as pl
import pytest

pytest.importorskip("arena_mcp")

from mcp_fixtures import build_runs, run_session, server_env

from arena_evaluation.mcp import analysis, query, schema, slices


def test_h2_evaluation_schema_lists_tables_and_columns_with_units(
    tmp_path: pathlib.Path,
) -> None:
    """evaluation_schema lists every catalog table and enriches metric columns with unit and meaning."""
    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    overview = schema.evaluation_schema(run_id=run_dir.name)
    assert overview["snapshot_id"].startswith("snap-")
    tables = {entry["name"]: entry for entry in overview["tables"]}
    assert "dim_episodes" in tables
    assert "fact_robots" in tables
    assert "v_episode_summary" in tables
    assert tables["dim_episodes"]["kind"] == "dimension"
    assert tables["v_episode_summary"]["kind"] == "view"
    assert tables["v_episode_summary"]["row_count"] > 0

    detail = schema.evaluation_schema(run_id=run_dir.name, table="fact_robots")
    assert detail["snapshot_id"] == overview["snapshot_id"]
    assert detail["example_query"].startswith("SELECT")
    cols = {col["name"]: col for col in detail["columns"]}
    assert cols["energy_total_wh"]["unit"] == "Wh"
    assert cols["energy_total_wh"]["meaning"]
    assert cols["l_aeq_t"]["unit"] == "dBA"
    assert cols["l_aeq_t"]["meaning"]


def test_h2_snapshot_id_propagates_across_tools_and_changes_on_reprocess(
    tmp_path: pathlib.Path,
) -> None:
    """snapshot_id is returned by schema, query, analysis, and slice tools and changes on data edit."""
    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    sch = schema.evaluation_schema(run_id=run_dir.name)
    q_res = query.evaluation_query(
        run_id=run_dir.name,
        query="SELECT planner, success FROM v_episode_summary",
    )
    s_res = analysis.evaluation_summarize(run_id=run_dir.name)
    sl_res = slices.evaluation_slice(run_id=run_dir.name, episode_id=0)

    snap = sch["snapshot_id"]
    assert q_res["snapshot_id"] == snap
    assert s_res["snapshot_id"] == snap
    assert sl_res.snapshot_id == snap

    # Modify an episode parquet file to simulate reprocessing
    ep0_robots = run_dir / "episodes" / "episode_000" / "metrics" / "fact_robots.parquet"
    df = pl.read_parquet(ep0_robots)
    df.with_columns(pl.lit(999.0).alias("energy_total_wh")).write_parquet(ep0_robots)

    sch_after = schema.evaluation_schema(run_id=run_dir.name)
    assert sch_after["snapshot_id"] != snap


def test_h2_describe_metric_returns_scopes_direction_and_rollup(
    tmp_path: pathlib.Path,
) -> None:
    """evaluation_describe_metric returns scopes, direction, unit, and rollup_rule."""
    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    desc = analysis.evaluation_describe_metric("time_to_goal", run_id=run_dir.name)
    assert desc["direction"] == "lower"
    assert desc["unit"] == "s"
    assert "robots" in desc["scopes"]
    assert desc["rollup_rule"]


def test_h2_merge_runs_references_without_copying_episodes(
    tmp_path: pathlib.Path,
) -> None:
    """evaluation_merge_runs creates a merged benchmark without copying episodes and exposes per-source labels."""
    from arena_evaluation.benchmark.state import RunDir
    from mcp_fixtures import manifest, metric_rows, write_star_schema

    data_dir = tmp_path / "data"
    benchmarks = data_dir / "benchmarks"
    r1 = RunDir.create(benchmarks, "run_isaac", manifest("run_isaac"))
    r1.progress.close()
    write_star_schema(r1.path, metric_rows())

    r2 = RunDir.create(benchmarks, "run_gazebo", manifest("run_gazebo"))
    r2.progress.close()
    write_star_schema(r2.path, metric_rows())

    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    merged = schema.evaluation_merge_runs(
        run_ids=["run_isaac", "run_gazebo"],
        name="merged_sim_cmp",
        labels={
            "run_isaac": {"simulator": "isaac"},
            "run_gazebo": {"simulator": "gazebo"},
        },
    )
    merged_dir = pathlib.Path(merged["path"])
    assert (merged_dir / "merge.yaml").is_file()
    assert not (merged_dir / "episodes").exists()

    q_res = query.evaluation_query(
        run_id="merged_sim_cmp",
        query="SELECT simulator, COUNT(*) AS n FROM v_episode_summary GROUP BY simulator ORDER BY simulator",
    )
    sims = {row["simulator"]: row["n"] for row in q_res["data"]}
    assert sims == {"gazebo": 10, "isaac": 10}


def test_h2_declaration_lint_reports_unresolvable_source_and_unknown_column(
    tmp_path: pathlib.Path,
) -> None:
    """evaluation_lint_declarations passes on shipped declarations and reports planted errors."""
    from arena_evaluation.storage.config_layout import package_config_roots

    data_dir = tmp_path / "data"
    run_dir = build_runs(data_dir / "benchmarks")
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    roots = package_config_roots()
    expected_queries = len({p.parent.name for r in roots for p in r.glob("queries/*/query.yaml")})
    expected_plots = len({p.parent.name for r in roots for p in r.glob("plots/*/plot.yaml")})
    expected_manifests = len({p.parent.name for r in roots for p in r.glob("reports/*/report.yaml")})

    clean_report = schema.evaluation_lint_declarations(run_id=run_dir.name)
    assert clean_report["valid"] is True
    assert clean_report["problems"] == []
    assert clean_report["checked_queries"] == expected_queries
    assert clean_report["checked_plots"] == expected_plots
    assert clean_report["checked_manifests"] == expected_manifests

    good_query_dir = run_dir / "analysis" / "queries" / "good_episode_query"
    good_query_dir.mkdir(parents=True, exist_ok=True)
    (good_query_dir / "query.yaml").write_text(
        "name: good_episode_query\nsql: SELECT planner, COUNT(*) AS n FROM v_episode_summary GROUP BY planner\n",
        encoding="utf-8",
    )
    reference_report = schema.evaluation_lint_declarations(run_id=run_dir.name)
    assert not any("good_episode_query" in p for p in reference_report["problems"])

    bad_query_dir = run_dir / "analysis" / "queries" / "bad_col_query"
    bad_query_dir.mkdir(parents=True, exist_ok=True)
    (bad_query_dir / "query.yaml").write_text(
        "name: bad_col_query\nsql: SELECT nonexistent_column_xyz FROM v_episode_summary\n",
        encoding="utf-8",
    )
    missing_source_dir = run_dir / "analysis" / "plots" / "missing_source_plot"
    missing_source_dir.mkdir(parents=True, exist_ok=True)
    (missing_source_dir / "plot.yaml").write_text(
        "id: missing_source_plot\ntitle: Missing\ntype: bar\ndata:\n  source: v_no_such_view\nchannels:\n  x: planner\n  y: time_to_goal\n",
        encoding="utf-8",
    )

    dirty = schema.evaluation_lint_declarations(run_id=run_dir.name)
    assert not dirty["valid"]
    assert any("bad_col_query" in p for p in dirty["problems"])
    assert any("missing_source_plot" in p for p in dirty["problems"])

    # Drop a required view column in fact_encounters on purpose so v_encounter_summary cannot resolve.
    for enc_path in (run_dir / "episodes").glob("*/metrics/fact_encounters.parquet"):
        enc_df = pl.read_parquet(enc_path)
        if "detour_time_s" in enc_df.columns:
            enc_df.drop("detour_time_s").write_parquet(enc_path)
    unresolvable_plot_dir = run_dir / "analysis" / "plots" / "unresolvable_source_plot"
    unresolvable_plot_dir.mkdir(parents=True, exist_ok=True)
    (unresolvable_plot_dir / "plot.yaml").write_text(
        "id: unresolvable_source_plot\ntitle: Unresolvable\ntype: bar\ndata:\n  source: v_encounter_summary\nchannels:\n  x: planner\n  y: min_clearance_m\n",
        encoding="utf-8",
    )
    unresolvable = schema.evaluation_lint_declarations(run_id=run_dir.name)
    assert not unresolvable["valid"]
    assert any("unresolvable_source_plot" in p and "cannot resolve" in p for p in unresolvable["problems"])


def test_h2_schema_resource_via_mcp_session(tmp_path: pathlib.Path) -> None:
    """arena://evaluation/runs/{id}/schema serves the self-describing schema over MCP."""
    run_dir = build_runs(tmp_path / "data" / "benchmarks")

    async def body(client):
        res = await client.read_resource(f"arena://evaluation/runs/{run_dir.name}/schema")
        return json.loads(res.contents[0].text)

    out = run_session(body, server_env(tmp_path), ("evaluation",))
    assert out["snapshot_id"].startswith("snap-")
    assert any(t["name"] == "v_episode_summary" for t in out["tables"])
