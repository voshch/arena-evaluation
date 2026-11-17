"""L5 close-outs that needed the merged FIX1/FIX6/FIX7 code (L5-006, L5-012, X-002)."""

from __future__ import annotations

import ast
import pathlib

import polars as pl

from mcp_fixtures import build_runs, write_star_schema

from arena_evaluation.mcp import analysis, slices


def _validity_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    """A run whose invalid dwb outlier would dominate every quick-look mean if it were kept."""
    run_dir = tmp_path / "data" / "benchmarks" / "l5_validity"
    write_star_schema(
        run_dir,
        [
            {
                "episode_id": 0,
                "planner": "dwb",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": False,
                "success": True,
                "invalid_reason": "",
                "time_to_goal": 40.0,
                "path_efficiency": 0.9,
            },
            {
                "episode_id": 1,
                "planner": "dwb",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": False,
                "success": True,
                "invalid_reason": "",
                "time_to_goal": 42.0,
                "path_efficiency": 0.8,
            },
            {
                "episode_id": 2,
                "planner": "dwb_social",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": False,
                "success": True,
                "invalid_reason": "",
                "time_to_goal": 60.0,
                "path_efficiency": 0.6,
            },
            {
                "episode_id": 3,
                "planner": "dwb_social",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": False,
                "success": True,
                "invalid_reason": "",
                "time_to_goal": 62.0,
                "path_efficiency": 0.5,
            },
            {
                "episode_id": 4,
                "planner": "dwb",
                "stage": "s1",
                "world": "map_empty",
                "is_reference": False,
                "success": False,
                "invalid_reason": "collision",
                "time_to_goal": 900.0,
                "path_efficiency": 0.99,
            },
        ],
    )
    return run_dir


def test_l5_006_quick_look_analyses_exclude_invalid_episodes(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Aggregate, compare and correlate keep the valid episodes only and drop the invalid outlier."""
    run_dir = _validity_fixture(tmp_path)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    compare = analysis.evaluation_compare_planners(
        run_dir.name, metrics=["time_to_goal"]
    )
    by_planner = {row["planner"]: row for row in compare["rankings"]}
    assert by_planner["dwb"]["time_to_goal"] == 41.0
    assert by_planner["dwb_social"]["time_to_goal"] == 61.0

    aggregate = analysis.evaluation_aggregate(
        run_dir.name, dimension="planner", metrics=["time_to_goal"]
    )
    means = {row["planner"]: row["time_to_goal"] for row in aggregate["data"]}
    assert means == {"dwb": 41.0, "dwb_social": 61.0}

    correlation = analysis.evaluation_correlation(
        run_dir.name, ["time_to_goal", "path_efficiency"]
    )
    # Only the four valid rows contribute; the invalid (900 s, 0.99) row would flip the sign.
    r = correlation["correlation_matrix"]["time_to_goal"]["path_efficiency"]
    assert r is not None and r < -0.9


def test_l5_006_the_invalid_row_is_still_published_and_counted(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """The invalid row stays in the Star Schema; the quick-look path excludes it by the flag."""
    run_dir = _validity_fixture(tmp_path)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))

    from arena_evaluation.analytics.catalog import BenchmarkCatalog

    summary = BenchmarkCatalog.open(run_dir).get_table("v_episode_summary").collect()
    assert summary.height == 5
    assert summary.filter(pl.col("is_valid_for_ranking")).height == 4


def test_l5_012_worst_encounters_default_metric_takes_its_direction_from_the_run_dictionary(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """The default min_clearance_m ranking reads 'higher' from dim_metrics and ranks worst first."""
    run_dir = tmp_path / "data" / "benchmarks"
    run = build_runs(run_dir)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path / "data"))
    from arena_evaluation.storage.registry.facts import FACT_TABLES

    schema = FACT_TABLES["encounters"].polars_schema()
    rows = [
        {
            **{name: None for name in schema.names()},
            "run_id": "run",
            "episode_id": 0,
            "robot_id": "robot_0",
            "env_id": "env_0",
            "encounter_id": encounter_id,
            "actor_1": "robot_0",
            "actor_2": ped,
            "actor_2_kind": "pedestrian",
            "t_start_ns": t_start,
            "t_end_ns": t_end,
            "min_clearance_m": clearance,
            "duration_s": duration,
        }
        for encounter_id, ped, t_start, t_end, clearance, duration in [
            ("e1", "ped_0", 100, 200, 0.4, 2.0),
            ("e2", "ped_1", 200, 350, 0.8, 5.0),
        ]
    ]
    pl.DataFrame(rows, schema=schema).write_parquet(
        run / "episodes" / "episode_000" / "metrics" / "fact_encounters.parquet"
    )

    ranked = slices.evaluation_worst_encounters(run.name, episode_id=0)

    assert ranked.metric == "min_clearance_m"
    assert ranked.direction == "higher"
    assert [row["encounter_id"] for row in ranked.rows] == ["e1", "e2"]

    override = slices.evaluation_worst_encounters(
        run.name, episode_id=0, direction="lower"
    )
    assert override.direction == "lower"
    assert [row["encounter_id"] for row in override.rows] == ["e2", "e1"]


def test_x_002_the_mcp_no_longer_imports_processing_sensitivity() -> None:
    """The declaration-time MCP -> processing.sensitivity edge is gone; the pin must not be needed."""
    mcp_root = pathlib.Path(__file__).parents[2] / "arena_evaluation" / "mcp"
    forbidden: list[str] = []
    for path in sorted(mcp_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("arena_evaluation.processing.sensitivity"):
                    forbidden.append(path.name)
            elif isinstance(node, ast.Import):
                if any(
                    alias.name.startswith("arena_evaluation.processing.sensitivity")
                    for alias in node.names
                ):
                    forbidden.append(path.name)
    assert not forbidden, f"MCP still imports processing.sensitivity: {forbidden}"
