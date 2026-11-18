"""Contract tests for PR5/H3: named analyses, synthetic truth, and sensitivity reruns."""

from __future__ import annotations

import os

import hashlib
import pathlib


from synthetic_benchmark import write_synthetic_benchmark

from arena_evaluation.mcp import analysis


def test_h3_list_procedures_returns_six_named_analyses() -> None:
    """evaluation_analysis() without procedure lists all 6 named analyses and schemas."""
    listing = analysis.evaluation_analysis()
    names = {p["name"] for p in listing["procedures"]}
    assert {
        "triage",
        "compare",
        "ranking_stability",
        "visibility_effect",
        "lead_measures",
        "attribution_shares",
    }.issubset(names)


def test_h3_compare_planted_diff_and_null_pair_and_all_null(
    tmp_path: pathlib.Path,
) -> None:
    """compare interval includes planted difference (-4.0 without swap) and includes 0 for null pair."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "syn_noswap"
    truth = write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    diff_res = analysis.evaluation_analysis(
        procedure="compare",
        run_id="syn_noswap",
        params={
            "planner_a": truth.planted_diff_a,
            "planner_b": truth.planted_diff_b,
            "metric": truth.planted_diff_metric,
        },
    )
    assert diff_res["evidence_id"]
    assert diff_res["snapshot_id"].startswith("snap-")
    row = diff_res["summary"]
    assert row["ci_low"] <= truth.planted_diff_delta <= row["ci_high"]
    assert row["ci_high"] < 0.0

    null_res = analysis.evaluation_analysis(
        procedure="compare",
        run_id="syn_noswap",
        params={
            "planner_a": truth.null_pair_a,
            "planner_b": truth.null_pair_b,
            "metric": truth.planted_diff_metric,
        },
    )
    null_row = null_res["summary"]
    assert null_row["ci_low"] <= 0.0 <= null_row["ci_high"]

    all_null_res = analysis.evaluation_analysis(
        procedure="compare",
        run_id="syn_noswap",
        params={
            "planner_a": truth.planted_diff_a,
            "planner_b": truth.planted_diff_b,
            "metric": truth.all_null_metric,
        },
    )
    assert all_null_res["n_rows"] == 0
    assert all_null_res["excluded_rows"]


def test_h3_ranking_stability_and_visibility_effect_detect_planted_swap(
    tmp_path: pathlib.Path,
) -> None:
    """visibility_effect is True on swap fixture and False on no-swap fixture."""
    data_dir = tmp_path / "data"
    swap_dir = data_dir / "benchmarks" / "syn_swap"
    noswap_dir = data_dir / "benchmarks" / "syn_noswap"
    write_synthetic_benchmark(swap_dir, include_swap=True)
    write_synthetic_benchmark(noswap_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    stab = analysis.evaluation_analysis(
        procedure="ranking_stability",
        run_id="syn_swap",
        params={"metric": "time_to_goal"},
    )
    assert stab["summary"]["n_swaps"] >= 1

    vis_true = analysis.evaluation_analysis(
        procedure="visibility_effect",
        run_id="syn_swap",
        params={"metric": "time_to_goal"},
    )
    assert vis_true["summary"]["effect_detected"] is True

    vis_false = analysis.evaluation_analysis(
        procedure="visibility_effect",
        run_id="syn_noswap",
        params={"metric": "time_to_goal"},
    )
    assert vis_false["summary"]["effect_detected"] is False


def test_h3_lead_measures_and_attribution_shares_and_triage(
    tmp_path: pathlib.Path,
) -> None:
    """lead_measures records disjoint seeds; attribution_shares uses summed parts."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "syn_swap"
    truth = write_synthetic_benchmark(run_dir, include_swap=True)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    triage_res = analysis.evaluation_analysis(
        procedure="triage", run_id="syn_swap", params={}
    )
    assert triage_res["n_rows"] > 0

    lead_res = analysis.evaluation_analysis(
        procedure="lead_measures", run_id="syn_swap", params={}
    )
    disc = set(lead_res["summary"]["discovery_seeds"])
    ver = set(lead_res["summary"]["verification_seeds"])
    assert disc and ver and disc.isdisjoint(ver)

    attr_res = analysis.evaluation_analysis(
        procedure="attribution_shares", run_id="syn_swap", params={}
    )
    pa_rows = [r for r in attr_res["preview"] if r["planner"] == "planner_a"]
    assert pa_rows
    total_robot = sum(r["robot_closing_sum"] for r in pa_rows)
    total_target = sum(r["target_closing_sum"] for r in pa_rows)
    robot_share = total_robot / (total_robot + total_target)
    assert (
        abs(robot_share - truth.closer_shares["planner_a"]["robot_closing_share"])
        < 1e-6
    )


def test_h3_sensitivity_leaves_published_files_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    """evaluation_sensitivity reruns in memory and never modifies published parquet files."""
    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "syn_sens"
    write_synthetic_benchmark(run_dir, include_swap=False)
    os.environ["ARENA_DATA_DIR"] = str(data_dir)

    def _published_hashes() -> dict[pathlib.Path, str]:
        return {
            p.relative_to(run_dir): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob("*.parquet"))
            if p.relative_to(run_dir).parts[0] != "analysis"
        }

    before = _published_hashes()
    res = analysis.evaluation_sensitivity(
        run_id="syn_sens",
        episodes=[0, 1],
        detector_params={"encounter_distance_m": 2.5},
    )
    assert res["is_sensitivity"] is True
    assert res["n_rows"] >= 2

    after = _published_hashes()
    assert before == after


def test_h3_sensitivity_parameter_change_alters_rerun_encounters(
    tmp_path: pathlib.Path,
) -> None:
    """Changing detector_params alters in-memory rerun encounter counts while keeping disk files untouched."""
    import polars as pl
    from arena_evaluation.analytics.catalog import BenchmarkCatalog
    from arena_evaluation.storage.registry.telemetry import (
        PedestrianAgentStates,
        RobotTracks,
    )

    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "synth_sens_params"
    write_synthetic_benchmark(run_dir, include_swap=False)
    ep0 = run_dir / "episodes" / "episode_000"
    (ep0 / "topics").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "robot_id": ["robot_0", "robot_0", "robot_0"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "vx": [1.0, 1.0, 1.0],
            "vy": [0.0, 0.0, 0.0],
        }
    ).write_parquet(ep0 / RobotTracks.relative_path())
    pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000, 2_000_000_000],
            "ped_id": [1, 1, 1],
            "ped_name": ["ped_1", "ped_1", "ped_1"],
            "x": [1.0, 1.0, 1.0],
            "y": [2.0, 2.0, 2.0],
            "vx": [0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0],
            "radius_m": [0.3, 0.3, 0.3],
        }
    ).write_parquet(ep0 / PedestrianAgentStates.relative_path())

    from arena_evaluation.storage.registry.dimensions import DIMENSION_TABLES

    dim_robots_schema = DIMENSION_TABLES["dim_robots"].polars_schema()
    pl.DataFrame(
        [
            {col: None for col in dim_robots_schema.names()}
            | {"robot_id": "robot_0", "robot_radius": 0.3}
        ],
        schema=dim_robots_schema,
    ).write_parquet(ep0 / DIMENSION_TABLES["dim_robots"].relative_path())

    cat = BenchmarkCatalog.open(run_dir, variant="sensitivity")
    assert cat.variant == "sensitivity"
    assert cat.is_sensitivity is True

    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    tight = analysis.evaluation_sensitivity(
        run_id="synth_sens_params",
        episodes=[0],
        detector_params={"encounter_edge_distance_m": 0.5},
    )
    wide = analysis.evaluation_sensitivity(
        run_id="synth_sens_params",
        episodes=[0],
        detector_params={"encounter_edge_distance_m": 4.0},
    )
    assert tight["preview"][0]["rerun_n_encounters"] == 0
    assert wide["preview"][0]["rerun_n_encounters"] >= 1
    assert wide["preview"][0]["rerun_total_duration_s"] > 0.0


def test_h3_sensitivity_missing_pedestrian_states_returns_null_with_reason(
    tmp_path: pathlib.Path,
) -> None:
    """Episodes missing PedestrianAgentStates stay in the result with NULL rerun values and null_reason."""
    import os
    import polars as pl
    from arena_evaluation.storage.registry.telemetry import RobotTracks

    data_dir = tmp_path / "data"
    run_dir = data_dir / "benchmarks" / "synth_sens_missing_ped"
    write_synthetic_benchmark(run_dir, include_swap=False)
    ep0 = run_dir / "episodes" / "episode_000"
    (ep0 / "topics").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "t_ns": [0, 1_000_000_000],
            "robot_id": ["robot_0", "robot_0"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "yaw": [0.0, 0.0],
            "vx": [1.0, 1.0],
            "vy": [0.0, 0.0],
        }
    ).write_parquet(ep0 / RobotTracks.relative_path())

    os.environ["ARENA_DATA_DIR"] = str(data_dir)
    res = analysis.evaluation_sensitivity(run_id="synth_sens_missing_ped", episodes=[0])
    row = res["preview"][0]
    assert row["rerun_n_encounters"] is None
    assert row["null_reason"] == "telemetry:no_pedestrian_states"
