from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import polars as pl

from arena_mcp.core import ToolError
from mcp_fixtures import EMPTY_RUN_ID, build_runs, metric_rows

from arena_evaluation.mcp import analysis, authoring, runs, slices


def _frame() -> pl.DataFrame:
    return pl.DataFrame(metric_rows(), infer_schema_length=None)


def _write_star_run(tmp_path: Path) -> str:
    run_dir = build_runs(tmp_path / "benchmarks")
    from arena_evaluation.storage.registry.facts import FACT_TABLES

    enc_schema = FACT_TABLES["encounters"].polars_schema()
    enc_rows = [
        {
            **{name: None for name in enc_schema.names()},
            "run_id": "run",
            "episode_id": 0,
            "robot_id": "robot_0",
            "encounter_id": eid,
            "actor_1": "robot_0",
            "actor_2": ped,
            "actor_2_kind": "pedestrian",
            "t_start_ns": ts,
            "t_end_ns": te,
            "min_clearance_m": clr,
            "duration_s": dur,
        }
        for eid, ped, ts, te, clr, dur in [
            ("e1", "ped_0", 100, 200, 0.4, 2.0),
            ("e2", "ped_1", 200, 350, 0.8, 5.0),
        ]
    ]
    enc_dir = run_dir / "episodes" / "episode_000" / "metrics"
    enc_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(enc_rows, schema=enc_schema).write_parquet(
        enc_dir / "fact_encounters.parquet"
    )
    return run_dir.name


def test_benchmark_argv_passes_only_what_the_caller_set():
    argv = runs.benchmark_argv(suite="basic", contest="basic", run_id="r1")
    assert argv == ["--suite", "basic", "--contest", "basic", "--run-id", "r1"]
    assert not any(":=" in token for token in argv)


def test_benchmark_argv_renders_runner_options_and_launch_args():
    argv = runs.benchmark_argv(
        suite={"stages": []},
        contest="[{name: a}]",
        run_id="r2",
        retries=0,
        strict=True,
        efficacy=0.5,
        spawn_budget=120.0,
        world_swap=True,
        lanes=3,
        ros_domain_id=57,
        sim="flatsim",
        headless=False,
        env_n=4,
        optim_obstacles="bbox",
        launch_args={"task.episode.fail_on_collision": True, "complexity": 2},
        viz=True,
        viz_port=9000,
    )
    assert argv[:6] == [
        "--suite",
        json.dumps({"stages": []}),
        "--contest",
        "[{name: a}]",
        "--run-id",
        "r2",
    ]
    assert {"--strict", "--world-swap", "--viz"} <= set(argv)
    assert argv[argv.index("--viz-port") + 1] == "9000"
    assert argv[argv.index("--retries") + 1] == "0"
    assert argv[argv.index("--efficacy") + 1] == "0.5"
    assert argv[argv.index("--lanes") + 1] == "3"
    assert argv[argv.index("--ros-domain-id") + 1] == "57"
    assert {
        "sim:=flatsim",
        "headless:=false",
        "env.n:=4",
        "optim.obstacles:=bbox",
        "task.episode.fail_on_collision:=true",
        "complexity:=2",
    } <= set(argv)


def test_benchmark_argv_resume_names_the_run_and_refuses_configs():
    assert runs.benchmark_argv(
        suite=None, contest=None, run_id="old", resume=True, retry_failed=True
    ) == ["--resume", "old", "--retry-failed"]
    with pytest.raises(ToolError, match="replays its stored suite"):
        runs.benchmark_argv(suite="basic", contest=None, run_id="old", resume=True)


def test_launch_tokens_reject_a_key_given_twice_and_reject_traversing_names():
    with pytest.raises(ToolError, match="sim given both"):
        runs.launch_tokens({"sim": "gazebo"}, sim="dummy")
    with pytest.raises(ToolError, match="one path component"):
        runs.benchmark_argv(suite="../etc", contest="basic", run_id="r")


def test_compare_groups_by_contestant_and_drops_references(tmp_path):
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    run_id = _write_star_run(tmp_path)
    result = analysis.evaluation_compare_planners(
        run_id, metrics=["success", "time_to_goal", "path_efficiency"]
    )
    names = [row["planner"] for row in result["rankings"]]
    assert sorted(names) == ["dwb", "dwb_social"]
    assert names[0] == "dwb"
    assert result["rankings"][0]["rank"] == 1
    assert all(
        0.0 <= row[key] <= 1.0
        for row in result["rankings"]
        for key in row
        if key.endswith("_norm")
    )
    with_refs = analysis.evaluation_compare_planners(
        run_id, metrics=["success"], include_references=True
    )
    assert "unhindered_peds" in [row["planner"] for row in with_refs["rankings"]]


def test_metrics_without_a_declared_direction_raise_hinted_tool_error(tmp_path):
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    run_id = _write_star_run(tmp_path)
    with pytest.raises(ToolError, match="direction"):
        analysis.evaluation_compare_planners(
            run_id, metrics=["success", "num_pedestrians"]
        )
    with pytest.raises(ToolError, match="direction"):
        slices.evaluation_worst_encounters(run_id, metric="num_pedestrians")


def test_a_canonical_measure_with_a_declared_direction_ranks_higher_is_better(tmp_path):
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    run_id = _write_star_run(tmp_path)
    desc = analysis.evaluation_describe_metric("mar", run_id=run_id)
    assert desc["direction"] == "higher"
    top = analysis.evaluation_top_n(run_id, n=1, metrics=["success", "time_to_goal"])
    assert top["top_n"][0]["planner"] in {"dwb", "dwb_social"}


def test_top_n_weights_must_match_the_metrics(tmp_path):
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    run_id = _write_star_run(tmp_path)
    with pytest.raises(ToolError, match="2 weights for 1 metrics"):
        analysis.evaluation_top_n(run_id, metrics=["success"], weights=[1.0, 2.0])
    weighted = analysis.evaluation_top_n(
        run_id, metrics=["success", "time_to_goal"], weights=[0.0, 1.0]
    )
    assert weighted["top_n"][0]["planner"] == "dwb"


def test_correlation_is_bounded_and_worst_encounters_reads_dim_metrics(tmp_path):
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    run_id = _write_star_run(tmp_path)
    result = analysis.evaluation_correlation(
        run_id, ["success", "time_to_goal", "path_efficiency"]
    )
    matrix = result["correlation_matrix"]
    assert matrix["success"]["success"] == 1.0
    assert all(
        -1.0 <= value <= 1.0
        for row in matrix.values()
        for value in row.values()
        if value is not None
    )
    worst = slices.evaluation_worst_encounters(run_id, metric="min_clearance_m")
    assert worst.direction == "higher"
    assert worst.rows[0]["encounter_id"] == "e1"


def test_column_stats_recognize_list_columns():
    frame = _frame().with_columns(pl.Series("path_speeds", [[0.5, 0.6]] * 10))
    stats = analysis.column_stats(frame)
    assert stats["path_speeds"]["dtype"].startswith("List")
    assert "mean_length" in stats["path_speeds"]
    assert stats["planner"]["n_unique"] == 3
    assert stats["time_to_goal"]["min"] == 40.0


def test_notes_replace_append_and_merge(tmp_path):
    path = tmp_path / "notes.yaml"
    runs.write_notes(
        path,
        [runs.Note(label="a", value="1"), runs.Note(label="b", value="2")],
        "replace",
    )
    runs.write_notes(path, [runs.Note(label="c", value="3")], "append")
    runs.write_notes(
        path,
        [runs.Note(label="a", value="new"), runs.Note(label="d", value="\u00fc")],
        "merge",
    )
    assert runs.load_notes(path) == [
        {"label": "a", "value": "new"},
        {"label": "b", "value": "2"},
        {"label": "c", "value": "3"},
        {"label": "d", "value": "\u00fc"},
    ]
    path.write_text("k1: v1\nk2: v2\n")
    assert runs.load_notes(path) == [
        {"label": "k1", "value": "v1"},
        {"label": "k2", "value": "v2"},
    ]


def test_mode_params_come_from_the_registered_schemas():
    from arena_runtime.env.task_layer import registered_modes, task_layer
    from task_generator.tasks.registry import (
        MODULE_MODES,
        OBSTACLES_MODES,
        ROBOTS_MODES,
    )

    layer = task_layer()
    registered = authoring.mode_params(layer, registered_modes(layer))
    for axis, registry in (
        ("robots", ROBOTS_MODES),
        ("obstacles", OBSTACLES_MODES),
        ("modules", MODULE_MODES),
    ):
        assert set(registered[axis]) == {key.value for key in registry.keys()}
        for key in registry.keys():
            assert bool(registered[axis][key.value]) == (
                registry.meta(key).schema is not None
            ), (axis, key)

    def names(axis: str, mode: str) -> dict[str, dict]:
        return {param["name"]: param for param in registered[axis][mode]}

    assert {"dynamic.n", "dynamic.models", "static.n"} <= set(
        names("obstacles", "random")
    )
    assert {"pos_x", "pos_y", "pos_theta"} <= set(names("robots", "stationary"))
    assert isinstance(names("obstacles", "scenario")["contact_mode"]["enum"], list)


def test_progress_rows_fold_lane_files_without_writing(tmp_path):
    from arena_evaluation.benchmark.state import progress_rows

    path = build_runs(tmp_path)
    before = (path / "progress.csv").read_text()
    rows = progress_rows(path)
    assert [row["step_key"] for row in rows] == ["dwb/s1", "dwb/s1", "dwb_social/s1"]
    assert (path / "progress.csv").read_text() == before


def test_process_exits_nonzero_when_nothing_was_written(tmp_path):
    build_runs(tmp_path / "benchmarks")
    script = "import sys\nfrom arena_evaluation.cli import main\nsys.argv[0] = 'evaluation'\nmain()"
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "process",
            "--benchmark-dir",
            str(tmp_path / "benchmarks" / EMPTY_RUN_ID),
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "ARENA_DATA_DIR": str(tmp_path)},
        stdin=subprocess.DEVNULL,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "Processing failed" in done.stdout


def test_runner_lookup_skips_flags_and_reads_the_run_id():
    from arena_evaluation.benchmark.debug import _classify, running_pids_by_run_id

    assert _classify("vim gazebo_simulator.py") is None
    assert _classify("uv run python tool.py") is None
    assert _classify("/usr/bin/gazebo --server") == "simulation"
    sleeper = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "evaluation",
            "benchmark",
            "--scale-episodes",
            "2",
        ],
        start_new_session=True,
    )
    named = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            "evaluation",
            "benchmark",
            "--run-id",
            "run-from-flag",
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 20.0
        found = running_pids_by_run_id()
        while "run-from-flag" not in found and time.monotonic() < deadline:
            time.sleep(0.1)
            found = running_pids_by_run_id()
        assert "--scale-episodes" not in found
        assert found.get("run-from-flag") == named.pid
    finally:
        for child in (sleeper, named):
            child.kill()
            child.wait()


def test_mode_params_declare_only_the_selected_modes():
    from arena_runtime.env.task_layer import task_layer

    layer = task_layer()
    assert authoring.mode_params(layer, {"obstacles": ["random"]}).keys() == {
        "obstacles"
    }
    assert list(
        authoring.mode_params(layer, {"obstacles": ["random"]})["obstacles"]
    ) == ["random"]
    assert authoring.mode_params(layer, {}) == {}
