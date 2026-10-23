from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("arena_mcp")
pl = pytest.importorskip("polars")

from arena_mcp.core import ToolError
from mcp_fixtures import EMPTY_RUN_ID, build_runs, metric_rows

from arena_evaluation.mcp import analysis, catalog, runs


def _frame() -> pl.DataFrame:
    return pl.DataFrame(metric_rows())


def _declared() -> dict[str, str]:
    return analysis.directions(analysis.metric_rows())


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
        sim="flatsim",
        headless=False,
        env_n=4,
        optim_obstacles="bbox",
        launch_args={"task.episode.fail_on_collision": True, "complexity": 2},
    )
    assert argv[:6] == ["--suite", json.dumps({"stages": []}), "--contest", "[{name: a}]", "--run-id", "r2"]
    assert {"--strict", "--world-swap"} <= set(argv)
    assert argv[argv.index("--retries") + 1] == "0"
    assert argv[argv.index("--efficacy") + 1] == "0.5"
    assert argv[argv.index("--lanes") + 1] == "3"
    assert {"sim:=flatsim", "headless:=false", "env.n:=4", "optim.obstacles:=bbox", "task.episode.fail_on_collision:=true", "complexity:=2"} <= set(argv)


def test_benchmark_argv_resume_names_the_run_and_refuses_configs():
    assert runs.benchmark_argv(suite=None, contest=None, run_id="old", resume=True, retry_failed=True) == ["--resume", "old", "--retry-failed"]
    with pytest.raises(ToolError, match="replays its stored suite"):
        runs.benchmark_argv(suite="basic", contest=None, run_id="old", resume=True)


def test_launch_tokens_reject_a_key_given_twice_and_reject_traversing_names():
    with pytest.raises(ToolError, match="sim given both"):
        runs.launch_tokens({"sim": "gazebo"}, sim="dummy")
    with pytest.raises(ToolError, match="one path component"):
        runs.benchmark_argv(suite="../etc", contest="basic", run_id="r")


def test_compare_groups_by_contestant_and_drops_references():
    result = analysis.compare_frame(_frame(), ["success", "time_to_goal", "collision_amount"], _declared())
    names = [row["planner"] for row in result["rankings"]]
    assert sorted(names) == ["dwb", "dwb_social"]
    assert names[0] == "dwb"
    assert result["rankings"][0]["rank"] == 1
    assert all(0.0 <= row[key] <= 1.0 for row in result["rankings"] for key in row if key.endswith("_norm"))
    with_refs = analysis.compare_frame(_frame(), ["success"], _declared(), include_references=True)
    assert "unhindered_peds" in [row["planner"] for row in with_refs["rankings"]]


def test_metrics_without_a_declared_direction_are_not_ranked():
    declared = _declared()
    assert "phases_total" not in declared
    result = analysis.compare_frame(_frame(), ["success", "phases_total"], declared)
    assert result["ranked_on"] == ["success"]
    assert result["unranked_metrics"] == ["phases_total"]
    assert not any("phases_total_norm" in row for row in result["rankings"])


def test_judge_agrees_ranks_higher_is_better():
    declared = _declared()
    assert declared["judge_agrees"] == "higher"
    top = analysis.top_n_frame(_frame(), ["judge_agrees"], declared, n=1)
    assert top["top_n"][0]["planner"] == "dwb"


def test_rows_without_any_value_rank_last_instead_of_raising():
    result = analysis.compare_frame(_frame(), ["success", "time_to_goal"], _declared(), include_references=True)
    assert result["rankings"][-1]["planner"] == "unhindered_peds"
    assert result["rankings"][-1]["composite_score"] is None
    assert result["rankings"][-1]["rank"] is None


def test_top_n_weights_must_match_the_metrics():
    with pytest.raises(ToolError, match="2 weights for 1 metrics"):
        analysis.top_n_frame(_frame(), ["success"], _declared(), weights=[1.0, 2.0])
    weighted = analysis.top_n_frame(_frame(), ["success", "time_to_goal"], _declared(), weights=[0.0, 1.0])
    assert weighted["top_n"][0]["planner"] == "dwb"
    with pytest.raises(ToolError, match="declares whether higher or lower"):
        analysis.top_n_frame(_frame(), ["phases_total"], _declared())


def test_correlation_is_bounded_and_refuses_list_columns():
    result = analysis.correlation_frame(analysis.episodes(_frame(), False), ["success", "time_to_goal", "collision_amount"])
    matrix = result["correlation_matrix"]
    assert matrix["success"]["success"] == 1.0
    assert all(-1.0 <= value <= 1.0 for row in matrix.values() for value in row.values() if value is not None)
    with pytest.raises(ToolError, match="one list per episode"):
        analysis.correlation_frame(_frame(), ["success", "path_speeds"])


def test_means_run_over_list_elements_per_column_and_count_episodes():
    frame = _frame().with_columns(pl.Series("speed_samples", [[1.0] * (index + 1) for index in range(10)]))
    rows = {row["stage"]: row for row in analysis.means(frame.filter(pl.col("planner") == "dwb"), ["stage"], ["path_speeds", "speed_samples", "time_to_goal"])}
    assert rows["s1"]["n_rows"] == 2
    assert rows["s1"]["path_speeds"] == pytest.approx((0.5 + 0.6 + 0.5 + 0.7) / 4)
    assert rows["s1"]["speed_samples"] == pytest.approx(1.0)
    assert rows["s2"]["time_to_goal"] == pytest.approx(67.0)


def test_column_stats_recognize_list_columns():
    stats = analysis.column_stats(_frame())
    assert stats["path_speeds"]["dtype"].startswith("List")
    assert "mean_length" in stats["path_speeds"]
    assert stats["planner"]["n_unique"] == 3
    assert stats["time_to_goal"]["min"] == 40.0


def test_notes_replace_append_and_merge(tmp_path):
    path = tmp_path / "notes.yaml"
    runs.write_notes(path, [runs.Note(label="a", value="1"), runs.Note(label="b", value="2")], "replace")
    runs.write_notes(path, [runs.Note(label="c", value="3")], "append")
    runs.write_notes(path, [runs.Note(label="a", value="new"), runs.Note(label="d", value="\u00fc")], "merge")
    assert runs.load_notes(path) == [{"label": "a", "value": "new"}, {"label": "b", "value": "2"}, {"label": "c", "value": "3"}, {"label": "d", "value": "\u00fc"}]
    path.write_text("k1: v1\nk2: v2\n")
    assert runs.load_notes(path) == [{"label": "k1", "value": "v1"}, {"label": "k2", "value": "v2"}]


def test_mode_params_come_from_the_registered_schemas():
    registries = catalog.mode_registries()
    for axis, registry in registries.items():
        for key in registry.keys():
            params = catalog.mode_params(registry, key)
            assert bool(params) == (registry.meta(key).schema is not None), (axis, key)
    random = catalog.describe_mode("tm_obstacles", registries["tm_obstacles"], next(key for key in registries["tm_obstacles"].keys() if key.value == "random"))
    names = {param["name"] for param in random["params"]}
    assert {"dynamic.n", "dynamic.models", "static.n"} <= names
    block = random["stage_example"]["config"]["random"]
    assert set(block["dynamic"]) == {"min", "max", "models"}
    stationary = catalog.describe_mode("tm_robots", registries["tm_robots"], next(key for key in registries["tm_robots"].keys() if key.value == "stationary"))
    assert {"pos_x", "pos_y", "pos_theta"} <= {param["name"] for param in stationary["params"]}
    scenario = catalog.describe_mode("tm_obstacles", registries["tm_obstacles"], next(key for key in registries["tm_obstacles"].keys() if key.value == "scenario"))
    contact = next(param for param in scenario["params"] if param["name"] == "contact_mode")
    assert contact["config_key"] == "config.scenario.contact_mode"
    assert isinstance(contact["enum"], list) and contact["default"] in contact["enum"]


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
    done = subprocess.run([sys.executable, "-c", script, "process", "--benchmark-dir", str(tmp_path / "benchmarks" / EMPTY_RUN_ID), "--workers", "1"], capture_output=True, text=True, timeout=300, env={**os.environ, "ARENA_DATA_DIR": str(tmp_path)}, stdin=subprocess.DEVNULL)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "Processing failed" in done.stdout


def test_runner_lookup_skips_flags_and_reads_the_run_id():
    from arena_evaluation.benchmark.debug import _classify, running_pids_by_run_id

    assert _classify("vim gazebo_simulator.py") is None
    assert _classify("uv run python tool.py") is None
    assert _classify("/usr/bin/gazebo --server") == "simulation"
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "evaluation", "benchmark", "--scale-episodes", "2"], start_new_session=True)
    named = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "evaluation", "benchmark", "--run-id", "run-from-flag"], start_new_session=True)
    try:
        time.sleep(0.5)
        found = running_pids_by_run_id()
        assert "--scale-episodes" not in found
        assert found.get("run-from-flag") == named.pid
    finally:
        for child in (sleeper, named):
            child.kill()
            child.wait()
