from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("polars")

import yaml
from mcp_fixtures import EMPTY_RUN_ID, RUN_ID, build_runs, error_text, payload, run_session, server_env, sourced

from arena_evaluation.mcp.base import domain

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "configs" / "benchmark"


def test_server_lists_every_tool_resource_and_template(tmp_path):
    async def body(client):
        tools = (await client.list_tools()).tools
        resources = (await client.list_resources()).resources
        templates = (await client.list_resource_templates()).resource_templates
        prompts = (await client.list_prompts()).prompts
        return tools, resources, templates, prompts

    tools, resources, templates, prompts = run_session(body, server_env(tmp_path), ("evaluation",))
    assert sorted(tool.name for tool in tools) == sorted(spec.name for spec in domain.tools)
    assert all(tool.name.startswith("evaluation_") and tool.description for tool in tools)
    destructive = {tool.name for tool in tools if tool.annotations.destructive_hint}
    assert {"evaluation_benchmark", "evaluation_preflight", "evaluation_stop", "evaluation_kill", "evaluation_record_start", "evaluation_record_stop"} <= destructive
    assert {str(resource.uri) for resource in resources} >= {"arena://evaluation/runs", "arena://evaluation/suites", "arena://evaluation/contests", "arena://evaluation/manifests"}
    assert {template.uri_template for template in templates} >= {"arena://evaluation/runs/{run_id}/state", "arena://evaluation/runs/{run_id}/notes", "arena://evaluation/runs/{run_id}/report", "arena://evaluation/runs/{run_id}/metrics/schema", "arena://evaluation/runs/{run_id}/metrics/summary"}
    assert [prompt.name for prompt in prompts] == ["evaluation_benchmark_workflow"]


def test_catalog_tools_read_the_registries_and_resolvers(tmp_path):
    async def body(client):
        calls = {
            "metric": ("evaluation_describe_metric", {"metric_name": "collision_metrics"}),
            "replay": ("evaluation_describe_metric", {"metric_name": "task_replay"}),
            "metrics": ("evaluation_metrics", {}),
            "suites": ("evaluation_suites", {}),
            "contests": ("evaluation_contests", {}),
            "manifests": ("evaluation_manifests", {}),
            "template": ("evaluation_config_template", {"kind": "suite", "name": "basic"}),
            "manifest_template": ("evaluation_config_template", {"kind": "manifest", "name": "standard.yaml"}),
            "schema": ("evaluation_config_schema", {"kind": "suite"}),
            "contest_schema": ("evaluation_config_schema", {"kind": "contest"}),
        }
        out = {key: payload(await client.call_tool(name, args)) for key, (name, args) in calls.items()}
        out["bad_metric"] = error_text(await client.call_tool("evaluation_describe_metric", {"metric_name": "zz"}))
        out["traversal"] = error_text(await client.call_tool("evaluation_config_template", {"kind": "suite", "name": "../secrets"}))
        out["suites_resource"] = json.loads((await client.read_resource("arena://evaluation/suites")).contents[0].text)
        return out

    out = run_session(body, server_env(tmp_path), ("evaluation",))
    assert out["metric"]["outputs"]["success"]["direction"] == "higher"
    assert out["metric"]["outputs"]["collision_amount"]["lower_is_better"] is True
    assert out["replay"]["outputs"]["judge_agrees"]["direction"] == "higher"
    assert "no metric 'zz'" in out["bad_metric"]
    assert {row["name"] for row in out["metrics"]["metrics"]} >= {"collision_metrics", "task_replay"}
    assert {path.stem for path in (CONFIGS / "suites").glob("*.yaml")} <= set(out["suites"]["suites"])
    assert out["suites_resource"] == out["suites"]["suites"]
    assert {path.stem for path in (CONFIGS / "contests").glob("*.yaml")} <= set(out["contests"]["contests"])
    assert {path.stem for path in (CONFIGS / "manifests").glob("*.yaml")} <= set(out["manifests"]["manifests"])
    assert yaml.safe_load(out["template"]["content"]) == yaml.safe_load((CONFIGS / "suites" / "basic.yaml").read_text())
    assert out["manifest_template"]["name"] == "standard"
    assert "one path component" in out["traversal"]
    assert {"name", "map", "robot", "tm_robots", "tm_obstacles", "config", "seed", "timeout", "optim", "timeout_peds"} <= set(out["schema"]["stage_fields"])
    assert "## Suite Files" in out["schema"]["readme"]
    assert out["contest_schema"]["contestant_fields"] == ["name", "args", "parts"]


def test_authoring_tools_validate_with_the_runner_parsers(tmp_path):
    suite_with_typos = yaml.safe_dump({"stages": [{"name": "s1", "map": "map_empty", "robot": "jackal", "episodes": 1, "tm_robots": "random", "tm_obstacles": "random", "config": {"random": {"dynamic": {"min": 1, "max": 2, "models": ["arenien"]}, "statik": {"min": 0, "max": 0}}}}]})
    bad_contest = "[{name: bad, mobile: {driver: nav2, local_planner: nope_planner}}, {name: warp, mobile: {driver: warp_drive}}]"

    async def body(client):
        out = {}
        out["basic_suite"] = payload(await client.call_tool("evaluation_validate_suite", {"yaml_content": (CONFIGS / "suites" / "basic.yaml").read_text()}))
        out["typos"] = payload(await client.call_tool("evaluation_validate_suite", {"yaml_content": suite_with_typos}))
        out["bad_mode"] = payload(await client.call_tool("evaluation_validate_suite", {"yaml_content": "stages: [{name: s, map: map_empty, robot: jackal, episodes: 1, tm_robots: warp, tm_obstacles: random}]"}))
        out["basic_contest"] = payload(await client.call_tool("evaluation_validate_contest", {"yaml_content": (CONFIGS / "contests" / "basic.yaml").read_text()}))
        out["bad_contest"] = payload(await client.call_tool("evaluation_validate_contest", {"yaml_content": bad_contest}))
        out["manifests"] = [payload(await client.call_tool("evaluation_validate_manifest", {"yaml_content": path.read_text()})) for path in sorted((CONFIGS / "manifests").glob("*.yaml"))]
        out["bad_manifest"] = payload(await client.call_tool("evaluation_validate_manifest", {"yaml_content": "plots:\n  - id: bad\n    type: violin\n    title: no data key\n"}))
        out["clobber"] = error_text(await client.call_tool("evaluation_create_suite", {"name": "basic", "yaml_content": (CONFIGS / "suites" / "basic.yaml").read_text()}))
        out["empty"] = error_text(await client.call_tool("evaluation_validate_suite", {"yaml_content": "  "}))
        return out

    out = run_session(body, server_env(tmp_path), ("evaluation",))
    assert out["basic_suite"]["valid"], out["basic_suite"]
    assert out["typos"]["valid"]
    assert any("arenien" in warning for warning in out["typos"]["warnings"])
    assert any("statik" in warning for warning in out["typos"]["warnings"])
    assert not out["bad_mode"]["valid"]
    assert out["basic_contest"]["valid"], out["basic_contest"]
    assert not out["bad_contest"]["valid"]
    assert any("nope_planner" in problem for problem in out["bad_contest"]["problems"])
    assert any("warp_drive" in problem for problem in out["bad_contest"]["problems"])
    assert all(verdict["valid"] and verdict["n_plots"] >= 1 for verdict in out["manifests"])
    assert not out["bad_manifest"]["valid"]
    assert "already exists" in out["clobber"]
    assert "empty suite" in out["empty"]


def test_run_and_analysis_tools_read_shared_runs(tmp_path):
    build_runs(tmp_path / "data" / "benchmarks")

    async def body(client):
        out = {}
        for key, name, args in (
            ("runs", "evaluation_runs", {}),
            ("only_done", "evaluation_runs", {"state": "completed"}),
            ("status", "evaluation_status", {"run_id": RUN_ID}),
            ("latest", "evaluation_status", {}),
            ("progress", "evaluation_progress", {"run_id": RUN_ID, "last": 2}),
            ("lane_progress", "evaluation_progress", {"run_id": RUN_ID, "contestant": "dwb_social"}),
            ("console", "evaluation_console", {"run_id": RUN_ID, "lines": 5}),
            ("lane_console", "evaluation_console", {"run_id": RUN_ID, "lane": "p2"}),
            ("episode", "evaluation_read_episode", {"run_id": RUN_ID, "episode_id": 0}),
            ("written", "evaluation_write_notes", {"run_id": RUN_ID, "notes": [{"label": "a", "value": "1"}]}),
            ("insight", "evaluation_append_insight", {"run_id": RUN_ID, "label": "b", "value": "2"}),
            ("notes", "evaluation_read_notes", {"run_id": RUN_ID}),
            ("combined", "evaluation_read_combined_metrics", {"run_id": RUN_ID}),
            ("query", "evaluation_query_metrics", {"run_id": RUN_ID, "group_by": ["planner"], "metrics": ["time_to_goal"], "filter": {"stage": "s1"}, "sort_by": "time_to_goal"}),
            ("aggregate", "evaluation_aggregate", {"run_id": RUN_ID, "dimension": "stage", "metrics": ["success"]}),
            ("compare", "evaluation_compare_planners", {"run_id": RUN_ID, "metrics": ["success", "collision_amount"]}),
            ("compare_default", "evaluation_compare_planners", {"run_id": RUN_ID}),
            ("top", "evaluation_top_n", {"run_id": RUN_ID, "n": 1, "metrics": ["success", "time_to_goal"]}),
            ("correlation", "evaluation_correlation", {"run_id": RUN_ID, "metrics": ["success", "time_to_goal"]}),
            ("summary", "evaluation_summarize", {"run_id": RUN_ID}),
            ("manifest", "evaluation_create_manifest", {"name": "my_report", "template": "standard", "run_id": RUN_ID}),
            ("run_manifests", "evaluation_manifests", {"run_id": RUN_ID}),
            ("custom_template", "evaluation_config_template", {"kind": "manifest", "name": "my_report", "run_id": RUN_ID}),
        ):
            out[key] = payload(await client.call_tool(name, args))
        out["missing_run"] = error_text(await client.call_tool("evaluation_status", {"run_id": "nope"}))
        out["unprocessed"] = error_text(await client.call_tool("evaluation_read_combined_metrics", {"run_id": EMPTY_RUN_ID}))
        out["bad_column"] = error_text(await client.call_tool("evaluation_query_metrics", {"run_id": RUN_ID, "group_by": ["planner"], "metrics": ["zz"]}))
        for uri in (f"arena://evaluation/runs/{RUN_ID}/state", f"arena://evaluation/runs/{RUN_ID}/notes", f"arena://evaluation/runs/{RUN_ID}/report", f"arena://evaluation/runs/{RUN_ID}/metrics/schema", f"arena://evaluation/runs/{RUN_ID}/metrics/summary", "arena://evaluation/runs"):
            out[uri] = json.loads((await client.read_resource(uri)).contents[0].text)
        return out

    out = run_session(body, server_env(tmp_path), ("evaluation",))
    assert [run["run_id"] for run in out["runs"]["runs"]] == [RUN_ID, EMPTY_RUN_ID]
    assert out["only_done"]["runs"] == []
    status = out["status"]
    assert status["steps_total"] == 4
    assert status["steps"] == {"ok": 1, "failed": 1, "partial": 0, "skipped": 0, "in_progress": 1, "pending": 1}
    assert status["state"] == "incomplete"
    assert status["lanes"] == ["p2"]
    assert status["failed"] == [{"key": "dwb_social/s1", "error_kind": "sim_dead", "error_detail": "gz died"}]
    assert status["episodes_logged"] == 3
    assert out["latest"]["run_id"] == RUN_ID
    assert [row["episode_id"] for row in out["progress"]["rows"]] == ["1", "2"]
    assert out["lane_progress"]["n_rows"] == 1
    assert out["console"]["logs"]["runner.log"] == [f"runner line {index}" for index in range(295, 300)]
    assert out["console"]["logs"]["runner.p2.log"] == ["lane p2 started", "lane p2 sim died"]
    assert out["console"]["alive"] is False
    assert out["lane_console"]["logs"] == {"runner.p2.log": ["lane p2 started", "lane p2 sim died"]}
    assert out["episode"]["metadata"]["planner"] == "dwb"
    assert out["notes"]["notes"] == [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}]
    assert out["combined"]["column_stats"]["path_speeds"]["dtype"].startswith("List")
    assert [row["planner"] for row in out["query"]["data"]] == ["dwb_social", "dwb"]
    assert {row["stage"] for row in out["aggregate"]["data"]} == {"s1", "s2"}
    assert [row["planner"] for row in out["compare"]["rankings"]] == ["dwb", "dwb_social"]
    assert "unhindered_peds" not in [row["planner"] for row in out["compare_default"]["rankings"]]
    assert out["top"]["top_n"][0]["planner"] == "dwb"
    assert out["correlation"]["correlation_matrix"]["success"]["success"] == 1.0
    assert out["summary"]["planners"] == ["dwb", "dwb_social"]
    assert out["summary"]["n_references"] == 2
    assert "dwb_social: 4 episodes" in out["summary"]["summary"]
    assert pathlib.Path(out["manifest"]["path"]).name == "my_report.yaml"
    assert out["run_manifests"]["run_manifests"] == ["my_report"]
    assert yaml.safe_load(out["custom_template"]["content"])["name"] == "my_report"
    assert "no run 'nope'" in out["missing_run"]
    assert "evaluation_process computes it" in out["unprocessed"]
    assert "no metric column zz" in out["bad_column"]
    assert out[f"arena://evaluation/runs/{RUN_ID}/state"]["lanes"] == ["p2"]
    assert out[f"arena://evaluation/runs/{RUN_ID}/notes"][0]["label"] == "a"
    assert out[f"arena://evaluation/runs/{RUN_ID}/report"]["report"] is None
    assert out[f"arena://evaluation/runs/{RUN_ID}/metrics/schema"]["dtypes"]["planner"] == "String"
    assert out[f"arena://evaluation/runs/{RUN_ID}/metrics/summary"]["planners"] == ["dwb", "dwb_social"]
    assert out["arena://evaluation/runs"]["runs"][0]["run_id"] == RUN_ID


@sourced
def test_jobs_surface_failures_and_process_control_stays_on_arena_processes(tmp_path):
    build_runs(tmp_path / "data" / "benchmarks")
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "world_generator"], start_new_session=True)
    runner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "evaluation", "benchmark", "--run-id", RUN_ID], start_new_session=True)
    bad_contest = "[{name: bad, mobile: {driver: nav2, local_planner: nope_planner}}]"
    inline_suite = {"stages": [{"name": "s1", "map": "map_empty", "robot": "jackal", "episodes": 1, "tm_robots": "random", "tm_obstacles": "random"}]}

    async def body(client):
        out = {}
        out["process"] = payload(await client.call_tool("evaluation_process", {"run_ids": [EMPTY_RUN_ID], "workers": 1}))
        out["process_done"] = payload(await client.call_tool("jobs_wait", {"job_id": out["process"]["job"]["id"], "timeout_s": 240}))
        started = time.monotonic()
        out["config_error"] = error_text(await client.call_tool("evaluation_benchmark", {"suite": inline_suite, "contest": bad_contest, "run_id": "bad-config", "confirm_s": 200}))
        out["config_error_s"] = time.monotonic() - started
        out["exists"] = error_text(await client.call_tool("evaluation_benchmark", {"suite": "basic", "contest": "basic", "run_id": RUN_ID, "confirm_s": 200}))
        out["no_resume"] = error_text(await client.call_tool("evaluation_benchmark", {"resume": True, "run_id": "nope"}))
        out["preflight_args"] = error_text(await client.call_tool("evaluation_preflight", {}))
        out["kill_nothing"] = error_text(await client.call_tool("evaluation_kill", {}))
        out["ps"] = payload(await client.call_tool("evaluation_ps", {}))
        out["kill"] = payload(await client.call_tool("evaluation_kill", {"pids": [stranger.pid, worker.pid], "grace_s": 5}))
        out["stranger_alive"] = stranger.poll() is None
        out["stop"] = payload(await client.call_tool("evaluation_stop", {"run_id": RUN_ID, "grace_s": 5}))
        out["stop_again"] = error_text(await client.call_tool("evaluation_stop", {"run_id": RUN_ID}))
        return out

    try:
        out = run_session(body, server_env(tmp_path))
    finally:
        for child in (stranger, worker, runner):
            if child.poll() is None:
                child.kill()
            child.wait()
    assert out["process_done"]["state"] == "failed"
    assert out["process_done"]["exit_code"] == 1
    assert "nope_planner" in out["config_error"]
    assert "before it prepared the run" in out["config_error"]
    assert "File exists" in out["exists"] and "before it prepared the run" in out["exists"]
    assert "no run 'nope' to resume" in out["no_resume"]
    assert "not both or neither" in out["preflight_args"]
    assert "nothing is stopped by default" in out["kill_nothing"]
    assert {worker.pid, runner.pid} <= {row["pid"] for row in out["ps"]["processes"]}
    killed = {row["pid"]: row for row in out["kill"]["results"]}
    assert killed[stranger.pid]["status"] == "refused: not an arena process"
    assert killed[worker.pid]["kind"] == "world_generator" and killed[worker.pid]["status"] == "stopped"
    assert out["stranger_alive"]
    assert out["stop"]["stopped"][0]["outcome"].startswith("stopped")
    assert "no runner" in out["stop_again"]


@sourced
def test_pipeline_verbs_run_the_evaluation_cli_as_jobs(tmp_path):
    build_runs(tmp_path / "data" / "benchmarks")

    async def body(client):
        out = {}
        for key, name, args in (
            ("acoustic", "evaluation_acoustic", {"run_id": RUN_ID, "action": "list"}),
            ("extract", "evaluation_extract", {"run_ids": [EMPTY_RUN_ID], "workers": 1}),
            ("plot", "evaluation_plot", {"run_ids": [RUN_ID], "report_manifest": "standard"}),
            ("report", "evaluation_report", {"run_ids": [RUN_ID], "report_manifest": "standard", "output_dir": str(tmp_path / "out")}),
        ):
            started = payload(await client.call_tool(name, args))
            out[key] = {"command": started["command"], "done": payload(await client.call_tool("jobs_wait", {"job_id": started["job"]["id"], "timeout_s": 280})), "log": (await client.call_tool("jobs_log", {"job_id": started["job"]["id"], "lines": 30})).content[0].text}
        out["bad_option"] = error_text(await client.call_tool("evaluation_acoustic", {"run_id": RUN_ID, "action": "list", "fps": 5}))
        out["both_inputs"] = error_text(await client.call_tool("evaluation_process", {"run_ids": [RUN_ID], "episode_dirs": [f"{RUN_ID}/episodes/episode_000"]}))
        out["not_episode"] = error_text(await client.call_tool("evaluation_process", {"episode_dirs": [RUN_ID]}))
        return out

    out = run_session(body, server_env(tmp_path))
    assert all(out[key]["done"]["state"] == "succeeded" for key in ("acoustic", "extract", "plot", "report")), {key: out[key]["log"] for key in out if isinstance(out[key], dict)}
    assert out["acoustic"]["command"].startswith("arena evaluation acoustic list --benchmark-dir ")
    assert "No episodes found" in out["extract"]["log"]
    assert any(path.endswith("bar_success_local.png") for path in out["plot"]["done"]["result"]["plots"])
    assert out["report"]["done"]["result"]["report"] == str(tmp_path / "out" / "report.html")
    assert "list takes no fps" in out["bad_option"]
    assert "not both" in out["both_inputs"]
    assert "is not an episode_NNN directory" in out["not_episode"]
