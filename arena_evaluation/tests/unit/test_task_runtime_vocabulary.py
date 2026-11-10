"""A suite parses, checks and builds its launch and queue requests against the selected task runtime, with task_generator not importable at all."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rclpy")

RUNTIME = """
CALLS = []


def mode_namespaces():
    return {
        "robots": {"patrol": "runtime.patrol", "idle": "runtime.idle"},
        "obstacles": {"Clutter": "runtime.clutter"},
        "fleet": {"convoy": "runtime.convoy", "wedge": "runtime.wedge"},
    }


def validate_modes(modes):
    vocabulary = {axis: list(known) for axis, known in mode_namespaces().items()}
    checked = {}
    for axis, names in modes.items():
        if axis not in vocabulary:
            raise ValueError(f"unknown axis {axis!r}, known: {sorted(vocabulary)}")
        canonical = []
        for name in names:
            match = next((known for known in vocabulary[axis] if known.lower() == name.lower()), None)
            if match is None:
                raise ValueError(f"unknown {axis} mode {name!r}, known: {vocabulary[axis]}")
            canonical.append(match)
        if axis != "fleet" and len(canonical) != 1:
            raise ValueError(f"the {axis} axis takes exactly one mode, got {canonical}")
        checked[axis] = canonical
    return checked


def declare_mode_schema(node, axis, mode):
    from arena_rclpy_mixins.params import Param, declare

    CALLS.append([axis, mode])
    if mode == "patrol":
        declare(node, Param[float]("runtime.patrol.speed", 1.0, "Patrol speed in m/s."))
    if mode == "convoy":
        declare(node, Param[float]("runtime.convoy.gap", 1.0, "Convoy gap in m."))
"""

DRIVER = """
import importlib.abc
import json
import sys


class NoTaskGenerator(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "task_generator":
            raise ModuleNotFoundError(name)


sys.meta_path.insert(0, NoTaskGenerator())

from arena_evaluation.benchmark.config import Contest, Suite
from arena_evaluation.benchmark.runner import build_launch_args, stage_request
from arena_evaluation.benchmark.step import Step
from arena_evaluation.mcp import authoring
from arena_runtime.env.task_layer import decode_modes, task_layer

extra = json.loads(sys.argv[1])
stage = {"name": "s", "episodes": 1, "robot": "jackal", "map": "map_empty", "modes": {"robots": "PATROL", "obstacles": "clutter", "fleet": "Convoy"}, "config": {"patrol": {"speed": 2.0, "bogus": 1}, "convoy": {"gap": 3.0}, "wedge": {"gap": 9.0}}}
suite = Suite.parse("t", {"stages": [stage], **extra})
legacy_keys = {key: value for key, value in stage.items() if key != "modes"} | {"tm_robots": "PATROL", "tm_obstacles": "clutter", "tm_fleet": "Convoy"}
layer = task_layer(suite.task_layer)


def refusal(modes):
    try:
        Suite.Stage.parse({**stage, "modes": modes}, layer)
    except ValueError as exc:
        return str(exc)


def verdict_problems(modes):
    return authoring.check_suite({"stages": [{**stage, "modes": modes}], **extra})["problems"]
step = Step(contestant=Contest.Contestant(name="c"), stage=suite.stages[0], task_layer=suite.task_layer)
request = stage_request(step)
verdict = authoring.check_suite({"stages": [stage], **extra})
runtime = sys.modules["generated_runtime"]
print("RESULT " + json.dumps({
    "task_layer": suite.task_layer,
    "modes": suite.stages[0].modes,
    "legacy_same": Suite.Stage.parse(legacy_keys, layer) == suite.stages[0],
    "unknown": refusal({"robots": "random"}),
    "unknown_axis": refusal({"no_such_axis": "random"}),
    "empty": refusal({"robots": ""}),
    "problems_unknown": verdict_problems({"robots": "random"}),
    "problems_empty": verdict_problems({"obstacles": ""}),
    "launch": build_launch_args(step, "gazebo"),
    "queued_modes": decode_modes(request.modes),
    "queued_params": sorted(p.name for p in request.params),
    "valid": verdict["valid"],
    "warnings": verdict["warnings"],
    "declared": runtime.CALLS,
    "task_generator_imported": any(name.split(".")[0] == "task_generator" for name in sys.modules),
}))
"""

UNKNOWN_DRIVER = """
import json
import sys

from arena_evaluation.mcp import authoring

stage = {"name": "s", "episodes": 1, "robot": "jackal", "map": "map_empty", "modes": {}}
print("RESULT " + json.dumps(authoring.check_suite({"stages": [stage], "task_layer": "no_such_runtime"})))
"""

MISSING_AXIS_DRIVER = """
import json

from arena_evaluation.mcp import authoring

stage = {"name": "s", "episodes": 1, "robot": "jackal", "map": "map_empty", "modes": {"robots": "patrol"}}
print("RESULT " + json.dumps(authoring.check_suite({"stages": [stage]})))
"""


def _run(driver: str, tmp_path: Path, *argv: str, **env: str) -> dict:
    from arena_runtime.testing.task_layer import register_task_layer

    register_task_layer(tmp_path, "generated_runtime", RUNTIME)
    environ = {key: value for key, value in os.environ.items() if key != "ARENA_TASK_LAYER"}
    environ.update(env, PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    done = subprocess.run([sys.executable, "-c", driver, *argv], env=environ, capture_output=True, text=True, timeout=120, check=False)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, done.stderr[-2000:]
    return json.loads(lines[0].removeprefix("RESULT "))


def _assert_runtime_used(out: dict) -> None:
    assert out["modes"] == {"robots": "patrol", "obstacles": "Clutter", "fleet": "convoy"}
    assert out["legacy_same"]
    assert out["unknown"] == "unknown robots mode 'random', known: ['patrol', 'idle']"
    assert out["unknown_axis"] == "unknown axis 'no_such_axis', known: ['fleet', 'obstacles', 'robots']"
    assert out["empty"] == "the robots axis takes exactly one mode, got []"
    assert out["problems_unknown"] == ["ValueError: " + out["unknown"]]
    assert out["problems_empty"] == ["ValueError: the obstacles axis takes exactly one mode, got []"]
    assert out["queued_modes"] == {"robots": ["patrol"], "obstacles": ["Clutter"], "fleet": ["convoy"]}
    assert out["queued_params"] == ["runtime.convoy.gap", "runtime.patrol.bogus", "runtime.patrol.speed"]
    assert {"task.robots:=patrol", "task.obstacles:=Clutter", "task.fleet:=convoy"} <= set(out["launch"])
    assert not any(arg.startswith("task.modules") for arg in out["launch"])
    assert any("config.patrol.bogus" in warning for warning in out["warnings"])
    assert out["declared"] == [["robots", "patrol"], ["obstacles", "Clutter"], ["fleet", "convoy"]]
    assert not out["task_generator_imported"]


def test_a_suite_uses_the_vocabulary_of_the_runtime_the_environment_selects(tmp_path: Path) -> None:
    out = _run(DRIVER, tmp_path, "{}", ARENA_TASK_LAYER="generated_runtime")
    _assert_runtime_used(out)
    assert out["task_layer"] == ""
    assert "task_layer:=generated_runtime" in out["launch"]


def test_a_suite_names_its_own_runtime_for_parsing_checking_and_launching(tmp_path: Path) -> None:
    out = _run(DRIVER, tmp_path, json.dumps({"task_layer": "generated_runtime"}))
    _assert_runtime_used(out)
    assert out["task_layer"] == "generated_runtime"
    assert "task_layer:=generated_runtime" in out["launch"]


def test_an_unknown_task_layer_is_a_problem_of_the_verdict(tmp_path: Path) -> None:
    out = _run(UNKNOWN_DRIVER, tmp_path)
    assert out["valid"] is False
    assert any("no_such_runtime" in problem for problem in out["problems"])


def test_a_stage_leaving_out_axes_passes_the_check(tmp_path: Path) -> None:
    out = _run(MISSING_AXIS_DRIVER, tmp_path, ARENA_TASK_LAYER="generated_runtime")
    assert out["valid"] is True
    assert out["problems"] == []
