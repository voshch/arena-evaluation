from __future__ import annotations

from pathlib import Path

import yaml

from arena_evaluation.benchmark.config import Suite
from arena_evaluation.benchmark.runner import _task_params
from arena_evaluation.generation.cli import main
from arena_evaluation.generation.scenario.materialize import GENERATED_MARKER
from arena_evaluation.generation.suite import build, scenarios


def _scenario(world: Path, name: str, *, generated: bool) -> None:
    (world / "scenarios" / name).mkdir(parents=True)
    (world / "scenarios" / name / "scenario.yaml").write_text((GENERATED_MARKER + "\n" if generated else "") + "robots: []\n")


def _world(tmp_path: Path) -> Path:
    world = tmp_path / "worlds" / "toy"
    for name in ("normal", "normal__u_001", "break__u_002"):
        _scenario(world, name, generated=True)
    for name in ("default", "edge_headon"):
        _scenario(world, name, generated=False)
    return world


def test_scenarios_default_to_what_the_generator_wrote(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert scenarios(world) == ["break__u_002", "normal", "normal__u_001"]
    assert scenarios(world, ["edge_*", "*__u_001"]) == ["edge_headon", "normal__u_001"]


def test_suite_parses_and_routes_the_scenario_file(tmp_path: Path) -> None:
    doc = build({"toy": ["normal__u_001", "normal"]}, robot="mpo700", episodes=3, seed=2000, timeout="45s")
    suite = Suite.parse("edge_case", doc)
    assert [s.name for s in suite.stages] == ["toy.normal__u_001", "toy.normal"]
    stage = suite.stages[0]
    assert (stage.map, stage.robot, stage.episodes, stage.seed, stage.timeout) == ("toy", "mpo700", 3, 2000, 45.0)
    assert stage.tm_robots.value == stage.tm_obstacles.value == "edge_case"
    assert suite.launch_args == {"human": "arena"}
    params = _task_params(stage.config, tm_obstacles=stage.tm_obstacles.value, tm_robots=stage.tm_robots.value)
    assert [(p.name, p.value.string_value) for p in params] == [("task.scenario.file", "normal__u_001")]


def test_cli_writes_the_suite(tmp_path: Path) -> None:
    _world(tmp_path)
    out = tmp_path / "s.yaml"
    assert main(["suite", "toy", "--worlds", str(tmp_path / "worlds"), "--out", str(out), "--episodes", "2"]) == 0
    stages = yaml.safe_load(out.read_text())["stages"]
    assert [s["config"]["scenario"]["file"] for s in stages] == ["break__u_002", "normal", "normal__u_001"]
    assert main(["suite", "toy", "--worlds", str(tmp_path / "worlds"), "--out", str(out), "--match", "nope*"]) == 1
