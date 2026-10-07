from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.providers import Http

from arena_evaluation.adapters import barn
from arena_evaluation.adapters.barn import BARN

FIXTURE = Path(__file__).parent / "fixtures" / "barn"
UPSTREAM = FIXTURE / "jackal_helper" / "worlds" / "BARN"
ARENA_CLI = Path(__file__).parents[3] / "_meta"
NATIVE_STAGE_KEYS = {"name", "map", "robot", "tm_robots", "tm_obstacles", "episodes", "timeout", "config"}

_RUNNER = """
import enum
import json
import attrs
from arena_evaluation.benchmark import runner
from arena_evaluation.benchmark.state import Manifest, compute_config_hash
from arena_evaluation.benchmark.tree import ContestIdentifier, SuiteIdentifier

verdicts = [identifier.resolve_source_sync() for identifier in (SuiteIdentifier(name="barn"), ContestIdentifier(name="barn"))]
suite, contest, suite_dict, contest_dict, bundle_dir, suite_provenance, contest_provenance = runner._load_suite_contest("barn", "barn")
steps = runner._all_steps(contest, suite, 1.0)
stage = steps[0].stage
obstacles, robots = runner._flatten_per_mode_params(stage.config, tm_obstacles=stage.tm_obstacles.value, tm_robots=stage.tm_robots.value)
records = [{"key": step.key, "contestant": attrs.asdict(step.contestant), "stage": {k: v.value if isinstance(v, enum.Enum) else v for k, v in step.stage._asdict().items()}} for step in steps]
manifest = Manifest(run_id="barn", created_at="", arena_git_sha=None, arena_git_dirty=False, cli_args=[], env_n=1, headless=True, config_hash=compute_config_hash(suite_dict, contest_dict), simulator="gazebo", scale_episodes=1.0, suite_name=suite.name, contest_name=contest.name, suite=suite_dict, contest=contest_dict, steps=records, suite_provenance=suite_provenance, contest_provenance=contest_provenance)
assert Manifest.from_yaml(manifest.to_yaml()) == manifest
print(json.dumps({
    "resolvers": [repr(verdict.resolver) for verdict in verdicts],
    "verdicts": [verdict.verdict.name for verdict in verdicts],
    "bundle_dir": str(bundle_dir),
    "keys": [step.key for step in steps],
    "launch": runner.build_launch_args(steps[0], "gazebo"),
    "preflight": runner._preflight_contest(contest),
    "budget": runner.BenchmarkRunner._episode_budget(steps[0]),
    "obstacles": [[p.name, p.value.string_value] for p in obstacles],
    "robots": [[p.name, p.value.string_value] for p in robots],
    "manifest": manifest.to_yaml(),
}))
"""

_REPORT = """
import json
import pathlib
import sys
from arena_evaluation.presentation.manifest_registry import available_manifests, find_manifest_file, resolve_manifest
from arena_evaluation.presentation.report_builder import ReportBuilder

run = pathlib.Path(sys.argv[1])
manifest = resolve_manifest("barn", benchmark_dir=run)
ReportBuilder.from_dirs([run], output_dir=run, manifest=manifest).build()
print(json.dumps({"available": available_manifests(), "file": str(find_manifest_file("barn")), "name": manifest.name}))
"""

_RUNTIME = """
import json
from arena_simulation_setup.tree.World import WorldIdentifier

verdict = WorldIdentifier("barn_0").resolve_source_sync()
view = WorldIdentifier("barn_0").resolve_sync()
world = view.load()
robot = view.scenario("barn").resolve_sync().load().robots[0]
goal = robot.phase_list()[0].pose
print(json.dumps({
    "path": str(verdict.path),
    "levels": list(world.levels),
    "walls": sum(len(list(level.all_walls)) for level in world.levels.values()),
    "start": [robot.start.position.x, robot.start.position.y],
    "goal": [goal.position.x, goal.position.y],
}))
"""

_PUSH = """
import json
from arena_cli import asset
from arena_evaluation.benchmark.tree import SuiteIdentifier

identifier = SuiteIdentifier(name="barn")
source = asset._config_source(identifier)
print(json.dumps({"source": str(source), "dangling": asset._suite_dangling(identifier.resolve_sync(), source)}))
"""


def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _challenge_path_length(path: np.ndarray) -> float:
    init = (-2.25, 3.0)
    goal = (init[0] + 0, init[1] + 10)
    radius = 0.075
    r_shift = -radius - (30 * radius * 2)
    c_shift = radius + 5
    points = [(x * radius * 2 + r_shift, y * radius * 2 + c_shift) for x, y in path]
    points = [init, *points, goal]
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True))


def _subprocess(script: str, *args: str, **env: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, env={**os.environ, **env}, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.splitlines()[-1])


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _ros_gate()
    root = tmp_path_factory.mktemp("assets")
    importer = BARN(cache=root / ".import" / "barn", provider=Http(FIXTURE.as_uri()), worlds=1)
    assert importer.import_item("barn", root / "_imported") == root / "_imported" / "suites" / "barn"
    return root


@pytest.fixture(scope="module")
def out(assets: Path) -> Path:
    return assets / "_imported"


@pytest.fixture(scope="module")
def suite(out: Path) -> dict:
    return yaml.safe_load((out / "suites" / "barn" / "suite.yaml").read_text())


def _files(root: Path) -> list[str]:
    return sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file())


def test_registry_loads_this_class():
    assert IMPORTERS.load("suite", "barn") is BARN
    assert BARN.meta.kind == "suite"
    assert list(BARN(cache=Path("/nonexistent")).items()) == ["barn"]
    assert BARN(cache=Path("/nonexistent")).worlds == 300
    assert BARN.default_provider().url("jackal_helper/worlds/BARN/world_0.world") == "https://raw.githubusercontent.com/Daffan/the-barn-challenge/main/jackal_helper/worlds/BARN/world_0.world"


def test_import_fetches_into_the_cache_and_writes_the_bucket_bundle_layout(assets: Path, out: Path):
    assert _files(assets / ".import" / "barn") == ["path_files/path_0.npy", "worlds/world_0.world"]
    assert _files(out / "suites" / "barn") == [
        ATTRIBUTION_FILE,
        "suite.yaml",
        "worlds/barn_0/kg.yaml",
        "worlds/barn_0/map/0.png",
        "worlds/barn_0/map/0.yaml",
        "worlds/barn_0/scenarios/barn/scenario.yaml",
        "worlds/barn_0/world.yaml",
    ]
    assert _files(out / "contests" / "barn") == [ATTRIBUTION_FILE, "contest.yaml"]
    assert _files(out / "manifests" / "barn") == [ATTRIBUTION_FILE, "manifest.yaml"]
    for bundle in ("suites", "contests", "manifests"):
        attribution = read_attribution(out / bundle / "barn" / ATTRIBUTION_FILE)
        assert attribution.license == "MIT"
        assert "2008.13315" in attribution.citation


def test_stage_carries_the_native_stage_keys(suite: dict):
    assert set(suite) == {"metrics", "stages"}
    assert suite["metrics"]["max_collisions"] == 1
    assert len(suite["stages"]) == 1
    stage = suite["stages"][0]
    assert set(stage) == NATIVE_STAGE_KEYS
    assert stage == {
        "name": "barn_0",
        "map": "barn_0",
        "robot": "jackal",
        "episodes": 1,
        "tm_robots": "scenario",
        "tm_obstacles": "scenario",
        "config": {"scenario": {"file": "barn"}},
        "timeout": "110s",
    }


def test_bundled_scenario_holds_challenge_start_and_goal(out: Path):
    scenario = yaml.safe_load((out / "suites" / "barn" / "worlds" / "barn_0" / "scenarios" / "barn" / "scenario.yaml").read_text())
    assert scenario.keys() == {"format", "robots"}
    robot = scenario["robots"][0]
    assert robot["start"][:2] == [2.25, 3.0]
    assert robot["start"][2] == pytest.approx(math.pi / 2, abs=1e-3)
    assert robot["phases"][0]["goto"][:2] == [2.25, 13.0]
    assert robot["phases"][0]["goto"][1] - robot["start"][1] == pytest.approx(10.0)


def test_contest_pins_the_challenge_episode_rules(out: Path):
    from task_generator.constants.runtime import EPISODE_PARAMS

    contest = yaml.safe_load((out / "contests" / "barn" / "contest.yaml").read_text())
    assert contest == [
        {
            "name": "dwb",
            "mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"},
            "task.episode.fail_on_collision": True,
            "task.episode.goto_pose.tolerance.radius": 1.0,
            "task.episode.goto_pose.tolerance.angle": "0.0",
        }
    ]
    assert {key for key in contest[0] if key.startswith("task.")} <= set(EPISODE_PARAMS)


def test_reference_block_matches_challenge_formula(suite: dict):
    reference = suite["metrics"]["barn"]
    path = np.load(UPSTREAM / "path_files" / "path_0.npy")
    expected = _challenge_path_length(path)
    assert reference["path_length"] == {"barn_0": pytest.approx(expected, abs=1e-9)}
    assert reference["start"] == [2.25, 3.0]
    assert reference["goal"] == [2.25, 13.0]
    assert expected > 10.0


def test_stages_follow_numeric_index_order(tmp_path: Path):
    _ros_gate()
    src = tmp_path / "src"
    for index in (10, 0, 2):
        for folder, name, source in (("path_files", f"path_{index}.npy", UPSTREAM / "path_files" / "path_0.npy"), ("worlds", f"world_{index}.world", UPSTREAM / "world_0.world")):
            (src / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy(source, src / folder / name)
    out = tmp_path / "barn"
    assert BARN(cache=tmp_path / "cache").convert("barn", src, out) == "suites/barn"
    suite = yaml.safe_load((out / "suites" / "barn" / "suite.yaml").read_text())
    assert [stage["name"] for stage in suite["stages"]] == ["barn_0", "barn_2", "barn_10"]
    assert [stage["map"] for stage in suite["stages"]] == ["barn_0", "barn_2", "barn_10"]
    assert list(suite["metrics"]["barn"]["path_length"]) == ["barn_0", "barn_2", "barn_10"]
    assert sorted(world.name for world in (out / "suites" / "barn" / "worlds").iterdir()) == ["barn_0", "barn_10", "barn_2"]


def test_convert_replaces_a_previous_bundle(tmp_path: Path):
    _ros_gate()
    src = tmp_path / "src"
    shutil.copytree(UPSTREAM / "path_files", src / "path_files")
    (src / "worlds").mkdir()
    shutil.copy(UPSTREAM / "world_0.world", src / "worlds" / "world_0.world")
    stale = tmp_path / "barn" / "suites" / "barn" / "worlds" / "barn_7"
    stale.mkdir(parents=True)
    BARN(cache=tmp_path / "cache").convert("barn", src, tmp_path / "barn")
    assert not stale.exists()
    assert (tmp_path / "barn" / "suites" / "barn" / "worlds" / "barn_0" / "world.yaml").is_file()


def test_convert_rejects_unknown_item(tmp_path: Path):
    with pytest.raises(KeyError):
        BARN(cache=tmp_path / "cache").convert("dynabarn", FIXTURE, tmp_path / "out")


def test_convert_rejects_empty_source(tmp_path: Path):
    _ros_gate()
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        BARN(cache=tmp_path / "cache").convert("barn", tmp_path / "empty", tmp_path / "out")


def test_runner_loads_the_import_through_the_registered_resolvers(assets: Path, out: Path, tmp_path: Path):
    from task_generator.constants.runtime import EPISODE_PARAMS

    from arena_evaluation.processing.pipeline import _suite_metrics

    seen = _subprocess(_RUNNER, ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out))
    assert all("ConfigPathResolver" in resolver and str(out) in resolver for resolver in seen["resolvers"])
    assert seen["verdicts"] == ["HIT", "HIT"]
    assert seen["bundle_dir"] == str(out / "suites" / "barn")
    assert seen["keys"] == ["dwb/barn_0"]
    assert seen["preflight"] == []
    assert seen["budget"] == 110.0
    assert seen["obstacles"] == [["file", "barn"]]
    assert seen["robots"] == [["file", "barn"]]
    launch = dict(arg.split(":=", 1) for arg in seen["launch"])
    assert launch["world"] == "barn_0"
    assert launch["robot"] == "jackal"
    assert launch["task.robots"] == "scenario"
    assert launch["task.obstacles"] == "scenario"
    assert launch["robot.mobile"] == "nav2"
    assert launch["robot.mobile.local_planner"] == "dwb"
    assert launch["robot.mobile.inter_planner"] == "navigate_w_replanning_time"
    episode = {key: yaml.safe_load(value) for key, value in launch.items() if key in EPISODE_PARAMS}
    assert episode == {
        "task.episode.auto_reset": False,
        "task.episode.fail_on_collision": True,
        "task.episode.goto_pose.tolerance.radius": 1.0,
        "task.episode.goto_pose.tolerance.angle": 0.0,
    }
    consumed = {"name", "mobile", *episode}
    assert set(barn.CONTESTANT) <= consumed
    manifest = yaml.safe_load(seen["manifest"])
    assert manifest["steps"][0]["stage"]["map"] == "barn_0"
    assert "generator" not in manifest["steps"][0]["stage"]
    episode_dir = tmp_path / "barn" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    (tmp_path / "barn" / "manifest.yaml").write_text(seen["manifest"])
    assert _suite_metrics(episode_dir, tmp_path) == {"max_collisions": 1, "barn": manifest["suite"]["metrics"]["barn"]}


def test_report_manifest_loads_through_the_manifest_identifier(out: Path):
    from arena_evaluation.benchmark.tree import ManifestIdentifier
    from arena_evaluation.processing.metrics.performance.barn_metrics import BARNScoreCalculator

    manifest = ManifestIdentifier(name="barn").load(out / "manifests" / "barn")
    assert manifest.name == "barn"
    assert manifest.data_source == "metrics"
    assert manifest.summary_group_by == ["planner"]
    assert [column.label for column in manifest.summary] == ["Avg Metric", "Avg Success", "Avg Collision", "Avg Timeout"]
    assert {plot.type for plot in manifest.plots} == {"table"}
    assert {plot.layout_group for plot in manifest.plots} == {group.id for group in manifest.groups}
    per_world = next(plot for plot in manifest.plots if plot.id == "barn_score_per_world")
    assert per_world.options["group_by"] == ["planner", "map"]
    native = {"success", "collision_amount", "time_to_goal"}
    columns = {column.metric for column in manifest.summary} | {column["metric"] for plot in manifest.plots for column in plot.options["columns"]}
    assert columns - native == set(BARNScoreCalculator.output_keys())
    assert yaml.safe_load((out / "manifests" / "barn" / "manifest.yaml").read_text()) == barn.MANIFEST


def test_report_renders_the_score_per_contestant_and_per_world(assets: Path, out: Path, tmp_path: Path):
    import polars as pl

    run = tmp_path / "20261007-000000-barn-barn"
    run.mkdir()
    pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "map": ["barn_0", "barn_1", "barn_0", "barn_1"],
            "stage": ["barn_0", "barn_1", "barn_0", "barn_1"],
            "episode": [0, 1, 2, 3],
            "barn_score": [0.25, 0.5, 0.0, 0.125],
            "barn_optimal_time": [6.8, 6.2, 6.8, 6.2],
            "barn_time": [27.2, 12.4, 3.1, 80.0],
            "barn_success": [True, True, False, True],
            "barn_collided": [False, False, True, False],
            "barn_timeout": [False, False, False, False],
            "success": [True, True, False, True],
            "collision_amount": [0, 0, 1, 0],
            "time_to_goal": [28.0, 13.2, 3.4, 81.2],
        }
    ).write_parquet(run / "combined_metrics.parquet")

    seen = _subprocess(_REPORT, str(run), ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out))
    assert seen["name"] == "barn"
    assert seen["file"] == str(out / "manifests" / "barn" / "manifest.yaml")
    assert {"barn", "standard"} <= set(seen["available"])
    assert yaml.safe_load((run / "report_manifest.yaml").read_text())["name"] == "barn"

    content = (run / "report.html").read_text()
    assert "BARN Challenge Report" in content
    rows = [re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S) for row in re.findall(r"<tr[^>]*>(.*?)</tr>", content, re.S)]
    assert ["planner", "Avg Metric", "Avg Success", "Avg Collision", "Avg Timeout"] in rows
    assert ["dwb", "0.3750", "1.0000", "0.0000", "0.0000"] in rows
    assert ["teb", "0.0625", "0.5000", "0.5000", "0.0000"] in rows
    assert ["Planner", "Avg Time (s)"] in rows
    assert ["dwb", "19.8000"] in rows
    assert ["teb", "80.0000"] in rows
    assert ["dwb", "100%", "0.00", "20.60"] in rows
    assert ["teb", "50%", "0.50", "42.30"] in rows
    assert ["Planner", "Map", "Metric", "Time (s)", "Optimal Time (s)", "Success", "Collision", "Timeout"] in rows
    assert ["dwb", "barn_0", "0.2500", "27.20", "6.80", "1.00", "0.00", "0.00"] in rows
    assert ["dwb", "barn_1", "0.5000", "12.40", "6.20", "1.00", "0.00", "0.00"] in rows
    assert ["teb", "barn_0", "0.0000", "3.10", "6.80", "0.00", "1.00", "0.00"] in rows
    assert ["teb", "barn_1", "0.1250", "80.00", "6.20", "1.00", "0.00", "0.00"] in rows


def test_runtime_resolves_the_bundled_world_from_the_exported_world_path(out: Path):
    worlds = out / "suites" / "barn" / "worlds"
    seen = _subprocess(_RUNTIME, ARENA_WORLD_PATH=str(worlds))
    assert seen["path"] == str(worlds / "barn_0")
    assert seen["levels"] == ["0"]
    assert seen["walls"] == 4 + 8 * 209
    assert seen["start"] == pytest.approx([2.25, 3.0])
    assert seen["goal"] == pytest.approx([2.25, 13.0])


def test_push_preflight_finds_no_dangling_world(assets: Path, out: Path):
    if not (ARENA_CLI / "arena_cli" / "asset.py").is_file():
        pytest.skip("arena_cli is not checked out beside this package")
    path = os.pathsep.join(filter(None, [str(ARENA_CLI), os.environ.get("PYTHONPATH")]))
    seen = _subprocess(_PUSH, ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out), PYTHONPATH=path)
    assert seen == {"source": str(out / "suites" / "barn"), "dangling": []}


def test_written_suite_and_contest_parse(out: Path):
    from task_generator.constants import Constants

    from arena_evaluation.benchmark.config import Contest, Suite

    suite = Suite.parse("barn", yaml.safe_load((out / "suites" / "barn" / "suite.yaml").read_text()))
    assert len(suite.stages) == 1
    stage = suite.stages[0]
    assert stage.map == "barn_0"
    assert stage.tm_robots is Constants.TaskMode.TM_Robots.SCENARIO
    assert stage.tm_obstacles is Constants.TaskMode.TM_Obstacles.SCENARIO
    assert stage.timeout == 110.0
    assert suite.metrics["max_collisions"] == 1
    assert suite.metrics["barn"]["goal"] == [2.25, 13.0]
    assert suite.launch_args == {}
    assert suite.references is False

    contest = Contest.parse("barn", yaml.safe_load((out / "contests" / "barn" / "contest.yaml").read_text()))
    assert [c.name for c in contest.contestants] == ["dwb"]
    assert contest.contestants[0].args["mobile"]["driver"] == "nav2"


def test_scenario_parses_as_robot_goal(out: Path):
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    scenario = yaml.safe_load((out / "suites" / "barn" / "worlds" / "barn_0" / "scenarios" / "barn" / "scenario.yaml").read_text())
    goal = RobotGoal.parse(scenario["robots"][0])
    phases = goal.phase_list()
    assert isinstance(phases[0], GoToPhase)
    assert goal.start.position.x == pytest.approx(2.25)
    assert phases[0].pose.position.y - goal.start.position.y == pytest.approx(10.0)


@pytest.mark.network
def test_fetch_downloads_one_world_and_its_path(tmp_path: Path):
    importer = BARN(cache=tmp_path / "cache", worlds=1)
    src = importer.fetch("barn")
    assert src == tmp_path / "cache"
    assert _files(src) == ["path_files/path_0.npy", "worlds/world_0.world"]
    path = np.load(src / "path_files" / "path_0.npy")
    assert path.shape[1] == 2
    assert (src / "worlds" / "world_0.world").read_bytes() == (UPSTREAM / "world_0.world").read_bytes()
    assert src == importer.fetch("barn")
