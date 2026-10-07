from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution

from arena_evaluation.adapters import socnavbench
from arena_evaluation.adapters.socnavbench import SocNavBenchSuite

ARENA_CLI = Path(__file__).parents[3] / "_meta"
NATIVE_STAGE_KEYS = {"name", "map", "robot", "tm_robots", "tm_obstacles", "episodes", "timeout", "config"}
REVISION = "2724ea85ff22ca61a88ee95285dfc3fa056656c6"
SCENES = {"eth": "eth", "zara01": "zara", "zara02": "zara"}
PLACEMENTS = {
    "eth": ("eth/world_coordinate_inter_eth.csv", 15, [1.4, 14.4, 0], True, True, False),
    "hotel": ("hotel/world_coordinate_inter_hotel.csv", 15, [12, 4.5, 0], True, False, False),
    "zara01": ("zara01/world_coordinate_inter.csv", 25, [9.5, 3.0, 0], False, True, True),
    "zara02": ("zara02/world_coordinate_inter.csv", 25, [9.5, 1.85, 0], False, True, True),
    "univ": ("univ/world_coordinate_inter.csv", 15, [10, 6, 0], False, True, True),
}
EPISODES = [
    ("t_ETH1", "eth", 0, [[8, 5, 1.5], [12, 15, 1.5]], 20),
    ("t_ETH2", "eth", 0, [[12, 15, 1.5], [6, 2, -1.5]], 60),
    ("t_eth_follow", "eth", 238, [[1.5, 15.5, 0], [9.5, 1, -1.5]], 60),
    ("t_eth_against", "eth", 238, [[9.5, 1, 1.5], [1.5, 15.5, 3]], 60),
    ("t_eth_dense_flow", "eth", 250, [[1.5, 15.5, 0], [9.5, 1, -1.5]], 60),
    ("t_eth_dense_against", "eth", 250, [[5, 1, 1.5], [12.5, 15.5, 0]], 60),
    ("t_zara1_basic_cross", "zara01", 0, [[8.0, 8, 0], [15, 3, -0.2]], 60),
    ("t_zara1_wrap", "zara01", 0, [[2, 2, 1.8], [17.5, 13, 0]], 50),
    ("t_zara1_dense", "zara01", 78, [[2, 5, 0], [19.5, 5, 0]], 60),
    ("t_zara1_dense_rev", "zara01", 78, [[19.5, 5, -3], [2, 5, -3]], 60),
    ("t_zara1_dense_cross", "zara01", 78, [[14, 13, -1.5], [14, 1, -1.5]], 60),
    ("t_zara1_dense_crosswrap_rev", "zara01", 78, [[4, 2, 0], [14, 13, -1.5]], 60),
    ("t_zara2", "zara02", 0, [[8.0, 8.5, 0], [15, 3, -0.2]], 60),
    ("t_zara2_basic_against", "zara02", 7, [[20, 5, -3], [1.5, 5, -3]], 60),
    ("t_zara2_wrap", "zara02", 20, [[20, 13, -1.5], [1.5, 8, -1.5]], 60),
    ("t_zara2_wrap2", "zara02", 20, [[20, 13, -1.5], [1.5, 1.5, -1.5]], 60),
    ("t_zara2_against_seq", "zara02", 44, [[1.5, 5, 0], [20, 5, 0]], 60),
    ("t_univ1", "univ", 0, [[8.0, 8.5, 0], [15, 1, -0.2]], 60),
    ("t_hotel_against", "hotel", 361, [[2, 5, 0], [15, 5, 0.2]], 60),
]
STAGES = [(name, f"socnavbench_{SCENES[dataset]}", max_time) for name, dataset, _, _, max_time in EPISODES if dataset in SCENES]

_RUNNER = """
import enum
import json
import attrs
from arena_evaluation.benchmark import runner
from arena_evaluation.benchmark.state import Manifest, compute_config_hash
from arena_evaluation.benchmark.tree import ContestIdentifier, SuiteIdentifier

verdicts = [identifier.resolve_source_sync() for identifier in (SuiteIdentifier(name="socnavbench"), ContestIdentifier(name="socnavbench"))]
suite, contest, suite_dict, contest_dict, bundle_dir, suite_provenance, contest_provenance = runner._load_suite_contest("socnavbench", "socnavbench")
steps = runner._all_steps(contest, suite, 1.0)
stage = steps[0].stage
obstacles, robots = runner._flatten_per_mode_params(stage.config, tm_obstacles=stage.tm_obstacles.value, tm_robots=stage.tm_robots.value)
records = [{"key": step.key, "contestant": attrs.asdict(step.contestant), "stage": {k: v.value if isinstance(v, enum.Enum) else v for k, v in step.stage._asdict().items()}} for step in steps]
manifest = Manifest(run_id="socnavbench", created_at="", arena_git_sha=None, arena_git_dirty=False, cli_args=[], env_n=1, headless=True, config_hash=compute_config_hash(suite_dict, contest_dict), simulator="gazebo", scale_episodes=1.0, suite_name=suite.name, contest_name=contest.name, suite=suite_dict, contest=contest_dict, steps=records, suite_provenance=suite_provenance, contest_provenance=contest_provenance)
assert Manifest.from_yaml(manifest.to_yaml()) == manifest
print(json.dumps({
    "resolvers": [repr(verdict.resolver) for verdict in verdicts],
    "verdicts": [verdict.verdict.name for verdict in verdicts],
    "bundle_dir": str(bundle_dir),
    "keys": [step.key for step in steps],
    "launch": runner.build_launch_args(steps[0], "gazebo", passthrough=suite.launch_args),
    "preflight": runner._preflight_contest(contest),
    "budgets": [runner.BenchmarkRunner._episode_budget(step) for step in steps],
    "worlds": [step.stage.map for step in steps],
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
manifest = resolve_manifest("socnavbench", benchmark_dir=run)
ReportBuilder.from_dirs([run], output_dir=run, manifest=manifest).build()
print(json.dumps({"available": available_manifests(), "file": str(find_manifest_file("socnavbench")), "name": manifest.name}))
"""

_RUNTIME = """
import json
from arena_simulation_setup.shared.track import Track, track_file
from arena_simulation_setup.tree.World import WorldIdentifier

verdict = WorldIdentifier("socnavbench_eth").resolve_source_sync()
view = WorldIdentifier("socnavbench_eth").resolve_sync()
world = view.load()
scenario = view.scenario("t_eth_dense_against").resolve_sync().load()
robot = scenario.robots[0]
goal = robot.phase_list()[0].pose
print(json.dumps({
    "path": str(verdict.path),
    "levels": list(world.levels),
    "scenarios": sorted(scenario.name for scenario in view.scenario.listall()),
    "start": [robot.start.position.x, robot.start.position.y],
    "goal": [goal.position.x, goal.position.y],
    "pedestrians": [entry.name for entry in scenario.humans],
    "track_starts": [Track.load(track_file(entry)).start for entry in scenario.humans],
}))
"""

_PUSH = """
import json
from arena_cli import asset
from arena_evaluation.benchmark.tree import SuiteIdentifier

identifier = SuiteIdentifier(name="socnavbench")
source = asset._config_source(identifier)
print(json.dumps({"source": str(source), "dangling": asset._suite_dangling(identifier.resolve_sync(), source)}))
"""

_IMPORT = """
import sys
import arena_evaluation.adapters.socnavbench
print(sorted(name for name in sys.modules if name.split(".")[0] == "task_generator"))
"""


def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _subprocess(script: str, *args: str, **env: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, env={**os.environ, **env}, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.splitlines()[-1])


def write_sequence(path: Path, idents: list[int]) -> None:
    """A SocNavBench sequence file: rows frame, id, y, x, a pedestrian per straight line, entering ten frames apart in id order."""
    columns = []
    for order, ident in enumerate(idents):
        for step in range(101):
            columns.append((10 * ident + step, ident, round(1.0 + order % 4, 4), round(0.08 * step if order % 2 else 8.0 - 0.08 * step, 4)))
    columns.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(zip(*columns, strict=True))


def seed(cache: Path) -> Path:
    for dataset in SCENES:
        starts = sorted({start for _, used, start, _, _ in EPISODES if used == dataset})
        write_sequence(cache / "agents" / "humans" / "datasets" / PLACEMENTS[dataset][0], [start + offset for start in starts for offset in (1, 2)])
    placements = ["[dataset_params]\n"]
    for name, (file, fps, offset, swapxy, flipxn, flipyn) in PLACEMENTS.items():
        placements.append(f"[{name}] ; recommended map\nfile_name={file}\nfps={fps} ; framerate of data capture\nspawn_delay_s = 0\noffset={offset!r}\nswapxy={swapxy}\nflipxn={flipxn}\nflipyn={flipyn}\n")
    episodes = ["[episodes_params]\ntests=['t_eth_dense_against']\n\nwithout_robot=False ; lite version without the robot\n"]
    for name, dataset, start, start_goal, max_time in EPISODES:
        episodes.append(f'[{name}]\nmap_name=Synthetic\npedestrian_datasets=["{dataset}"]\ndatasets_start_t = [0]\nped_ranges=[({start}, -1)]\nagents_start=[]\nagents_end=[]\nrobot_start_goal={start_goal!r}\nmax_time={max_time}\nwrite_episode_log=True\n')
    (cache / "params").mkdir(parents=True, exist_ok=True)
    (cache / "params" / "dataset_params.ini").write_text("\n".join(placements))
    (cache / "params" / "episode_params_val.ini").write_text("\n".join(episodes))
    return cache


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _ros_gate()
    root = tmp_path_factory.mktemp("assets")
    importer = SocNavBenchSuite(cache=seed(root / ".import" / "socnavbench"))
    assert importer.import_item("socnavbench", root / "_imported") == root / "_imported" / "suites" / "socnavbench"
    return root


@pytest.fixture(scope="module")
def out(assets: Path) -> Path:
    return assets / "_imported"


@pytest.fixture(scope="module")
def suite(out: Path) -> dict:
    return yaml.safe_load((out / "suites" / "socnavbench" / "suite.yaml").read_text())


def _files(root: Path) -> list[str]:
    return sorted(f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file())


def test_registry_loads_this_class():
    _ros_gate()
    assert IMPORTERS.load("suite", "socnavbench") is SocNavBenchSuite
    assert SocNavBenchSuite.meta.kind == "suite"
    assert list(SocNavBenchSuite(cache=Path("/nonexistent")).items()) == ["socnavbench"]
    assert SocNavBenchSuite.default_provider().url("params/episode_params_val.ini") == f"https://raw.githubusercontent.com/CMU-TBD/SocNavBench/{REVISION}/params/episode_params_val.ini"


def test_module_import_leaves_task_generator_unloaded():
    assert _subprocess(_IMPORT) == []


def test_import_writes_the_bucket_bundle_layout_with_both_scene_worlds(out: Path):
    bundle = out / "suites" / "socnavbench"
    assert sorted(path.name for path in bundle.iterdir()) == [ATTRIBUTION_FILE, "suite.yaml", "worlds"]
    assert sorted(path.name for path in (bundle / "worlds").iterdir()) == ["socnavbench_eth", "socnavbench_zara"]
    for world in ("socnavbench_eth", "socnavbench_zara"):
        assert sorted(path.name for path in (bundle / "worlds" / world).iterdir()) == [ATTRIBUTION_FILE, "map", "scenarios", "world.yaml"]
        assert _files(bundle / "worlds" / world / "map") == ["0.png", "0.yaml"]
        scenarios = sorted(name for name, stage_world, _ in STAGES if stage_world == world)
        assert sorted(path.name for path in (bundle / "worlds" / world / "scenarios").iterdir()) == scenarios
        for name in scenarios:
            files = _files(bundle / "worlds" / world / "scenarios" / name)
            assert {ATTRIBUTION_FILE, "scenario.yaml"} < set(files)
            assert all(file in (ATTRIBUTION_FILE, "scenario.yaml") or re.fullmatch(r"ped_\d+\.csv", file) for file in files)
    assert _files(out / "contests" / "socnavbench") == [ATTRIBUTION_FILE, "contest.yaml"]
    assert _files(out / "manifests" / "socnavbench") == [ATTRIBUTION_FILE, "manifest.yaml"]
    for kind in ("suites", "contests", "manifests"):
        attribution = read_attribution(out / kind / "socnavbench" / ATTRIBUTION_FILE)
        assert attribution.license == "MIT for the episodes, unspecified for the ETH and UCY trajectories"
        assert "Silvera" in attribution.authors and "2103.00047" in attribution.citation
    scene = read_attribution(bundle / "worlds" / "socnavbench_zara" / ATTRIBUTION_FILE)
    assert REVISION in scene.url and "Lerner" in scene.authors


def test_suite_lists_the_seventeen_episodes_with_their_time_budgets(suite: dict):
    assert set(suite) == {"launch", "metrics", "stages"}
    assert suite["launch"] == {"human": "replay"}
    assert suite["metrics"] == {"max_collisions": 1}
    assert len(suite["stages"]) == 17
    assert [(stage["name"], stage["map"], stage["timeout"]) for stage in suite["stages"]] == [(name, world, f"{max_time}s") for name, world, max_time in STAGES]
    assert {stage["timeout"] for stage in suite["stages"]} == {"20s", "50s", "60s"}
    assert [stage["name"] for stage in suite["stages"] if stage["timeout"] != "60s"] == ["t_ETH1", "t_zara1_wrap"]
    assert [stage["map"] for stage in suite["stages"]] == ["socnavbench_eth"] * 6 + ["socnavbench_zara"] * 11
    for stage in suite["stages"]:
        assert set(stage) == NATIVE_STAGE_KEYS
    assert suite["stages"][0] == {
        "name": "t_ETH1",
        "map": "socnavbench_eth",
        "robot": "jackal",
        "episodes": 1,
        "tm_robots": "scenario",
        "tm_obstacles": "scenario",
        "config": {"scenario": {"file": "t_ETH1"}},
        "timeout": "20s",
    }


def test_bundled_scenario_holds_the_episode_start_and_goal(out: Path):
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    scenario = yaml.safe_load((out / "suites" / "socnavbench" / "worlds" / "socnavbench_eth" / "scenarios" / "t_eth_dense_against" / "scenario.yaml").read_text())
    assert scenario.keys() == {"format", "humans", "robots"}
    assert scenario["robots"] == [{"start": [13.4, 3.6, round(1.5 + math.pi / 2, 3)], "phases": [{"goto": [-1.1, 11.1, round(math.pi / 2, 3)]}]}]
    assert [entry["name"] for entry in scenario["humans"]] == ["ped_251", "ped_252"]
    goal = RobotGoal.parse(scenario["robots"][0])
    (phase,) = goal.phase_list()
    assert isinstance(phase, GoToPhase)
    assert (goal.start.position.x, phase.pose.position.y) == pytest.approx((13.4, 11.1))
    zara = yaml.safe_load((out / "suites" / "socnavbench" / "worlds" / "socnavbench_zara" / "scenarios" / "t_zara2_wrap" / "scenario.yaml").read_text())
    assert zara["robots"][0]["start"][:2] == [-10.5, -10.0]
    assert zara["robots"][0]["phases"][0]["goto"][:2] == [8.0, -5.0]


def test_contest_pins_the_goal_margin_and_keeps_the_episode_running(out: Path):
    from task_generator.constants.runtime import EPISODE_PARAMS

    contest = yaml.safe_load((out / "contests" / "socnavbench" / "contest.yaml").read_text())
    assert contest == [
        {
            "name": "dwb",
            "mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"},
            "task.episode.goto_pose.tolerance.radius": 0.3,
            "task.episode.goto_pose.tolerance.angle": "0.0",
        }
    ]
    assert {key for key in contest[0] if key.startswith("task.")} <= set(EPISODE_PARAMS)


def test_suite_contest_and_manifest_load_through_their_identifiers(out: Path):
    from task_generator.constants import Constants

    from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier

    suite = SuiteIdentifier(name="socnavbench").load(out / "suites" / "socnavbench")
    assert [(stage.name, stage.map, stage.timeout) for stage in suite.stages] == [(name, world, float(max_time)) for name, world, max_time in STAGES]
    assert len(suite.stages) == 17
    assert {stage.tm_robots for stage in suite.stages} == {Constants.TaskMode.TM_Robots.SCENARIO}
    assert {stage.tm_obstacles for stage in suite.stages} == {Constants.TaskMode.TM_Obstacles.SCENARIO}
    assert {stage.robot for stage in suite.stages} == {"jackal"}
    assert {stage.episodes for stage in suite.stages} == {1}
    assert suite.launch_args == {"human": "replay"}
    assert suite.metrics == {"max_collisions": 1}
    assert suite.references is False

    contest = ContestIdentifier(name="socnavbench").load(out / "contests" / "socnavbench")
    assert [contestant.name for contestant in contest.contestants] == ["dwb"]
    assert contest.contestants[0].args["mobile"]["driver"] == "nav2"

    manifest = ManifestIdentifier(name="socnavbench").load(out / "manifests" / "socnavbench")
    assert manifest.name == "socnavbench"
    assert manifest.data_source == "metrics"
    assert manifest.summary_group_by == ["planner"]
    assert [column.metric for column in manifest.summary] == ["success", "collision_amount", "time_to_goal"]
    assert {plot.layout_group for plot in manifest.plots} == {group.id for group in manifest.groups}
    (per_episode,) = manifest.plots
    assert per_episode.type == "table"
    assert per_episode.options["group_by"] == ["planner", "stage"]
    assert [column["metric"] for column in per_episode.options["columns"]] == ["success", "collision_amount", "time_to_goal"]
    text = (out / "manifests" / "socnavbench" / "manifest.yaml").read_text()
    assert yaml.safe_load(text) == socnavbench.MANIFEST
    assert "&id" not in text


def test_runner_loads_the_import_through_the_registered_resolvers(assets: Path, out: Path, tmp_path: Path):
    from task_generator.constants.runtime import EPISODE_PARAMS

    from arena_evaluation.processing.pipeline import _suite_metrics

    seen = _subprocess(_RUNNER, ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out))
    assert all("ConfigPathResolver" in resolver and str(out) in resolver for resolver in seen["resolvers"])
    assert seen["verdicts"] == ["HIT", "HIT"]
    assert seen["bundle_dir"] == str(out / "suites" / "socnavbench")
    assert seen["keys"] == [f"dwb/{name}" for name, _, _ in STAGES]
    assert seen["preflight"] == []
    assert seen["budgets"] == [float(max_time) for _, _, max_time in STAGES]
    assert seen["worlds"] == [world for _, world, _ in STAGES]
    assert seen["obstacles"] == [["file", "t_ETH1"]]
    assert seen["robots"] == [["file", "t_ETH1"]]
    launch = dict(arg.split(":=", 1) for arg in seen["launch"])
    assert launch["world"] == "socnavbench_eth"
    assert launch["robot"] == "jackal"
    assert launch["human"] == "replay"
    assert launch["task.robots"] == "scenario"
    assert launch["task.obstacles"] == "scenario"
    assert launch["robot.mobile"] == "nav2"
    assert launch["robot.mobile.local_planner"] == "dwb"
    assert launch["robot.mobile.inter_planner"] == "navigate_w_replanning_time"
    episode = {key: yaml.safe_load(value) for key, value in launch.items() if key in EPISODE_PARAMS}
    assert episode == {
        "task.episode.auto_reset": False,
        "task.episode.goto_pose.tolerance.radius": 0.3,
        "task.episode.goto_pose.tolerance.angle": 0.0,
    }
    assert set(socnavbench.CONTESTANT) <= {"name", "mobile", *episode}
    manifest = yaml.safe_load(seen["manifest"])
    assert manifest["steps"][0]["stage"]["map"] == "socnavbench_eth"
    assert manifest["suite"]["launch"] == {"human": "replay"}
    episode_dir = tmp_path / "socnavbench" / "episodes" / "episode_000"
    episode_dir.mkdir(parents=True)
    (tmp_path / "socnavbench" / "manifest.yaml").write_text(seen["manifest"])
    assert _suite_metrics(episode_dir, tmp_path) == {"max_collisions": 1}


def test_report_renders_the_metrics_per_contestant_and_per_episode(assets: Path, out: Path, tmp_path: Path):
    import polars as pl

    run = tmp_path / "20261007-000000-socnavbench-socnavbench"
    run.mkdir()
    pl.DataFrame(
        {
            "planner": ["dwb", "dwb", "teb", "teb"],
            "map": ["socnavbench_eth", "socnavbench_zara", "socnavbench_eth", "socnavbench_zara"],
            "stage": ["t_ETH1", "t_zara2", "t_ETH1", "t_zara2"],
            "episode": [0, 1, 2, 3],
            "success": [True, True, False, True],
            "collision_amount": [0, 0, 1, 0],
            "time_to_goal": [18.0, 23.2, 20.0, 41.2],
        }
    ).write_parquet(run / "combined_metrics.parquet")

    seen = _subprocess(_REPORT, str(run), ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out))
    assert seen["name"] == "socnavbench"
    assert seen["file"] == str(out / "manifests" / "socnavbench" / "manifest.yaml")
    assert {"socnavbench", "standard"} <= set(seen["available"])
    assert yaml.safe_load((run / "report_manifest.yaml").read_text())["name"] == "socnavbench"

    content = (run / "report.html").read_text()
    assert "SocNavBench Report" in content
    rows = [re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S) for row in re.findall(r"<tr[^>]*>(.*?)</tr>", content, re.S)]
    assert ["planner", "Success", "Collisions", "Time to Goal (s)"] in rows
    assert ["dwb", "100%", "0.00", "20.60"] in rows
    assert ["teb", "50%", "0.50", "30.60"] in rows
    assert ["Planner", "Stage", "Success", "Collisions", "Time to Goal (s)"] in rows
    assert ["dwb", "t_ETH1", "100%", "0.00", "18.00"] in rows
    assert ["dwb", "t_zara2", "100%", "0.00", "23.20"] in rows
    assert ["teb", "t_ETH1", "0%", "1.00", "20.00"] in rows
    assert ["teb", "t_zara2", "100%", "0.00", "41.20"] in rows


def test_runtime_resolves_a_bundled_world_and_scenario_from_the_exported_world_path(out: Path):
    worlds = out / "suites" / "socnavbench" / "worlds"
    seen = _subprocess(_RUNTIME, ARENA_WORLD_PATH=str(worlds))
    assert seen["path"] == str(worlds / "socnavbench_eth")
    assert seen["levels"] == ["0"]
    assert seen["scenarios"] == sorted(name for name, world, _ in STAGES if world == "socnavbench_eth")
    assert seen["start"] == pytest.approx([13.4, 3.6])
    assert seen["goal"] == pytest.approx([-1.1, 11.1])
    assert seen["pedestrians"] == ["ped_251", "ped_252"]
    assert seen["track_starts"] == pytest.approx([0.0, 10 / 15], abs=0.1)


def test_push_preflight_finds_no_dangling_world(assets: Path, out: Path):
    if not (ARENA_CLI / "arena_cli" / "asset.py").is_file():
        pytest.skip("arena_cli is not checked out beside this package")
    path = os.pathsep.join(filter(None, [str(ARENA_CLI), os.environ.get("PYTHONPATH")]))
    seen = _subprocess(_PUSH, ARENA_ASSETS_DIR=str(assets), ARENA_ASSETS_DIR_IMPORTED=str(out), PYTHONPATH=path)
    assert seen == {"source": str(out / "suites" / "socnavbench"), "dangling": []}


def test_convert_replaces_a_previous_bundle(tmp_path: Path):
    _ros_gate()
    cache = seed(tmp_path / "cache")
    stale = tmp_path / "out" / "suites" / "socnavbench" / "worlds" / "socnavbench_hotel"
    stale.mkdir(parents=True)
    assert SocNavBenchSuite(cache=tmp_path / "elsewhere").convert("socnavbench", cache, tmp_path / "out") == "suites/socnavbench"
    assert not stale.exists()
    assert (tmp_path / "out" / "suites" / "socnavbench" / "worlds" / "socnavbench_eth" / "world.yaml").is_file()
    assert not (tmp_path / "elsewhere").exists()


def test_unknown_item_is_refused(tmp_path: Path):
    _ros_gate()
    importer = SocNavBenchSuite(cache=tmp_path / "cache")
    for call in (importer.fetch, importer.attribution, lambda item: importer.convert(item, tmp_path / "cache", tmp_path / "out")):
        with pytest.raises(KeyError, match="the only item is 'socnavbench'"):
            call("t_ETH1")
    assert not (tmp_path / "out").exists()


@pytest.mark.network
def test_live_import_lists_the_seventeen_episodes_with_the_upstream_time_budgets(tmp_path: Path):
    _ros_gate()
    from arena_evaluation.benchmark.tree import SuiteIdentifier

    importer = SocNavBenchSuite(cache=tmp_path / "cache")
    bundle = importer.import_item("socnavbench", tmp_path / "out")
    suite = SuiteIdentifier(name="socnavbench").load(bundle)
    assert [(stage.name, stage.map, stage.timeout) for stage in suite.stages] == [(name, world, float(max_time)) for name, world, max_time in STAGES]
    for stage in suite.stages:
        scenario = yaml.safe_load((bundle / "worlds" / stage.map / "scenarios" / stage.name / "scenario.yaml").read_text())
        assert len(scenario["robots"]) == 1 and len(scenario["humans"]) >= 14, stage.name
    eth = yaml.safe_load((bundle / "worlds" / "socnavbench_eth" / "scenarios" / "t_eth_dense_against" / "scenario.yaml").read_text())
    assert len(eth["humans"]) == 66
    assert eth["robots"][0] == {"start": [13.4, 3.6, 3.071], "phases": [{"goto": [-1.1, 11.1, 1.571]}]}
    defaults = importer.provider.get("params/.default_params.ini", tmp_path / "default_params.ini").read_text()
    assert re.search(r"^goal_margin=([0-9.]+)", defaults, re.M).group(1) == str(socnavbench.GOAL_MARGIN_M)
    user = importer.provider.get("params/user_params.ini", tmp_path / "user_params.ini").read_text()
    assert re.search(r"^keep_episode_running=(\w+)", user, re.M).group(1) == "True"
