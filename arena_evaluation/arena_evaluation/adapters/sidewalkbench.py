"""SidewalkBench protocol as a suite bundle with generated sidewalk worlds and the scanned scenes, plus a contest and a report manifest."""

from __future__ import annotations

import math
import shutil
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, write_attribution
from arena_assets.importers.providers import Provider
from arena_evaluation.processing.metrics.performance.sidewalk_metrics import GOAL_RADIUS_M

ITEM = "sidewalkbench"
ROBOT = "jackal"
HUMAN = "arena"
SEEDS = range(42, 47)
SHORT_TIMEOUT_S = 60
SPEED_MPS = 0.5
DETOUR = 1.25
START_SLACK_S = 10
BUDGET_STEP_S = 10
UNIT_LENGTH_M = 25.0
REACTIVE_LENGTH_M = 30.0
BLOCK_BLOCKS = 2
BLOCK_PEDESTRIANS = 10
UNIT_LAYOUTS = ("straight", "corner", "crosswalk")
BEHAVIORS = ("obstructing", "conversing", "queueing", "frontal", "lateral", "overtaking", "crossing", "gesturing")
REACTIVE_LAYOUT = {behavior: "crosswalk" if behavior == "crossing" else "straight" for behavior in BEHAVIORS}
SCENE_KINDS = ("straight", "curve", "crosswalk", "long")
WORLD_PREFIX = "sidewalkbench_"
GENERATED_PREFIX = "sidewalk_"

CONTESTANT = {
    "name": "dwb",
    "mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"},
    "task.episode.fail_on_collision": True,
    "task.episode.goto_pose.tolerance.radius": GOAL_RADIUS_M,
    "task.episode.goto_pose.tolerance.angle": "0.0",
}

GROUP_TITLES = {"straight": "Straight", "corner": "Corner and curve", "crosswalk": "Crosswalk", "long": "Long horizon"}
GROUP_OF_KIND = {"straight": "straight", "corner": "corner", "curve": "corner", "crosswalk": "crosswalk", "block": "long", "long": "long"}


def world_name(layout: str, seed: int, scenario: str = "empty") -> str:
    infix = "" if scenario == "empty" else f"{scenario}_"
    return f"{GENERATED_PREFIX}{layout}_{infix}{seed}"


def route_budget_s(length_m: float) -> int:
    """Episode budget for a route of length_m at the nominal robot speed with a detour factor and a start slack, never under SHORT_TIMEOUT_S."""
    seconds = length_m * DETOUR / SPEED_MPS + START_SLACK_S
    return max(SHORT_TIMEOUT_S, int(math.ceil(seconds / BUDGET_STEP_S) * BUDGET_STEP_S))


def pose_xy(value: object) -> tuple[float, float] | None:
    """x and y of a pose written as a list or a mapping, None for a zone or entity name."""
    if isinstance(value, Mapping):
        return float(value["x"]), float(value["y"])
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def route_length(scenario: dict) -> float:
    """Polyline length of the first robot's start and its pose goto phases."""
    robot = scenario["robots"][0]
    points = [pose_xy(robot["start"])]
    for phase in robot.get("phases", []):
        points.append(pose_xy(phase.get("goto")) if isinstance(phase, Mapping) else None)
    route = [point for point in points if point is not None]
    return float(sum(math.dist(a, b) for a, b in zip(route[:-1], route[1:], strict=True)))


def scenario_kind(name: str) -> str:
    return name.rsplit("_", 1)[0]


def generated_config(layout: str, scenario: str) -> dict:
    config: dict[str, typing.Any] = {"layout": layout, "scenario": scenario}
    if layout == "block":
        config.update(blocks=BLOCK_BLOCKS, pedestrians=BLOCK_PEDESTRIANS)
    else:
        config["length"] = REACTIVE_LENGTH_M if scenario != "empty" else UNIT_LENGTH_M
    return config


def generated_specs() -> list[tuple[str, str, str, int]]:
    """Name, layout, scenario and seed of every generated world of the protocol."""
    specs: list[tuple[str, str, str, int]] = []
    for seed in SEEDS:
        for layout in UNIT_LAYOUTS:
            specs.append((world_name(layout, seed), layout, "empty", seed))
        for behavior in BEHAVIORS:
            layout = REACTIVE_LAYOUT[behavior]
            specs.append((world_name(layout, seed, behavior), layout, behavior, seed))
        specs.append((world_name("block", seed), "block", "default", seed))
    return specs


def stage(name: str, world: str, binding: dict, timeout_s: int) -> dict:
    modes = {"robots": "scenario", "obstacles": "scenario", **binding["modes"]}
    config: dict[str, dict] = {}
    for axis, params in binding["params"].items():
        config.setdefault(modes[axis], {}).update(params)
    return {
        "name": name,
        "map": world,
        "robot": ROBOT,
        "episodes": 1,
        "modes": modes,
        "config": config,
        "timeout": f"{timeout_s}s",
    }


def scene_stages(scene: str, scenarios: Sequence[tuple[str, float]]) -> list[dict]:
    """One stage per route scenario of a scanned scene, the long routes with a budget from their length."""
    stages = []
    for name, length in scenarios:
        timeout = route_budget_s(length)
        binding = {"modes": {}, "params": {"robots": {"file": name}}}
        stages.append(stage(f"{scene}_{name}", f"{WORLD_PREFIX}{scene}", binding, timeout))
    return stages


def classify(stage_name: str) -> tuple[str, str | None]:
    """Layout group and pedestrian behavior of a stage name, the behavior None for unit-test and long-horizon stages."""
    if stage_name.startswith(GENERATED_PREFIX):
        parts = stage_name.removeprefix(GENERATED_PREFIX).split("_")
        return GROUP_OF_KIND[parts[0]], parts[1] if len(parts) == 3 else None
    return GROUP_OF_KIND[stage_name.rsplit("_", 2)[-2]], None


def _table(plot_id: str, title: str, group: str, members: Sequence[str], columns: list[dict]) -> dict:
    return {
        "id": plot_id,
        "type": "table",
        "title": title,
        "data_key": "*",
        "layout_group": group,
        "filter": {"stage": list(members)},
        "options": {"group_by": ["planner"], "columns": columns},
    }


def manifest(stage_names: Sequence[str]) -> dict:
    """Report manifest with the route completion per layout, the success per behavior, failures per 100 m and mean speed per planner."""
    labels = {name: classify(name) for name in stage_names}
    completion = [{"metric": "sidewalk_route_completion", "label": "Route completion", "format": "{:.0%}"}]
    plots: list[dict] = []
    for group, title in GROUP_TITLES.items():
        members = [n for n, (g, behavior) in labels.items() if g == group and behavior is None]
        if members:
            plots.append(_table(f"sidewalkbench_layout_{group}", f"Route completion, {title}", "layouts", members, completion))
    for behavior in BEHAVIORS:
        members = [n for n, (_, b) in labels.items() if b == behavior]
        if members:
            plots.append(_table(f"sidewalkbench_behavior_{behavior}", f"Success, {behavior}", "behaviors", members, [{"metric": "sidewalk_route_completion", "label": "Success", "format": "{:.0%}"}]))
    long_members = [n for n, (g, behavior) in labels.items() if g == "long" and behavior is None]
    if long_members:
        columns = [
            {"metric": "sidewalk_failures_per_100m", "label": "Failures per 100 m", "format": "{:.2f}"},
            {"metric": "sidewalk_mean_speed", "label": "Mean speed (m/s)", "format": "{:.2f}"},
        ]
        plots.append(_table("sidewalkbench_long_horizon", "Long horizon", "long", long_members, columns))
    return {
        "manifest_version": "1.0",
        "name": ITEM,
        "title": "SidewalkBench Report",
        "description": "Route completion per layout, success per pedestrian behavior, failures per 100 m and mean speed per contestant on the SidewalkBench stages.",
        "data_source": "metrics",
        "groups": [
            {"id": "layouts", "title": "Unit tests per layout"},
            {"id": "behaviors", "title": "Pedestrian-reactive behaviors"},
            {"id": "long", "title": "Long horizon"},
        ],
        "summary": [
            {"metric": "sidewalk_route_completion", "label": "Route completion", "format": "{:.0%}"},
            {"metric": "sidewalk_failures_per_100m", "label": "Failures per 100 m", "format": "{:.2f}"},
            {"metric": "sidewalk_mean_speed", "label": "Mean speed (m/s)", "format": "{:.2f}"},
        ],
        "summary_group_by": ["planner"],
        "plots": plots,
    }


def _dump(path: Path, obj: object) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


class SidewalkBench(Importer):
    meta = IMPORTERS.meta("suite", ITEM)

    def __init__(self, cache: Path | None = None, provider: Provider | None = None, scenes: Sequence[str] | None = None):
        super().__init__(cache, provider)
        self.scenes = None if scenes is None else list(scenes)

    @classmethod
    def default_provider(cls) -> Provider:
        return IMPORTERS.load("world", ITEM).default_provider()

    @staticmethod
    def _check(item: str) -> None:
        if item != ITEM:
            raise KeyError(f"no item '{item}', the only item is '{ITEM}'")

    def _worlds(self, cache: Path) -> Importer:
        return IMPORTERS.load("world", ITEM)(cache=cache, provider=self.provider)

    def items(self) -> Iterable[str]:
        return [ITEM]

    def fetch(self, item: str) -> Path:
        self._check(item)
        worlds = self._worlds(self.cache)
        for scene in self.scenes if self.scenes is not None else worlds.items():
            worlds.fetch(scene)
        return self.cache

    def _scene_scenarios(self, world_dir: Path) -> list[tuple[str, float]]:
        found = []
        for scenario_dir in sorted((world_dir / "scenarios").iterdir()):
            kind, _, index = scenario_dir.name.rpartition("_")
            if kind not in SCENE_KINDS or not index.isdigit():
                continue
            with open(scenario_dir / "scenario.yaml") as f:
                found.append((scenario_dir.name, route_length(yaml.safe_load(f))))
        return sorted(found, key=lambda n: (SCENE_KINDS.index(scenario_kind(n[0])), int(n[0].rpartition("_")[2])))

    def convert(self, item: str, src: Path, out: Path) -> str:
        self._check(item)

        from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
        from arena_simulation_setup.generative.world.world_generator import save_generated
        from arena_simulation_setup.tree.World import WorldView, derive

        from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier

        suite_dir = out / SuiteIdentifier(name=item).relpath()
        contest_dir = out / ContestIdentifier(name=item).relpath()
        manifest_dir = out / ManifestIdentifier(name=item).relpath()
        for bundle in (suite_dir, contest_dir, manifest_dir):
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir(parents=True)

        stages: list[dict] = []
        for name, layout, scenario, seed in generated_specs():
            generator = WorldGenerator(WorldGeneratorType.SIDEWALK, generated_config(layout, scenario), seed=seed)
            built = generator.compute()
            built.kg = derive(built)
            files = generator.files()
            save_generated(WorldView(suite_dir / "worlds" / name), built, files)
            binding = generator.params()
            file = binding["params"]["robots"]["file"]
            timeout = route_budget_s(route_length(yaml.safe_load(files[f"scenarios/{file}/scenario.yaml"])))
            stages.append(stage(name, name, binding, timeout))

        scenes = self.scenes if self.scenes is not None else list(self._worlds(src).items())
        if scenes:
            worlds = self._worlds(src)
            for scene in scenes:
                imported = worlds.import_item(scene, out / "worlds")
                stages.extend(scene_stages(scene, self._scene_scenarios(imported)))

        _dump(suite_dir / "suite.yaml", {"launch": {"human": HUMAN}, "metrics": {"max_collisions": 1}, "stages": stages})
        _dump(contest_dir / "contest.yaml", [CONTESTANT])
        _dump(manifest_dir / "manifest.yaml", manifest([s['name'] for s in stages]))
        SuiteIdentifier(name=item).load(suite_dir)
        ContestIdentifier(name=item).load(contest_dir)
        ManifestIdentifier(name=item).load(manifest_dir)

        for bundle in (contest_dir, manifest_dir):
            write_attribution(bundle / ATTRIBUTION_FILE, self.attribution(item))
        return str(SuiteIdentifier(name=item).relpath())

    def attribution(self, item: str) -> Attribution:
        self._check(item)
        return Attribution(
            source="SidewalkBench",
            url="https://github.com/VAIL-UCLA/SidewalkBench",
            license="Apache 2.0 for the benchmark code, MIT for the scanned scenes",
            authors="Zhizheng Liu, Honglin He, Vivek Alumootil, Akshat Pandya, Brad Squicciarini, Wayne Wu, Bolei Zhou",
            citation="Z. Liu, H. He, V. Alumootil, A. Pandya, B. Squicciarini, W. Wu, B. Zhou. SidewalkBench: Benchmarking Visual Navigation on Urban Sidewalks. CoRL 2026. arXiv:2606.16953",
        )
