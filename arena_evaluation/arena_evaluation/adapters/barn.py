"""BARN challenge as a suite bundle carrying its 300 worlds and reference path lengths, plus a contest and a report manifest."""

from __future__ import annotations

import math
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import yaml
from arena_simulation_setup.generative.world.barn import BASE_URL, CELL_M, GOAL, START, WORLD_DIR, WORLDS, X_SHIFT_M, world_file, world_name
from arena_simulation_setup.generative.world.barn import attribution as barn_attribution

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, write_attribution
from arena_assets.importers.providers import Http, Provider
from arena_evaluation.processing.metrics.performance.barn_metrics import GOAL_RADIUS_M, TIMEOUT_S

ITEM = "barn"
ROBOT = "jackal"
START_SLACK_S = 10
PATH_DIR = f"{WORLD_DIR}/path_files"
PATHS = "path_files"
SOURCES = "worlds"
CONTESTANT = {
    "name": "dwb",
    "mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"},
    "task.episode.fail_on_collision": True,
    "task.episode.goto_pose.tolerance.radius": GOAL_RADIUS_M,
    "task.episode.goto_pose.tolerance.angle": "0.0",
}

MANIFEST = {
    "manifest_version": "1.0",
    "name": ITEM,
    "title": "BARN Challenge Report",
    "description": "Navigation metric of the BARN challenge per contestant and per world, beside the native success, collision and time metrics.",
    "data_source": "metrics",
    "groups": [{"id": "contestants", "title": "Contestants"}, {"id": "worlds", "title": "Worlds"}],
    "summary": [
        {"metric": "barn_score", "label": "Avg Metric", "format": "{:.4f}"},
        {"metric": "barn_success", "label": "Avg Success", "format": "{:.4f}"},
        {"metric": "barn_collided", "label": "Avg Collision", "format": "{:.4f}"},
        {"metric": "barn_timeout", "label": "Avg Timeout", "format": "{:.4f}"},
    ],
    "summary_group_by": ["planner"],
    "plots": [
        {
            "id": "barn_time_of_successful_runs",
            "type": "table",
            "title": "Time of successful runs",
            "data_key": "*",
            "layout_group": "contestants",
            "filter": {"barn_success": True},
            "options": {"group_by": ["planner"], "columns": [{"metric": "barn_time", "label": "Avg Time (s)", "format": "{:.4f}"}]},
        },
        {
            "id": "barn_native_metrics",
            "type": "table",
            "title": "Native metrics of the same runs",
            "data_key": "*",
            "layout_group": "contestants",
            "options": {
                "group_by": ["planner"],
                "columns": [
                    {"metric": "success", "label": "Success", "format": "{:.0%}"},
                    {"metric": "collision_amount", "label": "Collisions", "format": "{:.2f}"},
                    {"metric": "time_to_goal", "label": "Time to Goal (s)", "format": "{:.2f}"},
                ],
            },
        },
        {
            "id": "barn_score_per_world",
            "type": "table",
            "title": "Navigation metric per world",
            "data_key": "*",
            "layout_group": "worlds",
            "options": {
                "group_by": ["planner", "map"],
                "columns": [
                    {"metric": "barn_score", "label": "Metric", "format": "{:.4f}"},
                    {"metric": "barn_time", "label": "Time (s)", "format": "{:.2f}"},
                    {"metric": "barn_optimal_time", "label": "Optimal Time (s)", "format": "{:.2f}"},
                    {"metric": "barn_success", "label": "Success", "format": "{:.2f}"},
                    {"metric": "barn_collided", "label": "Collision", "format": "{:.2f}"},
                    {"metric": "barn_timeout", "label": "Timeout", "format": "{:.2f}"},
                ],
            },
        },
    ],
}

_PATH_FILE = re.compile(r"^path_(\d+)\.npy$")


def challenge_xy(row: float, col: float) -> tuple[float, float]:
    return row * CELL_M - 4.575 + X_SHIFT_M, col * CELL_M + 5.075


def reference_path_length(path: np.ndarray) -> float:
    points = [START[:2], *(challenge_xy(r, c) for r, c in np.asarray(path, dtype=float).tolist()), GOAL]
    return float(sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:], strict=True)))


def load_paths(src: Path) -> dict[int, np.ndarray]:
    paths: dict[int, np.ndarray] = {}
    if src.is_dir():
        for file in src.iterdir():
            match = _PATH_FILE.match(file.name)
            if match:
                paths[int(match.group(1))] = np.load(file)
    if not paths:
        raise FileNotFoundError(f"no path_<n>.npy under {src}")
    return paths


def stage(name: str, binding: dict) -> dict:
    return {
        "name": name,
        "map": name,
        "robot": ROBOT,
        "episodes": 1,
        "tm_robots": binding["tm_robots"],
        "tm_obstacles": "scenario",
        "config": {binding["tm_robots"]: dict(binding["robots_params"])},
        "timeout": f"{round(TIMEOUT_S) + START_SLACK_S}s",
    }


def _dump(path: Path, obj: object) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


class BARN(Importer):
    meta = IMPORTERS.meta("suite", ITEM)

    def __init__(self, cache: Path | None = None, provider: Provider | None = None, worlds: int = WORLDS):
        super().__init__(cache, provider)
        self.worlds = worlds

    @classmethod
    def default_provider(cls) -> Provider:
        return Http(BASE_URL)

    @staticmethod
    def _check(item: str) -> None:
        if item != ITEM:
            raise KeyError(f"no item '{item}', the only item is '{ITEM}'")

    def items(self) -> Iterable[str]:
        return [ITEM]

    def fetch(self, item: str) -> Path:
        self._check(item)
        for index in range(self.worlds):
            self.provider.get(f"{PATH_DIR}/path_{index}.npy", self.cache / PATHS / f"path_{index}.npy")
            self.provider.get(f"{WORLD_DIR}/{world_file(index)}", self.cache / SOURCES / world_file(index))
        return self.cache

    def convert(self, item: str, src: Path, out: Path) -> str:
        self._check(item)
        paths = load_paths(src / PATHS)

        from arena_simulation_setup.adapters.barn import write_world

        from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier

        suite_dir = out / SuiteIdentifier(name=item).relpath()
        contest_dir = out / ContestIdentifier(name=item).relpath()
        manifest_dir = out / ManifestIdentifier(name=item).relpath()
        for bundle in (suite_dir, contest_dir, manifest_dir):
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir(parents=True)

        stages: list[dict] = []
        lengths: dict[str, float] = {}
        for index in sorted(paths):
            name = world_name(index)
            generator = write_world(index, src / SOURCES, suite_dir / "worlds")
            stages.append(stage(name, generator.params()))
            lengths[name] = reference_path_length(paths[index])

        reference = {"start": list(START[:2]), "goal": list(GOAL), "path_length": lengths}
        _dump(suite_dir / "suite.yaml", {"metrics": {"max_collisions": 1, "barn": reference}, "stages": stages})
        _dump(contest_dir / "contest.yaml", [CONTESTANT])
        _dump(manifest_dir / "manifest.yaml", MANIFEST)
        SuiteIdentifier(name=item).load(suite_dir)
        ContestIdentifier(name=item).load(contest_dir)
        ManifestIdentifier(name=item).load(manifest_dir)

        for bundle in (contest_dir, manifest_dir):
            write_attribution(bundle / ATTRIBUTION_FILE, self.attribution(item))
        return str(SuiteIdentifier(name=item).relpath())

    def attribution(self, item: str) -> Attribution:
        self._check(item)
        return barn_attribution()
