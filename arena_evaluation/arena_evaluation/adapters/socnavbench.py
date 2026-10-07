"""SocNavBench curated episodes as a suite bundle carrying its two scene worlds and a scenario per episode, plus a contest and a report manifest."""

from __future__ import annotations

import shutil
import typing
from collections.abc import Iterable
from pathlib import Path

import yaml

from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, write_attribution
from arena_assets.importers.providers import Provider

if typing.TYPE_CHECKING:
    from task_generator.adapters.socnavbench import SocNavBench

ITEM = "socnavbench"
ROBOT = "jackal"
HUMAN = "replay"
GOAL_MARGIN_M = 0.3
CONTESTANT = {
    "name": "dwb",
    "mobile": {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"},
    "task.episode.goto_pose.tolerance.radius": GOAL_MARGIN_M,
    "task.episode.goto_pose.tolerance.angle": "0.0",
}
_COLUMNS = [
    {"metric": "success", "label": "Success", "format": "{:.0%}"},
    {"metric": "collision_amount", "label": "Collisions", "format": "{:.2f}"},
    {"metric": "time_to_goal", "label": "Time to Goal (s)", "format": "{:.2f}"},
]

MANIFEST = {
    "manifest_version": "1.0",
    "name": ITEM,
    "title": "SocNavBench Report",
    "description": "Success, collisions and time to goal on the SocNavBench curated episodes per contestant and per episode.",
    "data_source": "metrics",
    "groups": [{"id": "episodes", "title": "Episodes"}],
    "summary": [dict(column) for column in _COLUMNS],
    "summary_group_by": ["planner"],
    "plots": [
        {
            "id": "socnavbench_per_episode",
            "type": "table",
            "title": "Metrics per episode",
            "data_key": "*",
            "layout_group": "episodes",
            "options": {"group_by": ["planner", "stage"], "columns": [dict(column) for column in _COLUMNS]},
        },
    ],
}


def stage(name: str, world: str, max_time: float) -> dict:
    return {
        "name": name,
        "map": world,
        "robot": ROBOT,
        "episodes": 1,
        "tm_robots": "scenario",
        "tm_obstacles": "scenario",
        "config": {"scenario": {"file": name}},
        "timeout": f"{max_time:g}s",
    }


def _dump(path: Path, obj: object) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


class SocNavBenchSuite(Importer):
    meta = IMPORTERS.meta("suite", ITEM)

    @classmethod
    def default_provider(cls) -> Provider:
        return IMPORTERS.load("scenario", ITEM).default_provider()

    @staticmethod
    def _check(item: str) -> None:
        if item != ITEM:
            raise KeyError(f"no item '{item}', the only item is '{ITEM}'")

    def _episodes(self, cache: Path) -> SocNavBench:
        return typing.cast("SocNavBench", IMPORTERS.load("scenario", ITEM)(cache=cache, provider=self.provider))

    def items(self) -> Iterable[str]:
        return [ITEM]

    def fetch(self, item: str) -> Path:
        self._check(item)
        episodes = self._episodes(self.cache)
        for name in episodes.items():
            episodes.fetch(name)
        return self.cache

    def convert(self, item: str, src: Path, out: Path) -> str:
        self._check(item)

        from arena_evaluation.benchmark.tree import ContestIdentifier, ManifestIdentifier, SuiteIdentifier

        suite_dir = out / SuiteIdentifier(name=item).relpath()
        contest_dir = out / ContestIdentifier(name=item).relpath()
        manifest_dir = out / ManifestIdentifier(name=item).relpath()
        for bundle in (suite_dir, contest_dir, manifest_dir):
            if bundle.exists():
                shutil.rmtree(bundle)
            bundle.mkdir(parents=True)

        episodes = self._episodes(src)
        stages: list[dict] = []
        for name in episodes.items():
            scenario = episodes.import_item(name, suite_dir / "worlds")
            stages.append(stage(name, scenario.parents[1].name, episodes.episode(name).max_time))

        _dump(suite_dir / "suite.yaml", {"launch": {"human": HUMAN}, "metrics": {"max_collisions": 1}, "stages": stages})
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
        return Attribution(
            source="SocNavBench",
            url="https://github.com/CMU-TBD/SocNavBench",
            license="MIT for the episodes, unspecified for the ETH and UCY trajectories",
            authors="Abhijat Biswas, Allan Wang, Gustavo Silvera, Aaron Steinfeld, Henny Admoni",
            citation="A. Biswas, A. Wang, G. Silvera, A. Steinfeld, H. Admoni. SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation. ACM THRI 2022. arXiv:2103.00047",
        )
