"""Scenario directories as a benchmark suite: one `edge_case` stage per (world, scenario)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .scenario.materialize import is_generated


def scenarios(world_root: Path, match: Sequence[str] = ()) -> list[str]:
    """Scenarios matching any glob in `match`; without one, every scenario the generator wrote."""
    paths = sorted((world_root / "scenarios").glob("*/scenario.yaml"))
    if match:
        return [p.parent.name for p in paths if any(fnmatch(p.parent.name, m) for m in match)]
    return [p.parent.name for p in paths if is_generated(p)]


def build(worlds: Mapping[str, Sequence[str]], *, robot: str, episodes: int, seed: int, timeout: str) -> dict[str, Any]:
    # one seed for every stage, so each contestant meets the same episodes and a stage is the pairing key
    return {
        "launch": {"human": "arena"},
        "stages": [
            {
                "name": f"{world}.{scenario}",
                "map": world,
                "robot": robot,
                "episodes": episodes,
                "seed": seed,
                "timeout": timeout,
                "tm_robots": "edge_case",
                "tm_obstacles": "edge_case",
                "config": {"scenario": {"file": scenario}},
            }
            for world, names in worlds.items()
            for scenario in names
        ],
    }
