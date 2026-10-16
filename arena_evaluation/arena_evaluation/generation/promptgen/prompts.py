"""The frozen prompt sets, and the place and duration phrases the system prompts document.

Each prompt carries its `targets` (the steps it exercises) and an `oracle` (the decisions its slots encode), which `--oracle` runs realise without a model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import attrs
import yaml

PLACE_ROUTE = (("right in the robot's way", 0.5), ("just ahead of the robot", 0.3), ("further down the robot's route", 0.7), ("halfway along the robot's route", 0.5),
               ("right at the start of the robot's route", 0.15), ("just before the robot's goal", 0.85))
DUR_C = (("for a moment", 4), ("for a few seconds", 6), ("for a while", 12), ("for a good half minute", 30), ("for a long time", 30))


@attrs.frozen
class Prompt:
    id: str
    text: str
    targets: tuple[str, ...]
    oracle: dict[str, Any]
    scope: str = "universal"
    world: str = ""


def read(path: Path) -> list[Prompt]:
    raw = yaml.safe_load(path.read_text()) or []
    return [Prompt(id=str(p["id"]), text=str(p["text"]), targets=tuple(p.get("targets") or ()), oracle=dict(p.get("oracle") or {}), scope=str(p.get("scope", "universal")), world=str(p.get("world", ""))) for p in raw]


def shipped_dir() -> Path:
    return Path(__file__).with_name("prompts")


def load_sets(world: str, *, extra: Sequence[Path] = ()) -> list[Prompt]:
    """The frozen universal set plus this world's own, plus any files given."""
    paths = [*sorted(shipped_dir().glob("universal*.yaml")), *(sorted(shipped_dir().glob(f"{world}*.yaml")) if world else []), *extra]
    out: list[Prompt] = []
    for p in paths:
        if p.is_file():
            out.extend(read(p))
    return out
