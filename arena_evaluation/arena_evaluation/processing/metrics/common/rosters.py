from __future__ import annotations

import re
import typing
from collections import defaultdict

import polars as pl

_ENV_PREFIX: typing.Final[re.Pattern[str]] = re.compile(r"^env_\d+/")
_WORLD_OBJECT_PREFIX: typing.Final[str] = "world_"


def strip_env(name: str) -> str:
    """Drop the leading `env_<n>/` segment, keeping the rest incl. any level suffix."""
    return _ENV_PREFIX.sub("", name, count=1)


def entity_roster(snapshot: pl.DataFrame | None) -> dict[str, str]:
    """Bare entity name -> recorded sim_path, for names resolving to a single sim_path, a world entity also by its name without the level suffix or the world object prefix."""
    if snapshot is None or len(snapshot) == 0:
        return {}
    paths: dict[str, set[str]] = defaultdict(set)
    for entity in snapshot["entity_name"].unique().to_list():
        paths[strip_env(entity)].add(entity)
    unsuffixed: dict[str, set[str]] = defaultdict(set)
    for bare, recorded in paths.items():
        if "/" in bare:
            unsuffixed[bare.rsplit("/", 1)[0]] |= recorded
        if bare.startswith(_WORLD_OBJECT_PREFIX):
            unsuffixed[bare.removeprefix(_WORLD_OBJECT_PREFIX)] |= recorded
    roster = {name: next(iter(p)) for name, p in unsuffixed.items() if name not in paths and len(p) == 1}
    return roster | {bare: next(iter(p)) for bare, p in paths.items() if len(p) == 1}


def ped_aliases(name: str) -> list[str]:
    """Names an atom may use for a recorded ped: env-stripped, plus without a level suffix on multi-level worlds."""
    bare = strip_env(name)
    return [bare, bare.rsplit("/", 1)[0]] if "/" in bare else [bare]


def ped_roster(peds: pl.DataFrame | None) -> dict[str, str]:
    """Bare ped name -> recorded name, only for names that resolve uniquely on the axis."""
    if peds is None or len(peds) == 0:
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    for name in peds["ped_name"].unique().to_list():
        for alias in ped_aliases(name):
            grouped[alias].add(name)
    return {bare: next(iter(full)) for bare, full in grouped.items() if len(full) == 1}
