"""Pedestrian profiles and the parameter tiers, the only numbers a model may name (`speed: low`).

A step that tunes somebody writes a derived agent type beside the scenario (`derive`) rather than editing the library."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from ..scenario.populate import KINEMATICS, MODELS

TIER_NAMES = ("low", "mid", "high")

#: Adult tier values. `speed` scales by the profile's own nominal (a child's "high" is a child
#: running, not an adult running); everything else is absolute.
TIERS: dict[str, tuple[float, float, float]] = {
    "speed": (0.6, 1.1, 1.7),
    #: The engine clamps to `max_velocity` (1.5 +- 0.1 by default): a high `speed` alone is a
    #: brisk walk. Pinned alongside it so "running" runs.
    "max_speed": (1.5, 1.8, 2.6),
    "vision_range": (2.0, 5.0, 8.0),
    "vision_fov": (90.0, 180.0, 270.0),
    "repulsion": (0.5, 2.1, 4.0),
    "repulsion_range": (0.1, 0.3, 0.6),
    "radius": (0.18, 0.25, 0.40),
    "reaction": (0.2, 0.5, 1.0),
    "need_decay": (0.3, 1.0, 3.0),
    "duration": (10.0, 30.0, 1.0e4),
}

#: Where each parameter lives on an agent type. `*` matches every key at that level.
PATHS: dict[str, tuple[str, ...]] = {
    "speed": ("desired_velocity",),
    "max_speed": ("max_velocity",),
    "vision_range": ("perception", "vision_range"),
    "vision_fov": ("perception", "vision_fov"),
    "repulsion": ("local_planner_params", "repulsion_strength"),
    "repulsion_range": ("local_planner_params", "repulsion_range"),
    "radius": ("agent_radius",),
    "reaction": ("local_planner_params", "relaxation_time"),
    "need_decay": ("needs", "*", "decay_rate"),
    "duration": ("sequences", "*", "steps", "*", "duration"),
}

#: Parameters that only exist on behaviour-tree types.
BT_ONLY = frozenset({"need_decay", "duration"})

ADULT_SPEED = KINEMATICS["adult"][0]

_MEDICAL = ("female_adult_medical_01", "male_adult_medical_01")
_CIVIL = tuple(m for m in MODELS if "medical" not in m)


@attrs.frozen(kw_only=True)
class Profile:
    name: str
    #: Builtin agent type this profile is made of.
    base: str
    meshes: tuple[str, ...]
    #: Multiplier on the adult speed tiers, so tiers read in this profile's own terms.
    speed_scale: float = 1.0
    note: str = ""


PROFILES: dict[str, Profile] = {
    "adult": Profile(name="adult", base="adult", meshes=_CIVIL, note="an adult walking normally"),
    "child": Profile(name="child", base="child", meshes=_CIVIL, speed_scale=KINEMATICS["child"][0] / ADULT_SPEED, note="small, quick, impulsive, short-sighted"),
    "elder": Profile(name="elder", base="elder", meshes=_CIVIL, speed_scale=KINEMATICS["elder"][0] / ADULT_SPEED, note="slow, wide turns, slow to react"),
    "hurried": Profile(name="hurried", base="hurried", meshes=_CIVIL, speed_scale=KINEMATICS["hurried"][0] / ADULT_SPEED, note="in a rush"),
    "distracted": Profile(name="distracted", base="distracted", meshes=_CIVIL, speed_scale=KINEMATICS["distracted"][0] / ADULT_SPEED, note="absent-minded: narrow attention, reacts late"),
    "nurse": Profile(name="nurse", base="adult", meshes=_MEDICAL, speed_scale=1.1, note="staff on their rounds"),
    "doctor": Profile(name="doctor", base="adult", meshes=_MEDICAL, speed_scale=1.05, note="staff between rooms"),
}

#: Words a prompt may use for a profile.
ALIASES: dict[str, str] = {
    "kid": "child",
    "kids": "child",
    "children": "child",
    "boy": "child",
    "girl": "child",
    "toddler": "child",
    "old man": "elder",
    "old woman": "elder",
    "old lady": "elder",
    "elderly": "elder",
    "pensioner": "elder",
    "grandmother": "elder",
    "grandfather": "elder",
    "senior": "elder",
    "rushing": "hurried",
    "in a hurry": "hurried",
    "late": "hurried",
    "runner": "hurried",
    "absent-minded": "distracted",
    "nurses": "nurse",
    "medic": "nurse",
    "doctors": "doctor",
    "surgeon": "doctor",
    "physician": "doctor",
    "person": "adult",
    "people": "adult",
    "someone": "adult",
    "man": "adult",
    "woman": "adult",
    "visitor": "adult",
    "colleague": "adult",
    "worker": "adult",
}


class ProfileError(ValueError):
    pass


def profile(name: str) -> Profile:
    key = str(name or "").strip().lower()
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        raise ProfileError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[key]


def tier_value(param: str, tier: str, *, prof: Profile | None = None) -> float:
    if param not in TIERS:
        raise ProfileError(f"unknown parameter {param!r}; allowed: {sorted(TIERS)}")
    key = str(tier).strip().lower()
    if key not in TIER_NAMES:
        raise ProfileError(f"{param}: tier {tier!r} is not one of {list(TIER_NAMES)}")
    value = TIERS[param][TIER_NAMES.index(key)]
    if param == "speed" and prof is not None:
        value *= prof.speed_scale
    return round(value, 3)


def point_mass(value: float) -> dict[str, float]:
    return {"mean": value, "std": 0.0, "clip_low": value, "clip_high": value}


def _set_path(doc: dict[str, Any], keys: Sequence[str], value: dict[str, float]) -> int:
    """Write `value` at every match of `keys`; returns how many fields were written."""
    if len(keys) == 1:
        doc[keys[0]] = dict(value)
        return 1
    head, rest = keys[0], keys[1:]
    if head == "*":
        return sum(_set_path(child, rest, value) for child in doc.values() if isinstance(child, dict))
    child = doc.get(head)
    if not isinstance(child, dict):
        if rest and rest[0] == "*":
            return 0
        child = doc[head] = {}
    return _set_path(child, rest, value)


def is_behaviour_tree(raw: Mapping[str, Any]) -> bool:
    return raw.get("mode") == "behavior_tree" and bool(raw.get("sequences"))


def derive(raw: Mapping[str, Any], params: Mapping[str, str], *, name: str, prof: Profile | None = None) -> dict[str, Any]:
    """`raw` (`{"extends": "<builtin>"}` or a full scenario-local type) with every tuned parameter pinned to its tier.
    A behaviour-tree-only parameter on a plain type raises rather than being written into nothing."""
    doc = copy.deepcopy(dict(raw))
    doc["name"] = name
    for param, tier in params.items():
        value = tier_value(param, tier, prof=prof)
        if param in BT_ONLY and not is_behaviour_tree(doc):
            raise ProfileError(f"{param} needs a behaviour-tree type; {doc.get('extends') or doc.get('name')!r} is not one")
        written = _set_path(doc, PATHS[param], point_mass(value))
        if written == 0:
            raise ProfileError(f"{param}: nothing at {'.'.join(PATHS[param])} on {doc.get('name')!r}")
    return doc


def derived_name(base: str, params: Mapping[str, str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", base.rsplit("/", 1)[-1].removesuffix(".yaml")).strip("_")
    tail = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"{stem}__{tail}" if tail else stem


def scaled_speed(prof: Profile, tier: str) -> float:
    return tier_value("speed", tier, prof=prof)
