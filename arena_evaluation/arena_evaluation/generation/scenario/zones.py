"""Zone roles from zone names: substring rules for the common names, `UNKNOWN` for the rest.

`UNKNOWN` routes a name to review (`zone_roles.yaml`) instead of guessing it.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from typing import Any


class Role(enum.StrEnum):
    WORK = "work"
    BREAK = "break"
    TRANSIT = "transit"
    PRIVATE = "private"
    UNKNOWN = "unknown"


#: Substring rules, first match wins: `nurse_resting_room` must reach BREAK (`resting`) before
#: WORK (`nurse`), and `wc_waiting` must reach BREAK before anything else claims it.
_RULES: tuple[tuple[Role, tuple[str, ...]], ...] = (
    (Role.BREAK, (
        "waiting", "break_room", "resting", "lounge", "cafeteria", "dining",
        "kitchen", "kitchenette", "staff_room", "bathroom", "ensuite", "wc_",
        "restroom", "toilet", "balcony", "living_room", "phone_booth",
    )),
    (Role.PRIVATE, ("bedroom", "master_bed")),
    (Role.WORK, (
        "work", "office", "nurse_station", "reception", "patient_room", "patient_ward",
        "exam", "consultation", "operating", "procedure", "triage", "lab",
        "meeting_room", "conference", "pharmacy", "file_room", "utility", "storage",
        "workspace",
    )),
    (Role.TRANSIT, (
        "hallway", "corridor", "junction", "arm_", "entrance", "entry", "link_",
        "lobby", "destination", "empty_zone",
    )),
)


def classify(name: str) -> Role:
    lowered = name.strip().lower()
    for role, needles in _RULES:
        if any(n in lowered for n in needles):
            return role
    return Role.UNKNOWN


def classify_all(names: Sequence[str], overrides: Mapping[str, str] | None = None) -> dict[str, Role]:
    """`overrides` (the reviewed `zone_roles.yaml` entries) win over the rules."""
    resolved = {n: classify(n) for n in names}
    for name, role in (overrides or {}).items():
        if name in resolved:
            resolved[name] = Role(role)
    return resolved


def zones_by_role(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> dict[Role, list[str]]:
    names = [str(z["name"]) for z in zones if isinstance(z, Mapping) and z.get("name")]
    out: dict[Role, list[str]] = {r: [] for r in Role}
    for name, role in classify_all(names, overrides).items():
        out[role].append(name)
    return {r: sorted(v) for r, v in out.items()}
