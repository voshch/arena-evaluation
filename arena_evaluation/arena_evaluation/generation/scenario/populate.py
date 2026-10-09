"""The four base scenarios for a world: `normal`, `break`, `night` and `empty`.

Every pedestrian is written in zone names (`pose: laboratory`) that resolve to a free point at run time;
talkers come in pairs, everyone else walks a route through other rooms.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .zones import Role, zones_by_role

#: Behaviour-tree agent type copied beside each generated scenario: the builtin types are all
#: `mode: simple`, which a behaviour-tree case cannot act on.
BT_AGENT_TYPE = "talker.yaml"

#: Talkers come in pairs, one pair per room: a talker with nobody in range waits out its patience
#: and stands there anyway. A behaviour-tree agent ignores its waypoints, so only a share of the cast talks.
TALKERS = 2
TALKER_SHARE = 0.3
MIN_WALKERS = 2

#: People per target zone, before the profile's `count` and the world's crowd factor.
PEOPLE_PER_ZONE = 2
MIN_CAST = 4

#: Metres between a home zone and the robot's route, for `zone_screen` to pick homes; nothing here measures it.
MIN_ROUTE_CLEARANCE = 1.5

#: (zones a walker visits besides its home, weight)
ROUTE_STOPS: tuple[tuple[int, float], ...] = ((1, 0.34), (2, 0.30), (3, 0.22), (4, 0.14))

#: Shortest one-way walker route, metres between zone centroids, where the world has the zones for it:
#: about 30 s there and back at 1 m/s.
MIN_PED_ROUTE_M = 15.0

#: `desired_velocity` (mean, std, clip_low, clip_high) per kinematic type, mirrored from humansim's
#: `config/agent_types/*.yaml` (a test refuses drift) because generation runs without ROS on the path.
KINEMATICS: dict[str, tuple[float, float, float, float]] = {
    "adult": (1.1, 0.12, 0.5, 1.5),
    "elder": (0.75, 0.12, 0.3, 1.1),
    "child": (1.0, 0.25, 0.4, 1.8),
    "hurried": (1.7, 0.15, 1.3, 2.2),
    "distracted": (0.95, 0.12, 0.5, 1.3),
}

#: Relative type weights per profile; `ONLY_IN` types are dropped before drawing.
MIX: dict[str, dict[str, int]] = {
    "normal": {"adult": 55, "hurried": 15, "distracted": 15, "elder": 10, "child": 5},
    "break": {"adult": 50, "distracted": 30, "elder": 10, "child": 10},
    "night": {"adult": 50, "elder": 30, "distracted": 10, "hurried": 10},
}

#: Type weights for the walkers `extra_agents` adds in the `busy` case.
BUSY_MIX: dict[str, int] = {"adult": 50, "hurried": 30, "distracted": 15, "elder": 5}

#: Types restricted to these kinds of building: a child on a hospital ward is the first thing a reviewer flags.
ONLY_IN: dict[str, tuple[str, ...]] = {
    "child": ("residential",),
}

#: Zone-name fragments of rooms nobody is placed in or sent to. Walkers spawned in hospital_1's `wc_waiting`
#: never left: its 0.7 m stall doors are sealed in HumanSim's inflated grid (2026-08-28).
NOT_FOR_PEOPLE = ("wc", "toilet", "bath", "restroom", "lavatory", "stall", "storage", "closet", "elevator")


def for_people(names: Sequence[str]) -> list[str]:
    return [n for n in names if not any(h in str(n).lower() for h in NOT_FOR_PEOPLE)]


_HOSPITAL_HINTS = ("patient", "nurse", "ward", "exam", "pharmacy", "operating", "triage", "consult")

#: Drawn once per world, so two worlds generated from one seed are not the same crowd in different rooms.
TEMPO_RANGE = (0.92, 1.08)
CROWD_RANGE = (0.8, 1.3)

#: `count` scales the world's nominal population, `speed` the walkers' speed.
PROFILES: dict[str, dict[str, float]] = {
    "normal": {"count": 1.0, "speed": 1.0},
    "break": {"count": 1.0, "speed": 1.0},
    "night": {"count": 0.5, "speed": 0.8},
    "empty": {"count": 0.0, "speed": 1.0},
}

#: HumanSim's default `max_velocity` mean (`config/agent_types`); `hurried` has its own 2.4 m/s cap.
MAX_VELOCITY_DEFAULT = 1.5

#: Meshes in shipped use: an unresolved model silently falls back to a default.
MODELS: tuple[str, ...] = (
    "female_adult_medical_01",
    "male_adult_medical_01",
    "male_adult_construction_01",
    "female_adult_business_02",
    "male_adult_construction_02",
    "female_adult_police_01",
)


def _polygon(zone: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(c["x"]), float(c["y"])) for c in zone.get("corners", []) if isinstance(c, Mapping)]


def _centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def _dist_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay, bx, by = *a, *b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def _endpoints(robot: Mapping[str, Any]) -> list[Any]:
    out: list[Any] = []
    if (start := robot.get("start")) is not None:
        out.append(start)
    for phase in robot.get("phases") or []:
        if isinstance(phase, Mapping) and (goto := phase.get("goto")) is not None:
            out.append(goto)
    if (goal := robot.get("goal")) is not None:
        out.append(goal)
    return out


def robot_zones(robot: Mapping[str, Any]) -> frozenset[str]:
    """Zones the robot spec names as endpoints; the robot may resolve anywhere inside them, so nobody starts there."""
    return frozenset(e for e in _endpoints(robot) if isinstance(e, str))


def route_points(
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[float, float]]:
    """Start, phase goals and goal as a polyline. Zone-named endpoints resolve to centroids only when
    `zones` is given and are dropped otherwise, which leaves an empty route for zone-named robots."""
    centroids = {str(z["name"]): _centroid(_polygon(z)) for z in (zones or []) if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3}
    pts: list[tuple[float, float]] = []
    for endpoint in _endpoints(robot):
        if isinstance(endpoint, str):
            if (centre := centroids.get(endpoint)) is not None:
                pts.append(centre)
        elif isinstance(endpoint, Sequence):
            pts.append((float(endpoint[0]), float(endpoint[1])))
    return pts


def clearance(pt: tuple[float, float], route: Sequence[tuple[float, float]]) -> float:
    if len(route) < 2:
        return math.dist(pt, route[0]) if route else float("inf")
    return min(_dist_to_segment(pt, route[i], route[i + 1]) for i in range(len(route) - 1))


def nominal_count(zones_of_role: Sequence[str]) -> int:
    return max(1, PEOPLE_PER_ZONE * len(zones_of_role))


def cast_split(n: int) -> tuple[int, int]:
    """(talkers, walkers) for a cast of `n`: `TALKER_SHARE` in whole pairs, at least one, dropping
    pairs until `MIN_WALKERS` walk."""
    pairs = max(1, int(n * TALKER_SHARE) // TALKERS)
    while pairs > 0 and n - pairs * TALKERS < MIN_WALKERS:
        pairs -= 1
    talkers = pairs * TALKERS
    return talkers, n - talkers


def _targets(profile: str, by_role: dict[Role, list[str]]) -> tuple[list[str], str]:
    """Zones this profile populates, and the basis. A world with no work zones reads as a home: living
    areas are where people are, private rooms where they go on `break`."""
    work, brk, private = by_role[Role.WORK], by_role[Role.BREAK], by_role[Role.PRIVATE]
    if work:
        return (brk or work, "workplace") if profile == "break" else (work, "workplace")

    if brk or private:
        return (private or brk, "residential") if profile == "break" else (brk or private, "residential")

    return by_role[Role.TRANSIT], "transit-only"


def _route(
    home: str,
    pool: Sequence[str],
    index: int,
    stops: int,
    centroids: Mapping[str, tuple[float, float]],
) -> list[str]:
    """Up to `stops` zones besides `home`, more while the walk is under `MIN_PED_ROUTE_M`, taken
    round-robin from `index` so neighbouring agents get different itineraries. Empty in a one-zone world."""
    others = [z for z in pool if z != home and z in centroids]
    out: list[str] = []
    length, last = 0.0, centroids.get(home)
    for k in range(len(others)):
        if k >= stops and length >= MIN_PED_ROUTE_M:
            break
        zone = others[(index + k) % len(others)]
        if last is not None:
            length += math.dist(last, centroids[zone])
        last = centroids[zone]
        out.append(zone)
    return out


def _waypoint_mode(legs: int, rng: random.Random) -> str:
    """`reverse` for two points; from three, `repeat`, `reverse` or `random`. Never `once`: it stops
    at the last waypoint and the agent is furniture for the rest of the episode."""
    if legs < 3:
        return "reverse"
    return rng.choices(("repeat", "reverse", "random"), weights=(0.5, 0.3, 0.2))[0]


def build(
    profile: str,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    seed: int = 0,
    homes: frozenset[str] = frozenset(),
    world: str = "",
) -> dict[str, Any]:
    """Deterministic in its inputs, `world` among them. `homes` are the zones `zone_screen` cleared as
    starts; empty means unscreened, and the target zones minus the robot's own serve instead."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")

    doc: dict[str, Any] = {"robots": [dict(robot)], "dynamic": []}
    spec = PROFILES[profile]
    if spec["count"] == 0.0:
        return doc

    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets(profile, by_role)
    if not targets:
        return doc

    centroids = {str(z["name"]): _centroid(_polygon(z)) for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3}
    route = route_points(robot, zones)
    forbidden = robot_zones(robot)

    # fewer than three screened target rooms would stack the crowd, so other screened rooms join (hospital_1: two of six work zones pass)
    starts = for_people([z for z in targets if z in homes and z in centroids])
    if len(starts) < 3 and homes:
        starts += for_people([z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in homes and z in centroids and z not in starts and z not in forbidden])
    starts = starts or for_people([z for z in targets if z not in forbidden and z in centroids]) or for_people([z for z in targets if z in centroids]) or [z for z in targets if z in centroids]
    if not starts:
        return doc

    # the robot's own zones are fine as destinations: the agent arrives later
    destinations = for_people([z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in centroids]) or [z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in centroids]

    rng = random.Random(f"{world}:{profile}:{seed}")
    world_rng = random.Random(f"{world}:{seed}")
    tempo = world_rng.uniform(*TEMPO_RANGE)
    crowd = world_rng.uniform(*CROWD_RANGE)
    models = list(MODELS)
    world_rng.shuffle(models)

    n = max(MIN_CAST, round(nominal_count(targets) * spec["count"] * crowd))
    talkers, _walkers = cast_split(n)
    talker_rooms = list(starts)
    rng_rooms = random.Random(f"{world}:{profile}:{seed}:talkers")
    rng_rooms.shuffle(talker_rooms)

    kind_of_world = world_kind(zones, overrides)
    weights = mix_for(profile, kind_of_world)
    types, type_weights = list(weights), list(weights.values())

    agents: list[dict[str, Any]] = []
    for i in range(n):
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]

        if i < talkers:
            home = talker_rooms[(i // TALKERS) % len(talker_rooms)]
            kind = f"./{BT_AGENT_TYPE}"
            speed = speed_for("adult", rng, spec["speed"] * tempo)
            # the behaviour tree drives a talker; two entries because the backend appends the spawn pose to a one-entry list
            waypoints: list[str] = [home, home]
        else:
            home = starts[i % len(starts)]
            kind = rng.choices(types, weights=type_weights)[0]
            speed = speed_for(kind, rng, spec["speed"] * tempo)
            legs = _route(home, destinations, i, stops, centroids)
            waypoints = [home, *legs] if legs else [home, home]

        agent: dict[str, Any] = {"agent_type": kind, "desired_velocity": speed}
        agents.append(
            {
                "name": f"{profile}_{i + 1}",
                "model": models[i % len(models)],
                "pose": home,
                "agent": agent,
                "waypoints": waypoints,
                "waypoint_mode": _waypoint_mode(len(waypoints), rng),
            }
        )

    # a prompt about "the elderly" must match someone: 25 scenarios were dropped as `matched nobody` without this
    walkers = [a for a in agents if not str(a["agent"]["agent_type"]).startswith("./")]
    if walkers and all(a["agent"]["agent_type"] != "elder" for a in walkers):
        last = walkers[-1]
        last["agent"] = {"agent_type": "elder", "desired_velocity": speed_for("elder", random.Random(f"{world}:{profile}:{seed}:elder"), spec["speed"] * tempo)}

    doc["dynamic"] = agents
    return doc


def bt_type_source() -> Path:
    return Path(__file__).resolve().parent / "agent_types" / BT_AGENT_TYPE


def basis_for(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    """`workplace`, `residential` or `transit-only`: the mapping `build` uses for this world."""
    return _targets("normal", zones_by_role(zones, overrides))[1]


def work_zones(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> list[str]:
    """The zones `normal` populates, residential shift included, so "back to work" in a case means what it meant here."""
    return list(_targets("normal", zones_by_role(zones, overrides))[0])


def world_kind(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    """`hospital`, `residential` or `office` from the zone names; hospital wins (a care home has wards and bedrooms)."""
    names = [str(z["name"]).lower() for z in zones if isinstance(z, Mapping) and z.get("name")]
    if any(hint in n for n in names for hint in _HOSPITAL_HINTS):
        return "hospital"
    if basis_for(zones, overrides) == "residential":
        return "residential"
    return "office"


def mix_for(profile: str, kind: str) -> dict[str, int]:
    weights = MIX.get(profile, {"adult": 1})
    allowed = {t: w for t, w in weights.items() if kind in ONLY_IN.get(t, (kind,))}
    return allowed or {"adult": 1}


def speed_for(kind: str, rng: random.Random, scale: float = 1.0) -> float:
    mean, std, lo, hi = KINEMATICS[kind]
    raw = rng.gauss(mean, std) * scale
    return round(min(max(raw, lo), hi), 2)


def extra_agents(
    n: int,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    homes: frozenset[str] = frozenset(),
    seed: int = 0,
    world: str = "",
    mix: Mapping[str, int] = BUSY_MIX,
    prefix: str = "busy",
    speed_scale: float = 1.0,
    overrides: Mapping[str, str] | None = None,
    existing: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """`n` more walkers for an existing scenario (the `busy` case), homed in the `normal` targets and routed
    like `build`'s walkers. Names continue from `<prefix>_1`, skipping any `existing` already uses."""
    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets("normal", by_role)
    centroids = {str(z["name"]): _centroid(_polygon(z)) for z in zones if isinstance(z, Mapping) and z.get("name") and len(_polygon(z)) >= 3}
    forbidden = robot_zones(robot)
    starts = for_people([z for z in targets if z in homes and z in centroids]) or [z for z in targets if z not in forbidden and z in centroids] or [z for z in targets if z in centroids]
    if not starts or n <= 0:
        return []
    destinations = for_people([z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in centroids])
    rng = random.Random(f"{world}:{prefix}:{seed}")
    models = list(MODELS)
    rng.shuffle(models)
    weights = mix_for("normal", world_kind(zones, overrides)) if not mix else {t: w for t, w in mix.items() if world_kind(zones, overrides) in ONLY_IN.get(t, (world_kind(zones, overrides),))}
    weights = weights or {"adult": 1}
    types, type_weights = list(weights), list(weights.values())
    taken = {str(a.get("name")) for a in ((existing or {}).get("dynamic") or []) if isinstance(a, Mapping)}
    agents: list[dict[str, Any]] = []
    i = 0
    while len(agents) < n:
        i += 1
        name = f"{prefix}_{i}"
        if name in taken:
            continue
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]
        home = starts[i % len(starts)]
        kind = rng.choices(types, weights=type_weights)[0]
        legs = _route(home, destinations, i, stops, centroids)
        waypoints = [home, *legs] if legs else [home, home]
        agents.append(
            {
                "name": name,
                "model": models[i % len(models)],
                "pose": home,
                "agent": {"agent_type": kind, "desired_velocity": speed_for(kind, rng, speed_scale)},
                "waypoints": waypoints,
                "waypoint_mode": _waypoint_mode(len(waypoints), rng),
            }
        )
    return agents
