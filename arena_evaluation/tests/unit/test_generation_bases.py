from __future__ import annotations

import math

from arena_evaluation.generation.scenario import (
    MAX_LEGS,
    MIN_EPISODE_S,
    ROBOT_SPEED_MPS,
    build,
    robot_for,
    route_seconds,
)
from arena_evaluation.generation.scenario.populate import (
    MAX_VELOCITY_DEFAULT,
    MIN_PED_ROUTE_M,
    MIN_WALKERS,
    PEOPLE_PER_ZONE,
    TALKERS,
    cast_split,
)


def zone(name, x0, y0, x1, y1, doors=()):
    return {
        "name": name,
        "corners": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}],
        "doors": list(doors),
    }


def door(name, x0, y0, x1, y1):
    return {"name": name, "start": {"x": x0, "y": y0, "z": 0}, "end": {"x": x1, "y": y1, "z": 0}}


# A building: a long east-west corridor with rooms above and below it, and a second corridor.
ZONES = [
    zone("main_hallway", 0.0, 14.0, 40.0, 18.0, doors=[door("office_1_door", 10.0, 14.0, 11.5, 14.0)]),
    zone("side_hallway", 42.0, 0.0, 46.0, 30.0),
    zone("office_1", 4.0, 4.0, 14.0, 12.0),
    zone("office_2", 18.0, 4.0, 28.0, 12.0),
    zone("meeting_room_1", 4.0, 20.0, 14.0, 28.0),
    zone("kitchen", 18.0, 20.0, 28.0, 28.0),
    zone("reception", 30.0, 4.0, 40.0, 12.0),
]
ROBOT = {"start": [2.0, 16.0, 0.0], "goal": [38.0, 16.0, 0.0]}
HOMES = frozenset(z["name"] for z in ZONES if z["name"] not in ("main_hallway", "side_hallway"))


def _pop(profile="normal", **kw):
    return build(profile, zones=ZONES, robot=ROBOT, seed=1, homes=HOMES, world="w", **kw)["dynamic"]


def _talkers(agents):
    return [a for a in agents if "talker" in a["agent"]["agent_type"]]


def test_the_shipped_route_is_kept_and_made_long_enough() -> None:
    spec = robot_for(ROBOT, ZONES)
    assert spec["start"][:2] == [2.0, 16.0] and "phases" in spec and "goal" not in spec
    assert route_seconds(spec, ZONES) >= MIN_EPISODE_S
    short = {"start": [2.0, 16.0, 0.0], "goal": [6.0, 16.0, 0.0]}
    spec = robot_for(short, ZONES)
    assert MIN_EPISODE_S <= route_seconds(spec, ZONES) or len(spec["phases"]) == MAX_LEGS


def test_the_floor_holds_for_the_faster_controller() -> None:
    assert ROBOT_SPEED_MPS == 0.5


def test_cast_split_keeps_walkers_and_whole_pairs() -> None:
    assert cast_split(3) == (0, 3) and cast_split(4) == (2, 2)  # a pair as soon as two walkers remain
    assert cast_split(7) == (2, 5)
    assert cast_split(14) == (4, 10)
    for n in range(1, 40):
        talkers, walkers = cast_split(n)
        assert talkers % TALKERS == 0 and talkers + walkers == n
        assert walkers >= min(n, MIN_WALKERS)


def test_a_normal_base_is_two_per_work_zone_with_pairs_in_rooms() -> None:
    agents = _pop()
    work = [z for z in ZONES if z["name"] in ("office_1", "office_2", "reception", "meeting_room_1")]
    assert PEOPLE_PER_ZONE * len(work) * 0.8 <= len(agents) <= PEOPLE_PER_ZONE * len(work) * 1.3 + 1
    talkers = _talkers(agents)
    assert talkers and len(talkers) % TALKERS == 0
    rooms = {}
    for a in talkers:
        rooms.setdefault(a["pose"], 0)
        rooms[a["pose"]] += 1
    assert all(count == TALKERS for count in rooms.values()), rooms  # one pair per room
    assert len(agents) - len(talkers) >= MIN_WALKERS
    assert len({a["pose"] for a in agents}) >= 3, "the cast is spread over rooms"


def test_night_is_half_and_still_walks() -> None:
    night = _pop("night")
    assert len(night) < len(_pop())
    assert len(night) - len(_talkers(night)) >= MIN_WALKERS


def test_a_fast_walker_is_hurried_and_carries_no_cap() -> None:
    agents = [a for seed in range(6) for p in ("normal", "break", "night") for a in build(p, zones=ZONES, robot=ROBOT, seed=seed, homes=HOMES, world="w")["dynamic"]]
    fast = [a for a in agents if a["agent"]["desired_velocity"] > MAX_VELOCITY_DEFAULT]
    assert fast, "the mix has hurried walkers"
    assert all(a["agent"]["agent_type"] == "hurried" for a in fast)
    assert all("max_velocity" not in a["agent"] for a in agents)


def test_pedestrians_walk_at_least_the_route_floor_where_the_world_allows() -> None:
    centroid = {z["name"]: ((z["corners"][0]["x"] + z["corners"][2]["x"]) / 2, (z["corners"][0]["y"] + z["corners"][2]["y"]) / 2) for z in ZONES}

    def length(agent):
        pts = [centroid[w] if isinstance(w, str) else (w[0], w[1]) for w in agent["waypoints"]]
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))

    walks = [length(a) for a in _pop() if "talker" not in a["agent"]["agent_type"]]
    assert walks and min(walks) >= MIN_PED_ROUTE_M * 0.8


def test_no_base_carries_regions_and_empty_is_empty() -> None:
    assert _pop("empty") == []
    for profile in ("normal", "break", "night", "empty"):
        assert "regions" not in build(profile, zones=ZONES, robot=ROBOT, seed=0, world="w")


def test_nobody_is_placed_in_or_sent_to_a_toilet() -> None:
    from arena_evaluation.generation.scenario.populate import extra_agents, for_people

    zones = [*ZONES, zone("wc_1", 30.0, 20.0, 34.0, 24.0), zone("storage", 36.0, 20.0, 40.0, 24.0)]
    homes = HOMES | {"wc_1", "storage"}
    agents = build("break", zones=zones, robot=ROBOT, seed=1, homes=homes, world="w")["dynamic"]
    agents += extra_agents(6, zones=zones, robot=ROBOT, homes=homes, seed=1, world="w")
    assert agents
    for a in agents:
        assert a["pose"] not in ("wc_1", "storage")
        assert not {"wc_1", "storage"} & set(a["waypoints"])
    assert for_people(["kitchen", "wc_2", "Toilet_A", "bathroom", "storage_1", "closet"]) == ["kitchen"]
